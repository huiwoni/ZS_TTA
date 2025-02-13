from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import math
import torch.nn as nn

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from data.cls_to_names import *


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.0):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = F.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()

## following APE.
def important_channel_indice(args, model, only_use_txt=True):
    if only_use_txt or args.shot  == 0:
        feats = model.text_feat.unsqueeze(1)  ## C * 1 * D
    else:
        feats = model.fixed_global_feat_vanilla ## C * L * D, including text feat & few shot image feat.
    cate_num, samp_num, feat_dim = feats.shape

    sim_sum = torch.zeros((feat_dim)).to(feats.device)
    count = 0
    # ipdb.set_trace()
    for i in range(cate_num):
        for j in range(cate_num):
            for m in range(samp_num):
                for n in range(samp_num):
                    if i != j:
                        sim_sum += feats[i, m, :] * feats[j, n, :]
                        count += 1
    sim = sim_sum / count
    # ipdb.set_trace()
    criterion = (-1) * args.lambda_ape * sim + (1-args.lambda_ape) * torch.var(model.text_feat, dim=0)
    _, indices = torch.topk(criterion, k=args.num_important_channel)
    return indices


def select_confident_samples(prob, top):
    # ipdb.set_trace()
    batch_entropy = -(prob * torch.log(prob + 1e-6)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)] ## pick the min entropy
    idx_confused = torch.argsort(batch_entropy, descending=False)[int(batch_entropy.size()[0] * top):] ## pick the max entropy
    return prob[idx], idx, prob[idx_confused], idx_confused

def avg_entropy(outputs):
    ## N*Class
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True) # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


## the main component.
class DualMem(nn.Module):
    def __init__(self, args=None, beta=5.5, feat_dim=1024, class_num=1000, mapping='bias'):
        super(DualMem, self).__init__()
        self.args =  args
        self.indice = args.indice  ## indice of important channels.
        self.beta = beta
        self.rank = 4
        self.init_pred = 0
        if args.shared_param:
            self.global_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.
            self.global_bias_key = self.global_bias
            self.global_bias_value = self.global_bias

            self.global_ffn_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_ffn_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.
            self.text_affine = self.global_ffn_affine
            self.text_bias = self.global_ffn_bias
        else:
            self.global_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.
            self.global_bias_key = nn.Parameter(torch.zeros((class_num, feat_dim)))  ## unknown use the category mean.
            self.global_bias_value = nn.Parameter(torch.zeros((class_num, feat_dim)))  ## unknown use the category mean.

            self.global_ffn_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.global_ffn_bias = nn.Parameter(torch.zeros((class_num, feat_dim))) ## unknown use the category mean.
            self.text_affine = nn.Parameter(torch.zeros((feat_dim, feat_dim)))
            self.text_bias = nn.Parameter(torch.zeros((class_num, feat_dim)))
        self.learnable_mapping = args.mapping ### bias | affine | all


    def update_memory_bank(self, model, target):
        # updating 
        mean_prob = self.init_pred[0]  
        value, indice = mean_prob.max(0)
        pseudo_label = indice.item()
        # print(value, indice, target)
        text_features = model.text_feat[pseudo_label]  ## 512
        selected_image_features_global = model.image_features_global[:1]
        current_instance_entropy = -(mean_prob * (torch.log(mean_prob + 1e-8))).sum()
        if model.image_feature_count[pseudo_label] == model.memory_size:
            ###### if the new one is low entropy, find the sample with the max entropy, and replace it with the new one
            if (current_instance_entropy < model.image_entropy_mem[pseudo_label]).sum() == 0:
                pass  ## the entropy of current test image is very large.
            else:
                _, indice = torch.sort(model.image_entropy_mem[pseudo_label])
                to_replace_indice = indice[-1]  ## with max entropy, ascending.
                model.image_feature_memory[pseudo_label][to_replace_indice] = selected_image_features_global
                model.image_prediction_mem[pseudo_label][to_replace_indice] = mean_prob[0]
                model.image_entropy_mem[pseudo_label][to_replace_indice] = current_instance_entropy
        else:
            model.image_feature_memory[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = selected_image_features_global
            model.image_prediction_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = mean_prob[0]
            model.image_entropy_mem[pseudo_label][model.image_feature_count[pseudo_label, 0].item()] = current_instance_entropy
            model.image_feature_count[pseudo_label] += 1


    def get_image_pred(self, model, return_full=False, return_logit=False):
        ## prediction with dynamic memory.
        img_feat = model.image_features_global[:1]  # 1*1024
        count_image_feat = model.image_feature_count.clone()
        num_class = model.image_feature_memory.shape[0]
        image_classifier = 'similarity_weighted'  ## category_center | entropy_weighted | similarity_weighted
        ### similarity_weighted achieves the best results.
        memorized_image_feat = torch.cat((model.image_feature_memory, model.fixed_global_feat_vanilla), dim=1)  ## 200*11*1024
        if image_classifier == 'entropy_weighted':
            ############## weighted combine the memorized feature as the final classifier.
            merged_entropy = torch.cat((model.image_entropy_mem,  torch.zeros(num_class,1).to(merged_image_feat.device)), dim=1) ## 200*11
            filled_image_feat = (merged_image_feat * (- merged_entropy - math.log(1./ num_class)).unsqueeze(-1)).sum(1)  ## weighting with entropy.
            filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * img_feat @ filled_image_feat.t()
            return logits.softmax(dim=1)
        elif image_classifier == 'category_center':
            ############### assign each feature with equal weights.
            filled_image_feat = memorized_image_feat.sum(1) / (count_image_feat + 1)  ### no zero. 200*1024
            filled_image_feat = filled_image_feat / filled_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * img_feat @ filled_image_feat.t()
            return logits.softmax(dim=1)
        elif image_classifier == 'similarity_weighted':  ## this is an instance adaptative method.
            ## calculate the cos similarity betweeen image feature and memory feature, and then weighted the memorized features according to similarity.
            ###################### 有一些memory 是空的，现在却往里面塞了一个self.global_bias， 这不合理，还要把它继续置空。
            img_feat_mappling = img_feat
            memorized_image_feat_K = memorized_image_feat
            memorized_image_feat_V = memorized_image_feat
            with torch.no_grad():
                if self.args.position == 'query':
                    img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*1024
                elif self.args.position == 'key':
                    memorized_image_feat_K = memorized_image_feat  + self.global_bias_key.unsqueeze(1)  ## class*shot*1024
                elif self.args.position == 'value':
                    memorized_image_feat_V = memorized_image_feat  + self.global_bias_value.unsqueeze(1)  ## class*shot*1024
                elif self.args.position == 'qkv' or self.args.position == 'all':
                    img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*1024
                    memorized_image_feat_K = memorized_image_feat + self.global_bias_key.unsqueeze(1)  ## class*shot*1024
                    memorized_image_feat_V = memorized_image_feat + self.global_bias_value.unsqueeze(1)  ## class*shot*1024
                else:
                    pass
                memorized_image_feat_K = memorized_image_feat_K / memorized_image_feat_K.norm(dim=-1, keepdim=True)
                ## some memorized_image_feat slots are empty before mapping, reseting them to empty.
                memorized_image_feat_K[memorized_image_feat.sum(-1) == 0] = 0
                memorized_image_feat_V = memorized_image_feat_V / memorized_image_feat_V.norm(dim=-1, keepdim=True)
                memorized_image_feat_V[memorized_image_feat.sum(-1) == 0] = 0
                img_feat_mappling = img_feat_mappling / img_feat_mappling.norm(dim=-1, keepdim=True)

            similarity_matrix = (img_feat_mappling * memorized_image_feat_K).sum(-1) ## 200*11  idealy [-1,1], practically [0.1, 0.2]  
            similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))
            ### weighting memoried features with similarity weights. 
            adaptive_image_feat = (memorized_image_feat_V * similarity_matrix.unsqueeze(-1)).sum(1)
            ## torch.Size([1, class, dim])
            adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
            if self.args.position == 'output' or self.args.position == 'all':
                adaptive_image_feat = adaptive_image_feat + self.global_ffn_bias.unsqueeze(0)  ## class*shot*1024

            adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            # adaptive_image_feat: torch.Size([1, 102, 1024])
            # img_feat: torch.Size([1, 1024])
            logits = logit_scale * adaptive_image_feat @ img_feat.unsqueeze(-1)  ## used feat is not update.
            logits = logits[:,:,0]
            return logits.softmax(dim=1)
        else:
            raise NotImplementedError

    def get_image_pred_fewshot_global(self, model, return_full=False, return_logit=False):
        ## prediction with static memory.
        if return_full:
            img_feat = model.image_features_global  # 1*1024
        else:
            img_feat = model.image_features_global[:1, :]  # 1*1024
        num_class = model.image_feature_memory.shape[0]
        memorized_image_feat = model.fixed_global_feat  ## 200*11*1024, few shot samples and text features.
        img_feat_mappling = img_feat
        memorized_image_feat_K = memorized_image_feat
        memorized_image_feat_V = memorized_image_feat

        if self.args.position == 'query':
            img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*1024
        elif self.args.position == 'key':
            memorized_image_feat_K = memorized_image_feat + self.global_bias_key.unsqueeze(1)  ## class*shot*1024
        elif self.args.position == 'value':
            memorized_image_feat_V = memorized_image_feat + self.global_bias_value.unsqueeze(1)  ## class*shot*1024
        elif self.args.position == 'qkv' or self.args.position == 'all':
            img_feat_mappling = img_feat + self.global_bias.mean(0, keepdim=True)  ## N*1024
            memorized_image_feat_K = memorized_image_feat + self.global_bias_key.unsqueeze(1)  ## class*shot*1024
            memorized_image_feat_V = memorized_image_feat + self.global_bias_value.unsqueeze(1)  ## class*shot*1024

        memorized_image_feat_K = memorized_image_feat_K / memorized_image_feat_K.norm(dim=-1, keepdim=True)
        memorized_image_feat_V = memorized_image_feat_V / memorized_image_feat_V.norm(dim=-1, keepdim=True)
        img_feat_mappling = img_feat_mappling / img_feat_mappling.norm(dim=-1, keepdim=True)
        ## calculate the cos similarity betweeen image feature and memory feature, and then weighted the memorized probability.
        ##  200*11*200；
        similarity_matrix = memorized_image_feat_K @ img_feat_mappling.T ## class*shot*Batch
        similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))
        adaptive_image_feat = memorized_image_feat_V.transpose(1,2) @ similarity_matrix ## class * D * batch, 102*1024*204
        adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=1, keepdim=True)
        logit_scale = model.logit_scale.exp()
        adaptive_image_feat = adaptive_image_feat.transpose(0,2).transpose(1,2) ## 204*102*1024
        if self.args.position == 'output' or self.args.position == 'all':
            adaptive_image_feat = adaptive_image_feat + self.global_ffn_bias.unsqueeze(0)  ## class*shot*1024

        adaptive_image_feat = adaptive_image_feat / adaptive_image_feat.norm(dim=-1, keepdim=True)
        # ipdb.set_trace()
        # adaptive_image_feat: 1*102*1024
        # img_feat: 1*1024
        logits = logit_scale * adaptive_image_feat[..., self.args.indice] @ img_feat[..., self.args.indice].unsqueeze(-1) ## memoried features are not updated.
        if return_logit:
            return logits[:,:,0]
        else:
            return logits[:,:,0].softmax(dim=1)

    def get_text_prediction(self, model, return_full=True, return_logit=False):
        logit_scale = model.logit_scale.exp()
        if self.args.position == 'output' or self.args.position == 'all':
            text_feat = model.text_feat + self.text_bias
        else:
            text_feat = model.text_feat
        text_feat = text_feat / text_feat.norm(dim=1, keepdim=True)  ## already filtered with indice.
        img_text_logit = logit_scale * model.image_features_global @ text_feat.t() ## 128*200
        if return_full:
            pass
        else:
            img_text_logit = img_text_logit[:1]
        if return_logit:
            return img_text_logit
        else:
            return img_text_logit.softmax(-1)
        
    def get_weighed_text_prediction(self, model, return_full=True, return_logit=False):
        logit_scale = model.logit_scale.exp()
        # if self.args.position == 'output' or self.args.position == 'all':
        #     text_feat = model.text_feat_full + self.text_bias
        # else:
        #     text_feat = model.text_feat
        text_feat = model.text_feat_full
        text_feat = text_feat / text_feat.norm(dim=2, keepdim=True)  ## already filtered with indice.
        
        similarity_matrix = text_feat @ model.image_features_global[:1].t()  ## 128*200
        similarity_matrix = torch.exp(-self.beta * (-similarity_matrix + 1))

        similarity_matrix = torch.nan_to_num(similarity_matrix, nan=0.0)
        text_feat = torch.nan_to_num(text_feat, nan=0.0)

        weighted_text_features = (text_feat * similarity_matrix).sum(dim=1) ## 200*11  idealy [-1,1], practically [0.1, 0.2]  
        weighted_text_features = weighted_text_features / weighted_text_features.norm(dim=1, keepdim=True)
        weighted_text_features = torch.nan_to_num(weighted_text_features, nan=0.0)

        img_text_logit = logit_scale * model.image_features_global[:1] @ weighted_text_features.t()

        if return_full:
            pass
        else:
            img_text_logit = img_text_logit[:1]
        if return_logit:
            return img_text_logit
        else:
            return img_text_logit.softmax(-1)