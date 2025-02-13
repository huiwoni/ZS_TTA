import argparse
import time

from copy import deepcopy

from PIL import Image
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import os

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models

from clip.fix_clip import dummy_get_fixed_clip

from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset, AugMemAugmenter
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
import ipdb

from typing import Callable

from src.model.dualmem import DualMem, select_confident_samples
from src.utils.init import init_image_memory
from src.loss.loss import entropy

def print_logger(
        old_print: Callable,
        file_name: str,
) -> Callable:
    """Returns a function which calls `old_print` twice, specifying a `file=` on the second call.

    Arguments:
        old_print: The `print` function to call twice.
        file_name: The name to give the log file.
    """

    def log_print(*args, **kwargs):
        old_print(*args, **kwargs)
        with open(file_name, "a") as log_file:
            old_print(*args, file=log_file, **kwargs)

    return log_print

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def get_searched_param(set_id, n_shot, ft):
    if ft:
        if set_id == 'I':
            return [0], [0.3], [0.00001], [100]  
        elif set_id == 'Flower102':
            return [0], [0.3], [0.001], [100]  
        elif set_id == 'DTD':
            return [0], [0.3], [0.0001], [100]  
        elif set_id == 'Pets':
            return [0], [0.3], [0.0001], [20] 
        elif set_id == 'Cars':
            return [0], [0.3], [0.0001], [100] 
        elif set_id == 'UCF101':
            return [0], [0.3], [0.0001], [100] 
        elif set_id == 'Caltech101':
            return [0], [0.3], [0.0001], [20] 
        elif set_id == 'Food101':
            if n_shot >=8:
                return [0], [0.3], [0.0001], [100] 
            else:
                return [0], [0.3], [0.0001], [20] 
        elif set_id == 'SUN397':
            return [0], [0.3], [0.0001], [20] 
        elif set_id == 'Aircraft':
            return [0], [0.3], [0.0001], [100] 
        elif set_id == 'eurosat':
            if n_shot >=8:
                return [0], [0.3], [0.001], [100] 
            else:
                return [0], [0.3], [0.0001], [100] 
        else:
            raise NotImplementedError
    else:
        return [0], [0.3], [0.1], [20]  ## not used.

def main():
    args = parser.parse_args()
    args.log = args.log + '_' + str(args.gpu)
    set_random_seed(args.seed)
    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))


    ################################################################################ create model (zero-shot clip model (ViT-L/14@px336) with promptruning) st
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes

    model = dummy_get_fixed_clip(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init, memory_size=args.memory_size, text_prompt=args.text_prompt)
    model_state = None

    for name, param in model.named_parameters():
        param.requires_grad_(False)
    
    print("=> Model created: visual backbone {}".format(args.arch))
    
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    num_important_channel_list = args.num_important_channel.split("/")
    lambda_ape_list = args.lambda_ape.split("/")
    lr_list = args.lr.split("/")
    epoch_list = args.epoch.split("/")
    results = {}
    print_log = print_logger(print, os.path.join(args.log + '.txt'))
    ################################################################################ create model (zero-shot clip model (ViT-L/14@px336) with promptruning) ed


    for set_id in datasets:
        if args.use_searched_param:
            num_important_channel_list, lambda_ape_list, lr_list, epoch_list = get_searched_param(set_id, args.n_shot, args.ft)
        best_acc = 0
        print_log("processing the dataset{} \n".format(set_id), end="	")
        for num_important_channel in num_important_channel_list:
            for lambda_ape in lambda_ape_list:
                for lr in lr_list:
                    for epoch in epoch_list:
                        print('adopt num_important_channel {}, lambda_ape: {}'.format(num_important_channel, lambda_ape))
                        args.lr = float(lr)
                        args.epoch = int(epoch)
                        args.num_important_channel = int(num_important_channel)
                        args.lambda_ape = float(lambda_ape)


                        ######################################################################################## build dataset st
                        base_transform = transforms.Compose([
                            transforms.Resize(args.resolution, interpolation=BICUBIC),
                            transforms.CenterCrop(args.resolution)])
                        preprocess = transforms.Compose([
                            transforms.ToTensor(),
                            normalize])
                        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1,
                                                             augmix=len(set_id) > 1) ### aug mix not used for ImageNet test set.
                        # data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1, augmix=False)

                        test_transform = transforms.Compose([
                            transforms.Resize(args.resolution, interpolation=BICUBIC),
                            transforms.CenterCrop(args.resolution), transforms.ToTensor(), normalize])
                        batchsize = 1

                        print("evaluating: {}".format(set_id))
                        # reset the model
                        # Reset classnames of custom CLIP model
                        if len(set_id) > 1:
                            # fine-grained classification datasets
                            classnames = eval("{}_classes".format(set_id.lower()))
                        else:
                            assert set_id in ['A', 'R', 'K', 'V', 'I']
                            classnames_all = imagenet_classes
                            classnames = []
                            if set_id in ['A', 'R', 'V']:
                                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                                if set_id == 'R':
                                    for i, m in enumerate(label_mask):
                                        if m:
                                            classnames.append(classnames_all[i])
                                else:
                                    classnames = [classnames_all[i] for i in label_mask]
                            else:
                                classnames = classnames_all

                        model.reset_classnames(classnames, set_id)

                        val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
                        print("number of test samples: {}".format(len(val_dataset)))
                        val_loader = torch.utils.data.DataLoader(
                                    val_dataset,
                                    batch_size=batchsize, shuffle=True,  ## the input has been shuffled.
                                    num_workers=args.workers, pin_memory=True)
                        args.set_id = set_id
                        ######################################################################################## build dataset ed


                        model.eval()


                        ################################################################################### get text feature st
                        with torch.no_grad():
                            text_feat, text_feat_full = model.dummy_get_text_features()
                        ################################################################################### get text feature ed


                        if args.n_shot:
                            if args.n_augview == 0:
                                train_dataset_mem = build_dataset(set_id, test_transform, args.data, mode='train', n_shot=args.n_shot)
                                print("number of training samples: {}".format(len(train_dataset_mem)))
                                train_loader_mem = torch.utils.data.DataLoader(
                                            train_dataset_mem,
                                            batch_size=1, shuffle=False,  ## the input has been shuffled.
                                            num_workers=args.workers, pin_memory=True)
                                init_image_memory(train_loader_mem, model, args)
                                del train_dataset_mem, train_loader_mem
                            else:
                                ######### generate num_aug_view augmented views for each samples; APE adopt ten...
                                assert args.n_augview % args.n_shot == 0
                                num_aug_view = int(args.n_augview / args.n_shot)
                                data_transform_aug = AugMemAugmenter(base_transform, preprocess, n_views=num_aug_view - 1,
                                                                 augmix=len(set_id) > 1)  ### aug mix not used for ImageNet test set.
                                train_dataset_mem = build_dataset(set_id, data_transform_aug, args.data, mode='train', n_shot=args.n_shot)
                                print("number of training samples: {}, number of augview: {}".format(len(train_dataset_mem), args.n_augview))
                                train_loader_mem = torch.utils.data.DataLoader(
                                            train_dataset_mem,
                                            batch_size=1, shuffle=False,  ## the input has been shuffled.
                                            num_workers=args.workers, pin_memory=True)
                                init_image_memory(train_loader_mem, model, args)
                                del train_dataset_mem, train_loader_mem
                        ########## extract the importance channels via APE.
                        if args.num_important_channel != 0:
                            important_indice = important_channel_indice(args, model) ##
                            args.indice = important_indice
                        else:
                            important_indice = torch.arange(model.text_feat.shape[1]).to(model.text_feat.device) ## use all channels.
                            args.indice = important_indice


                        ####################################################################################### inference st
                        results_temp = direct_inference(val_loader, model, args)
                        ####################################################################################### inference ed


                        print_log("lr: {}, epoch:{}, num_important_channel{}, lambda_ape: {}, best acc{:.2f} \n".format(lr, epoch, num_important_channel, lambda_ape, results_temp[3]), end="	")
                        if results_temp[3] > best_acc:
                            results[set_id] = results_temp
                            best_acc = results_temp[3]
                        # results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args)
                        del val_dataset, val_loader
                        try:
                            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
                        except:
                            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))
                        length = len(results[set_id])

    args.indice = 0
    log = open(os.path.join(args.log + '.txt'), 'a')
    state = {k: v for k, v in args._get_kwargs()}
    log.write(json.dumps(state) + '\n')
    log.close()
    print_log("======== Result Summary ========")
    print_log("params: bs	lr	selection_p")
    print_log("params: {}	{}	{}".format(args.batch_size, args.lr, args.selection_p))
    print_log("\t\t [set_id] \t\t Top-1 acc. \t\t Top-1 local acc, \t\t Top-1 global acc \t\t Searched acc \t\t beta \t\t gama.")
    for id in results.keys():
        print_log("{}".format(id), end="	")
    print_log('mean', end="	")
    print_log("\n")
    for i in range(length):
        cul_acc = 0
        cul_count = 0
        for id in results.keys():
            print_log("{:.3f}".format(results[id][i]), end="	")
            cul_acc += float(results[id][i])
            cul_count += 1
        print_log("{:.3f}".format(cul_acc), end="	")
        print_log("\n")


def direct_inference(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top1_global = AverageMeter('AccImg@1', ':6.2f', Summary.AVERAGE)
    top1_global_fewshot = AverageMeter('AccGF@1', ':6.2f', Summary.AVERAGE)
    top1_text_vote = AverageMeter('AccVote1@1', ':6.2f', Summary.AVERAGE)
    top1_weighted_text_vote = AverageMeter('WWAccVote1@1', ':6.2f', Summary.AVERAGE)
    top1_global_fewshot_vote = AverageMeter('AccVoteG@1', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top1_global, top1_global_fewshot, top1_text_vote, top1_weighted_text_vote, top1_global_fewshot_vote],
        prefix='Test: ')

    # reset model and switch to evaluate mode
    model.eval()
    if model.first_flag:
        with torch.no_grad():
            text_feat, text_feat_full = model.get_text_features()
    else:
        print('the text feat has already initilized, pass it here.')
    ## text_feat: 200*1024
    ## text_feat_full:  200 * 7 * 1024
    class_num, feat_dim = model.text_feat.shape[0], model.text_feat.shape[1]
    pred_vanilla = []
    pred_global = []
    pred_weighted = []
    pred_local = []
    pred_fewshot_global = []
    pred_fewshot_local = []
    labels = []
    
    dmnet = DualMem(args=args, beta=args.beta, feat_dim=feat_dim, class_num=class_num, mapping=args.mapping).cuda()

    dmnet.eval()
    end = time.time()
    timestamp = time.time()
    time_parts = time.gmtime(timestamp)
    hours = time.strftime("%H", time_parts)
    minutes = time.strftime("%M", time_parts)
    seconds = time.strftime("%S", time_parts)
    print("test start Time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))

    for i, (images, target) in enumerate(val_loader):
        ##################################################################################################################### for문 test st
        assert args.gpu is not None
        if isinstance(images, list):  ### augmix return, list
            images = torch.cat(images, dim=0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images[:1]
        else: ## standard return, Tensor
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images[:1]
        target = target.cuda(args.gpu, non_blocking=True)


        ############################################################################ get image feature st
        with torch.no_grad():
            image_features_global, image_features_local =  model.get_image_features(images)
        # image_features_global: torch.Size([128, 1024])
        # image_features_local: torch.Size([128, 49, 1024])
        ############################################################################ get image feature ed


        ################################################################################################## clip classification st
        with torch.no_grad():
            img_text = dmnet.get_text_prediction(model)
        img_text_pred = img_text[:1]  ## current prediction.
        confidence_prediction, selected_idx, confused_weak_output, confused_idx = select_confident_samples(img_text, args.selection_p)
        # ipdb.set_trace()
        ## vote of multiple predictions, this is typically worse than img_text_pred, but introduce information of other views.
        dmnet.init_pred = confidence_prediction.mean(0, keepdim=True)
        acc1, _ = accuracy(dmnet.init_pred, target, topk=(1, 5))
        top1_text_vote.update(acc1[0], image.size(0))
        ################################################################################################## clip classification ed


        ################################################################################################## clip weighted classification st
        with torch.no_grad():
            weighted_img_text_pred = dmnet.get_weighed_text_prediction(model)

        acc1, _ = accuracy(weighted_img_text_pred, target, topk=(1, 5))
        top1_weighted_text_vote.update(acc1[0], image.size(0))
        ################################################################################################## clip weighted classification ed


        ############################################################################################## few shot classification st
        if args.n_shot:
            with torch.no_grad():
                with torch.no_grad():
                    fewshot_global_pred_fullview = dmnet.get_image_pred_fewshot_global(model) ## N*class, probability
                fewshot_global_pred = fewshot_global_pred_fullview[:1] ## 1*class
                confidence_prediction_fewshot_global, _, _, _ = select_confident_samples(fewshot_global_pred_fullview, 1.0)

                acc1, _ = accuracy(confidence_prediction_fewshot_global.mean(0, keepdim=True), target, topk=(1, 5))
                top1_global_fewshot_vote.update(acc1[0], image.size(0))
        ############################################################################################## few shot classification ed


        ################################################################################ get dynamic memory classification st
        dmnet.update_memory_bank(model, target) 

        with torch.no_grad():
            img_global_pred = dmnet.get_image_pred(model)  ## with updated local

        pred_vanilla.append(img_text_pred)
        pred_weighted.append(weighted_img_text_pred)
        pred_global.append(img_global_pred)


        if args.n_shot:
            pred_fewshot_global.append(fewshot_global_pred)
        labels.append(target)

        # # measure accuracy and record loss
        acc1, _ = accuracy(img_text_pred, target, topk=(1, 5))
        acc1_global, _ = accuracy(img_global_pred, target, topk=(1, 5))
        if args.n_shot:
            acc1_global_fs, _ = accuracy(fewshot_global_pred, target, topk=(1, 5))

        top1.update(acc1[0], image.size(0))
        top1_global.update(acc1_global[0], image.size(0))
        if args.n_shot:
            top1_global_fewshot.update(acc1_global_fs[0], image.size(0))
        ################################################################################## get dynamic memory classification ed

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        torch.cuda.empty_cache()

        if (i+1) % args.print_freq == 0:
            progress.display(i)
        ##################################################################################################################### for문 test ed


    timestamp = time.time()
    time_parts = time.gmtime(timestamp)
    hours = time.strftime("%H", time_parts)
    minutes = time.strftime("%M", time_parts)
    seconds = time.strftime("%S", time_parts)
    print("end Time: {} hours, {} minutes, {} seconds".format(hours, minutes, seconds))

    progress.display_summary()
    pred_vanilla = torch.cat(pred_vanilla, dim=0)
    pred_global = torch.cat(pred_global, dim=0)
    pred_weighted = torch.cat(pred_weighted, dim=0)
    # pred_local = torch.cat(pred_local, dim=0)
    if args.n_shot:
        pred_fewshot_global = torch.cat(pred_fewshot_global, dim=0)
        # pred_fewshot_local = torch.cat(pred_fewshot_local, dim=0)
    else:
        pred_fewshot_global = pred_vanilla
        # pred_fewshot_local = pred_vanilla
    labels = torch.cat(labels, dim=0)
    ########## put the hyper parameter search here.
    ## final prediction = image_text_pred + alpha * global + beta * local
    weight_search = True
    search_step = 10
    if weight_search:
        beta1_list = [1.0]
        beta2_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        if args.n_shot:
            beta3_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        else:
            beta3_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
        # beta1_list = [i * (10 - 0.001) / search_step + 0.001 for i in range(search_step)]  ## 0.001 - 10
        print('-' * 20)
        print('Starting searching...')
        print('     beta1 searching range: [0.001, ' + str(10) + ']')
        print('     beta2 searching range: [0.001, ' + str(10) + ']')
        print('     beta3 searching range: [0.001, ' + str(10) + ']')
        print('-' * 20)

        best_acc = 0.
        best_beta2 = 0.
        best_beta3 = 0.


        ############################################################################################### final prediction start
        for beta1 in beta1_list:
            for beta2 in beta2_list:
                for beta3 in beta3_list:
                    logits = pred_vanilla * beta1 + pred_global * beta2 + pred_weighted * beta3
                    acc, _ = accuracy(logits, labels, topk=(1, 5))
                    acc = acc.item()
                    if acc > best_acc:
                        print('New best setting, beta1: {:.4f}; beta2: {:.4f}; beta3: {:.4f}; Acc: {:.2f}'.format(beta1, beta2,beta3, acc))
                        best_acc = acc
                        best_beta1 = beta1
                        best_beta2 = beta2
                        best_beta3 = beta3
        ##############################################################################################  final prediction end


        print(f"Searched Acc: {best_acc:.2f} with beta1 {best_beta1:.3f}, dynamic {best_beta2:.3f} and static {best_beta3:.3f}")

    return [top1.avg, top1_global.avg, top1_weighted_text_vote.avg, best_acc, best_beta1, best_beta2, best_beta3]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I', help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')

    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')

    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_shot', type=int, default=None)
    parser.add_argument('--n_augview', type=int, default=0, help='use augmented few shot samples') 
    parser.add_argument('--ft', action='store_true', default=False, help="fine tuning the attention weight with few labeled data.")
    parser.add_argument('--use_searched_param', action='store_true', default=False, help="using searched param for each dataset")
    
    parser.add_argument('--beta',  default=5.5, type=float, help='loss weight')
    parser.add_argument('--mapping', type=str, default='bias', help='bias | affine | all')
    parser.add_argument('--position', type=str, default='all', help='query | key | value | qkv | output | all')
    parser.add_argument('--optimizer', type=str, default='adamw', help='adamw | sgd')
    parser.add_argument('--eps',  default=1e-8, type=float, help='eps, default 1e-8')
    parser.add_argument('--wd',  default=1e-4, type=float, help='weight decay')
    parser.add_argument('--lr',  default='0.0001', type=str, help='learning rate')
    parser.add_argument('--epoch', type=str, default='20')
    parser.add_argument('--shared_param', action='store_true', default=False, help="shared parameters acorss local | global | text.")
    parser.add_argument('--num_important_channel', type=str, default='0') ## if 0, use all channels; otherwise, selecting the ape_channel_num
    parser.add_argument('--lambda_ape', default='0.7', type=str, help='following ape.')
    parser.add_argument('--memory_size', type=int, default=50)
    parser.add_argument('--text_prompt', type=str, default='tip_cupl', help='simple | tip | full | tip_cupl')
    parser.add_argument('--log', type=str, default='loga', help='some places to write note')


    main()