def init_image_memory(train_loader, model, args):
    model.eval()
    if model.first_flag:
        with torch.no_grad():
            text_feat, text_feat_full = model.get_text_features()
    else:
        print('the text feat has already initilized, pass it here.')
    memorized_image_global_feat = [] ## N*[shot*aug]*C
    memorized_image_local_feat = []  ## N*[shot*aug]*C
    memorized_image_global_feat_vanilla = [] ## N*shot*C
    memorized_image_local_feat_vanilla = []  ## N*shot*C
    memorized_labels = []

    for i in range(model.n_cls):
        memorized_image_global_feat.append([])
        memorized_image_local_feat.append([])
        memorized_image_global_feat_vanilla.append([])
        memorized_image_local_feat_vanilla.append([])
        memorized_labels.append([])

    for i, (images, target) in enumerate(train_loader):
        assert args.gpu is not None
        if isinstance(images, list):  ### augmix return, list
            images = torch.cat(images, dim=0)
            images = images.cuda(args.gpu, non_blocking=True)
        else: ## standard return, Tensor
            if len(images.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        with torch.no_grad():
            image_features_global, image_features_local =  model.get_image_features(images) ## 4*1024; 4*49*1024.
        text_features = model.text_feat[target]  ## 512
        ## only use the original ?? we should use all; however, only use the vanilla one in the dynamic memory.
        selected_image_features_local = model.image_features_local
        cos_sim = (selected_image_features_local * text_features).sum(-1)  ## between 0.2-0.3, very close.
        weight_prob = (cos_sim * 100).softmax(-1)   ## 1*197, following clip temperature.
        ########
        attented_feat = (weight_prob.unsqueeze(-1) * selected_image_features_local).sum(1)  ## 1*512
        attented_feat = attented_feat / attented_feat.norm(dim=-1, keepdim=True)  ## 1*512
        memorized_image_global_feat[target].append(image_features_global) ## aug*C
        memorized_image_local_feat[target].append(attented_feat)   # aug * C
        memorized_image_global_feat_vanilla[target].append(image_features_global[:1]) ## aug*C
        memorized_image_local_feat_vanilla[target].append(attented_feat[:1])   # aug * C
        one_hot_target = torch.zeros(1, model.n_cls).to(target.device)
        one_hot_target[0, target] = 1
        memorized_labels[target].append(one_hot_target)   ## 1 * C, turn it to one hot labels.

    for i in range(model.n_cls):
        memorized_image_global_feat[i] = torch.cat(memorized_image_global_feat[i], dim=0).unsqueeze(0) ## 1*augshot*C
        memorized_image_local_feat[i] = torch.cat(memorized_image_local_feat[i], dim=0).unsqueeze(0)
        memorized_image_global_feat_vanilla[i] = torch.cat(memorized_image_global_feat_vanilla[i], dim=0).unsqueeze(0) ## 1*shot*C
        memorized_image_local_feat_vanilla[i] = torch.cat(memorized_image_local_feat_vanilla[i], dim=0).unsqueeze(0)
        memorized_labels[i] = torch.cat(memorized_labels[i], dim=0).unsqueeze(0)

    memorized_image_global_feat = torch.cat(memorized_image_global_feat, dim=0) ## n*shot*c
    memorized_image_local_feat = torch.cat(memorized_image_local_feat, dim=0)
    memorized_image_global_feat_vanilla = torch.cat(memorized_image_global_feat_vanilla, dim=0) ## n*shot*c
    memorized_image_local_feat_vanilla = torch.cat(memorized_image_local_feat_vanilla, dim=0)
    memorized_labels = torch.cat(memorized_labels, dim=0)

    ######## memorized few shot features and labels.
    model.fewshot_image_global_feat = memorized_image_global_feat ## class*augshot*c
    model.fewshot_image_local_feat = memorized_image_local_feat
    model.fewshot_image_global_feat_vanilla = memorized_image_global_feat_vanilla ## class*shot*c
    model.fewshot_image_local_feat_vanilla = memorized_image_local_feat_vanilla
    model.fewshot_label = memorized_labels  ## class*shot*c, one hot labels

    ############# add features of labeled data to the dynamic memory. This is important when there are more labeled data.
    model.fixed_global_feat_vanilla = torch.cat((model.fixed_global_feat, memorized_image_global_feat_vanilla), dim=1)  ## N*1*C
    model.fixed_local_feat_vanilla = torch.cat((model.fixed_local_feat, memorized_image_local_feat_vanilla), dim=1)  ## N*1*C

    ###################### for static memory, with text feature and augmented image feat
    model.fixed_global_feat = torch.cat((model.fixed_global_feat, memorized_image_global_feat), dim=1)  ## N*1*C
    model.fixed_local_feat = torch.cat((model.fixed_local_feat, memorized_image_local_feat), dim=1)  ## N*1*C

    print('appending the few shot image feature to fixed image memories.')