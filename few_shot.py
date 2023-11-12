import torch
from dataset import *



from collections import OrderedDict
def initialize_memory(obj_list):

    mid = []
    large = []
    patch = []
    for x in obj_list:
        mid.append((x, []))
        large.append((x, []))
        patch.append((x, []))
    mid_memory   = OrderedDict(mid)
    large_memory = OrderedDict(large)
    patch_memory = OrderedDict(patch)
    return mid_memory, large_memory, patch_memory


@torch.no_grad()
def memory(model, obj_list, dataset_dir, save_path, preprocess, transform, k_shot, few_shot_features,
           dataset_name, device):
    normal_features_ls = {}
    mid_memory, large_memory, patch_memory = initialize_memory(obj_list)
    for i in range(len(obj_list)):
        if dataset_name == 'mvtec':
            normal_data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                       aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path,
                                       obj_name=obj_list[i])
        elif dataset_name == 'visa':
            normal_data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                      mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj_list[i])

        normal_dataloader = torch.utils.data.DataLoader(normal_data, batch_size=1, shuffle=False)
        for index, items in enumerate(normal_dataloader):

            images = items['img'].to(device)
            cls_name = items['cls_name']
            cls_id = items['cls_id']
            patch_size = 16
            gt_mask = items['img_mask']
            gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
            # print("class_name", cls_name)
            large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_scale, mid_scale = model.encode_image(images, patch_size)
            # print("large_scale_tokens", large_scale_tokens.shape, mid_scale_tokens.shape, patch_tokens.shape)
            for class_name, tokens in zip(cls_name, large_scale_tokens):
                large_memory[class_name].append(tokens)
            for class_name, tokens in zip(cls_name, mid_scale_tokens):
                mid_memory[class_name].append(tokens)
            for class_name, tokens in zip(cls_name, patch_tokens):
                patch_memory[class_name].append(tokens)
            #     print("lennnnnshape", tokens.shape)
            # print("large_memory", large_memory)
            # print("mid_memory", mid_memory)
            # print("large_memory", patch_memory)
    for class_name in obj_list:
        large_memory[class_name] = torch.cat(large_memory[class_name])
        mid_memory[class_name] = torch.cat(mid_memory[class_name])
        patch_memory[class_name] = torch.cat(patch_memory[class_name])
        # print("lennnnnshape", patch_memory[class_name].shape)


    return large_memory, mid_memory, patch_memory