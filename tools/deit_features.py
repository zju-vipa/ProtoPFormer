import torch
import torch.nn as nn
import torch.nn.functional as F
# import tools.deit_models as deit_models
import tools.deit_models_attn as deit_models
# import tools.deit_models_cls as deit_models
from timm.models import create_model


def get_pretrained_weights_path(model_name):

    if model_name in ["deit_small_patch16_224", "deit_base_patch16_224", "deit_tiny_patch16_224",
            "deit_tiny_distilled_patch16_224"]:
        if model_name == "deit_small_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth'
        elif model_name == "deit_base_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
        elif model_name == "deit_tiny_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth'
        elif model_name == "deit_tiny_distilled_patch16_224":
            finetune = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth'

    return finetune


def get_pretrained_weights(model_name, model):
    finetune = get_pretrained_weights_path(model_name)
    if finetune.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            finetune, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(finetune, map_location='cpu')

    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]

    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed

    model.load_state_dict(checkpoint_model, strict=False)

    return model


def deit_tiny_patch_features(pretrained=False, nb_classes=1000, drop=0.0, drop_path=0.1, **kwargs):
    model_name = 'deit_tiny_patch16_224'
    model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=nb_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
        )
    # if pretrained == True:
    #     model = get_pretrained_weights(model_name, model)

    return model

def deit_small_patch_features(pretrained=False, nb_classes=1000, drop=0.0, drop_path=0.1, **kwargs):
    model_name = 'deit_small_patch16_224'
    model = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=nb_classes,
            drop_rate=drop,
            drop_path_rate=drop_path,
        )
    # if pretrained == True:
    #     model = get_pretrained_weights(model_name, model)

    return model