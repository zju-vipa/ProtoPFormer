import torch
import torch.nn as nn
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.cait import Cait
from timm.models.registry import register_model
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_
from timm.models.helpers import build_model_with_cfg, overlay_external_default_cfg

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 384, 384), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = dict(
    cait_xxs24_224=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/XXS24_224.pth',
        input_size=(3, 224, 224),
    ),
    cait_s24_224=_cfg(
        url='https://dl.fbaipublicfiles.com/deit/S24_224.pth',
        input_size=(3, 224, 224),
    ),
)


class ClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, M, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        # eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, M, N)
        eye = torch.zeros(1, 1, M, N, device=attn_policy.device)
        eye[:, :, :, 0] = 1.

        # attn_policy = attn_policy + (1.0 - attn_policy) * eye
        attn_policy = attn_policy

        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy=None):
        B, N, C = x.shape
        q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls, attn


class TalkingHeadAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add Talking Heads Attention (https://arxiv.org/pdf/2003.02436v1.pdf)
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.num_heads = num_heads

        head_dim = dim // num_heads

        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1))

        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        attn = attn.softmax(dim=-1)

        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class LayerScaleBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add layerScale
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_block=TalkingHeadAttn,
            mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        ori_x = x
        x, attn = self.attn(self.norm1(x))
        x = ori_x + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x, attn


class LayerScaleBlockClassAttn(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_block=ClassAttn,
            mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_cls, policy=None):
        u = torch.cat((x_cls, x), dim=1)
        ori_x_cls = x_cls
        x_cls, attn = self.attn(self.norm1(u), policy)
        x_cls = ori_x_cls + self.drop_path(self.gamma_1 * x_cls)
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls, attn


class MyCait(Cait):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        block_layers = LayerScaleBlock
        block_layers_token = LayerScaleBlockClassAttn
        attn_block_token_only = ClassAttn
        mlp_block_token_only = Mlp
        act_layer = nn.GELU
        attn_block = TalkingHeadAttn
        mlp_block = Mlp
        attn_drop_rate = 0.
        mlp_ratio = 4.
        qkv_bias = True
        depth_token_only = 2
        mlp_ratio_clstk = 4.0
        self.layer_nums = [kwargs['depth'], depth_token_only]

        dpr = [kwargs['drop_path_rate'] for i in range(kwargs['depth'])]
        self.blocks = nn.ModuleList([
            block_layers(
                dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=kwargs['drop_rate'], attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, attn_block=attn_block, mlp_block=mlp_block, init_values=kwargs['init_scale'])
            for i in range(kwargs['depth'])])

        self.blocks_token_only = nn.ModuleList([
            block_layers_token(
                dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer, attn_block=attn_block_token_only,
                mlp_block=mlp_block_token_only, init_values=kwargs['init_scale'])
            for i in range(depth_token_only)])
        self.apply(self._init_weights)

    def attn_rollout_cait(self, all_attn, discard_ratio=0.9, head_fusion='max', layer_nums=[]):
        pre_layer_num = layer_nums[0]
        n_all_attn = []
        for k, attn in enumerate(all_attn):
            if head_fusion == "mean":
                attn_fused = attn.mean(axis=1)
            elif head_fusion == "max":
                attn_fused = attn.max(axis=1)[0]    # (batch_size, 196, 196)
            elif head_fusion == "min":
                attn_fused = attn.min(axis=1)[0]
            
            flat = attn_fused.view(attn_fused.shape[0], -1) # (batch_size, 196 * 196)
            _, indices = flat.topk(int(flat.shape[-1] * discard_ratio), -1, False)
            # flat[indices] = 0
            flat.scatter_(1, indices, 0)


            I = torch.eye(attn_fused.shape[-1]).cuda()
            I = I[:attn_fused.shape[1]]
            # a = (attn_fused + 1.0 * I) / 2
            # a = attn_fused
            identity_w = 0.2
            a = (attn_fused + identity_w * I) / (1. + identity_w)

            a = a / a.sum(dim=-1).unsqueeze(dim=-1)
            n_all_attn.append(a)
        all_attn = n_all_attn
        all_attn, cls_attn = all_attn[:pre_layer_num], all_attn[pre_layer_num:]

        result = torch.eye(all_attn[0].shape[-1]).unsqueeze(dim=0).repeat(all_attn[0].shape[0], 1, 1).cuda()
        for attn in all_attn:
            result = torch.matmul(attn, result)

        cls_result = torch.cat(cls_attn, dim=1) # (B, 2, 197)
        cls_result = cls_result.mean(dim=1, keepdim=True) # (B, 1, 197)
        cls_result = cls_result[:, :, 1:]  # (B, 1, 196)
        cls_result = cls_result @ result    # (B, 1, 196)

        return result, cls_result

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)

        for i, blk in enumerate(self.blocks_token_only):
            cls_tokens, attn = blk(x, cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        return x[:, 0]

    def forward_feature_maps(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)

        for i, blk in enumerate(self.blocks_token_only):
            cls_tokens, attn = blk(x, cls_tokens)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        return x[:, 1:]

    def forward_feature_patch_embed_all(self, x):
        B, patch_num = x.shape[0], x.shape[1]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        return cls_tokens, x

    def forward_feature_mask_train_direct(self, cls_embed, x_embed, token_attn, reserve_layer_nums=[]):
        B, patch_num = x_embed.shape[0], x_embed.shape[1]
        cls_tokens, x = cls_embed, x_embed

        all_attn = []
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)
            all_attn.append(attn)

        layer_ids = [x[0] for x in reserve_layer_nums]
        policy = torch.ones(B, 1 + patch_num, 1, device=cls_tokens.device) # (B, 1 + 196, 1)

        for i, blk in enumerate(self.blocks_token_only):
            if i in layer_ids:
                _, cls_attn_ma = self.attn_rollout_cait(all_attn, discard_ratio=0.9, head_fusion='mean', layer_nums=[self.layer_nums[0], i])
                cls_attn_ma = cls_attn_ma.detach()  # detach !!!
                cls_token_attn = cls_attn_ma[:, 0]

                reserve_token_num = reserve_layer_nums[layer_ids.index(i)][1]
                reserve_token_indice = torch.topk(cls_token_attn, k=reserve_token_num, dim=-1)[1]   # (B, reserve_token_num)
                reserve_token_indice = reserve_token_indice.sort(dim=-1)[0]
                reserve_token_indice += 1   # omit cls token
                policy = torch.cat([torch.ones(B, 1, device=x.device), torch.zeros(B, patch_num, device=x.device)], dim=1)
                policy.scatter_(1, reserve_token_indice, 1) # (B, 1 + patch_num)
                policy = policy[:, :, None]
            cls_tokens, attn = blk(x, cls_tokens, policy)
            all_attn.append(attn)

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.norm(x)
        return x, (cls_token_attn, None)


def checkpoint_filter_fn(state_dict, model=None):
    if 'model' in state_dict:
        state_dict = state_dict['model']
    checkpoint_no_module = {}
    for k, v in state_dict.items():
        checkpoint_no_module[k.replace('module.', '')] = v
    return checkpoint_no_module


def _create_cait(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        MyCait, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    # delete head
    del model.head
    return model


@register_model
def cait_xxs24_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=192, depth=24, num_heads=4, init_scale=1e-5, **kwargs)
    model = _create_cait('cait_xxs24_224', pretrained=pretrained, **model_args)
    return model


@register_model
def cait_s24_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, embed_dim=384, depth=24, num_heads=8, init_scale=1e-5, **kwargs)
    model = _create_cait('cait_s24_224', pretrained=pretrained, **model_args)
    return model