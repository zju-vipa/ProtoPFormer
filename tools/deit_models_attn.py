from turtle import forward
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, PatchEmbed, Mlp, DropPath

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384', 'deit_tiny_patch2_32', 'deit_tiny_patch2_32_wo_pos',
]

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        B, N, _ = policy.size()
        B, H, N, N = attn.size()
        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, policy=None):
        x_ori = x
        x, attn = self.attn(self.norm1(x), policy)
        x = x_ori + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class MyVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        norm_layer = kwargs['norm_layer'] or partial(nn.LayerNorm, eps=1e-6)
        act_layer = None or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, kwargs['drop_path_rate'], kwargs['depth'])]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], mlp_ratio=kwargs['mlp_ratio'], qkv_bias=kwargs['qkv_bias'], drop=kwargs['drop_rate'],
                attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(kwargs['depth'])])
        self.init_weights('')

        del self.head

    def attn_rollout(self, all_attn, discard_ratio=0.9, head_fusion='mean'):
        result = torch.eye(all_attn[0].shape[-1]).unsqueeze(dim=0).repeat(all_attn[0].shape[0], 1, 1).cuda()
        for attn in all_attn:
            # attn : (batch_size, head_num, 196, 196)
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
            # a = (attn_fused + 1.0 * I) / 2
            # a = attn_fused  # mark, identity
            identity_w = 0.2
            a = (attn_fused + identity_w * I) / (1. + identity_w)

            a = a / a.sum(dim=-1).unsqueeze(dim=-1)

            result = torch.matmul(a, result)
        return result

    def forward_feature_maps(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        all_attn = []
        for blk in self.blocks:
            x, attn = blk(x)
            all_attn.append(attn)
        attn_rollout = self.attn_rollout(all_attn)

        x = self.norm(x)

        return x[:, 1:]

    def forward_feature_maps_wtcls(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:]
        x = self.pos_drop(x)

        all_attn, all_feas = [], []
        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)
            all_attn.append(attn)
            all_feas.append(x)
        attn_rollout = self.attn_rollout(all_attn)

        x = self.norm(x)

        return x, (attn_rollout, all_feas)

    def forward_feature_patch_embed(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:]
        x = self.pos_drop(x)

        return x

    def forward_feature_patch_embed_all(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        return x[:, :1], x[:, 1:]

    def forward_feature_mask_train_thresh(self, cls_embed, x_embed, token_attn=None, reserve_layer_nums=[]):
        '''
        cls_embed : (B, 1, dim)
        x_embed : (B, 196, dim)
        '''
        B, patch_num = x_embed.shape[0], x_embed.shape[1]
        layer_ids = [x[0] for x in reserve_layer_nums]
        
        policy = torch.ones(B, 1 + patch_num, 1, device=x_embed.device) # (B, 1 + 196, 1)
        x = torch.cat([cls_embed, x_embed], dim=1)  # (B, 1 + 196, dim)
        all_attn = []
        for i, blk in enumerate(self.blocks):
            if i in layer_ids:
                thresh = 1. / 196
                policy = (token_attn >= thresh).type(torch.cuda.FloatTensor)
                policy = torch.cat([torch.ones(B, 1, device=x.device), policy], dim=1)
                policy = policy[:, :, None]
            x, attn = blk(x, policy)
            all_attn.append(attn)
        all_attn = all_attn[:layer_ids[0]]  # select the first several layers (196 * 196)
        attn_rollout = self.attn_rollout(all_attn)

        x = self.norm(x)

        return x, (attn_rollout, None)

    def forward_feature_mask_train_direct(self, cls_embed, x_embed, token_attn=None, reserve_layer_nums=[]):
        '''
        directly uses the attn rollout as token attn to discard tokens
        cls_embed : (B, 1, dim)
        x_embed : (B, 196, dim)
        '''
        B, patch_num = x_embed.shape[0], x_embed.shape[1]
        layer_ids = [x[0] for x in reserve_layer_nums]
        
        policy = torch.ones(B, 1 + patch_num, 1, device=x_embed.device) # (B, 1 + 196, 1)
        x = torch.cat([cls_embed, x_embed], dim=1)  # (B, 1 + 196, dim)
        all_attn = []
        for i, blk in enumerate(self.blocks):
            if i in layer_ids:
                all_attn = all_attn[:i] # (196, 196)
                attn_rollout = self.attn_rollout(all_attn)
                attn_rollout = attn_rollout.detach()    # detach !!!
                cls_token_attn = attn_rollout[:, 0, 1:]

                reserve_token_num = reserve_layer_nums[layer_ids.index(i)][1]
                reserve_token_indice = torch.topk(cls_token_attn, k=reserve_token_num, dim=-1)[1]   # (B, reserve_token_num)
                reserve_token_indice = reserve_token_indice.sort(dim=-1)[0]
                reserve_token_indice += 1   # omit cls token
                policy = torch.cat([torch.ones(B, 1, device=x.device), torch.zeros(B, patch_num, device=x.device)], dim=1)
                policy.scatter_(1, reserve_token_indice, 1) # (B, 1 + patch_num)
                policy = policy[:, :, None]
            x, attn = blk(x, policy)
            all_attn.append(attn)

        x = self.norm(x)

        return x, (cls_token_attn, None)


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = MyVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = MyVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], strict=False)
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = MyVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_patch2_32(pretrained=False, **kwargs):
    model = MyVisionTransformer(
        img_size=32, patch_size=2,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_patch2_28(pretrained=False, **kwargs):
    model = MyVisionTransformer(
        img_size=28, patch_size=2,
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model