import math
import time
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange
from torch.autograd import Function

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import trunc_normal_, lecun_normal_, to_2tuple
from timm.models.registry import register_model

_logger = logging.getLogger(__name__)

#####------------ Gradient Reversal Layer ------------####
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grad_reverse(x, lambda_=1.0):
    return GradientReversal.apply(x, lambda_)

#####----------------------------------------------------####


def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x    
    
class PatchEmbed_3d(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]),
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        # x = x.permute(0, 2, 1, 3, 4)
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = rearrange(x, 'b c t h w -> b c t (h w)')
        x = rearrange(x, 'b c t n -> b c (t n)').transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., keep_rate=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)
        
    def forward(self, x, keep_rate=None, tokens=None):
        if keep_rate is None:
            keep_rate = self.keep_rate
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Token pruning logic
        left_tokens = N - 2  # because we assume 2 CLS tokens
        if (self.keep_rate < 1 and keep_rate < 1) or (tokens is not None):
            left_tokens = math.ceil(keep_rate * (N - 2))
            if tokens is not None:
                left_tokens = tokens
            if left_tokens == N - 2:
                return x, None, None, None, None, None, left_tokens
            assert left_tokens >= 1

            act_attn = attn[:, :, 0, 2:]  
            priv_attn = attn[:, :, 1, 2:]    

            act_attn = act_attn.mean(dim=1)  
            priv_attn = priv_attn.mean(dim=1)      
            
            # normalize
            act_attn = act_attn / (act_attn.sum(dim=1, keepdim=True))
            priv_attn = priv_attn / (priv_attn.sum(dim=1, keepdim=True))
            
            lambda_priv = 0.5   # lambda_priv > 1 strong privacy removal, lambda_priv < 1 strong utility preservation
            attn_score = act_attn - lambda_priv * priv_attn  

            # Top-k based on importance
            _, idx = torch.topk(attn_score, left_tokens, dim=1, largest=True, sorted=True)  
            index = idx.unsqueeze(-1).expand(-1, -1, C)  
            
            return x, index, idx, attn_score, act_attn, priv_attn, left_tokens
            
        return x, None, None, None, None, None, left_tokens


class Block(nn.Module):
    def __init__(self,dim,num_heads,mlp_ratio=4.,qkv_bias=False,drop=0.,attn_drop=0.,drop_path=0.,act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,keep_rate=0.,fuse_token=False):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,num_heads=num_heads,qkv_bias=qkv_bias,attn_drop=attn_drop,proj_drop=drop,keep_rate=keep_rate)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,hidden_features=mlp_hidden_dim,act_layer=act_layer,drop=drop)

        self.keep_rate = keep_rate
        self.mlp_hidden_dim = mlp_hidden_dim
        self.fuse_token = fuse_token

    def forward(self, x, keep_rate=None, tokens=None, get_idx=False):
        if keep_rate is None:
            keep_rate = self.keep_rate
        B, N, C = x.shape
        
        tmp, index, idx, _, act_attn, _, _ = self.attn(self.norm1(x), keep_rate, tokens)
        x = x + self.drop_path(tmp)

        # --- Token pruning ---
        if index is not None:
            non_cls = x[:, 2:]  
            x_others = torch.gather(non_cls, dim=1, index=index)  

            if self.fuse_token:
                compl = complement_idx(idx, N - 2)  
                non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  
                   
                non_topk_act_attn  = torch.gather(act_attn, dim=1, index=compl)  
                extra_token = torch.sum(non_topk * non_topk_act_attn.unsqueeze(-1), dim=1, keepdim=True)  
                x = torch.cat([x[:, 0:2], x_others, extra_token], dim=1)
            else:
                x = torch.cat([x[:, 0:2], x_others], dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        n_tokens = x.shape[1] - 2
        if get_idx and index is not None:
            return x, n_tokens, idx
        return x, n_tokens, None
    


class EViT(nn.Module):
    """EViT: Privacy-Preserving Token-Pruning Transformer for Video Recognition"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        tubelet_size=2,
        all_frames=16,
        num_classes=1000,
        num_priv=5,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        act_layer=None,
        weight_init="",
        keep_rate=(1,),
        fuse_token=False,
        get_idx=False,
    ):
        super().__init__()

        # -------------------------------
        # Core hyperparameters
        # -------------------------------
        self.img_size = img_size
        if len(keep_rate) == 1:
            keep_rate = keep_rate * depth
        self.keep_rate = keep_rate
        self.depth = depth
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.num_priv = num_priv
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 3 if distilled else 2  # act + priv (+ distill)
        self.num_tubelet = all_frames // tubelet_size
        self.fuse_token = fuse_token
        self.get_idx = get_idx
        self.distilled = distilled

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # -------------------------------
        # Patch embedding
        # -------------------------------
        self.patch_embed = PatchEmbed_3d(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            num_frames=all_frames,
            tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        self.num_tubelet = all_frames // tubelet_size

        # -------------------------------
        # Tokens and position embedding
        # -------------------------------
        self.cls_token_act = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_priv = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # -------------------------------
        # Transformer blocks
        # -------------------------------
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    keep_rate=keep_rate[i],
                    fuse_token=fuse_token,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # -------------------------------
        # Representation layer
        # -------------------------------
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # -------------------------------
        # Classification heads
        # -------------------------------
        self.head_act = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_priv = nn.Linear(self.num_features, num_priv) if num_priv > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # -------------------------------
        # Weight initialization
        # -------------------------------
        self.init_weights(weight_init)

    # -------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------
    def init_weights(self, mode=""):
        assert mode in ("jax", "jax_nlhb", "nlhb", "")
        trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.cls_token_act, std=0.02)
        trunc_normal_(self.cls_token_priv, std=0.02)
        self.apply(_init_vit_weights)

    # -------------------------------------------------------------
    # Mask builder: kept tubelet global ids -> (B,1,T,H,W)
    # Supports fuse_token=True via sentinel -1 in kept_global.
    # -------------------------------------------------------------
    def _kept_ids_to_mask_video(self, kept_global, T, H, W):
        tubelet = self.patch_embed.tubelet_size
        patch = self.patch_embed.patch_size

        # handle tuple patch_size / tubelet_size
        if isinstance(patch, (tuple, list)):
            assert len(patch) == 2, patch
            ph, pw = patch
            assert ph == pw, "Only square patches supported for mask grid"
            patch = ph

        if isinstance(tubelet, (tuple, list)):
            # common forms: (t,) or (t, ph, pw)
            tubelet = tubelet[0]

        Tg = T // tubelet
        Hg = H // patch
        Wg = W // patch
        N0 = Tg * Hg * Wg

        valid = (kept_global >= 0)
        idx = kept_global.clamp(min=0).long()
        w = valid.to(torch.float32)

        tube_mask = torch.zeros((kept_global.size(0), N0), device=kept_global.device, dtype=torch.float32)
        tube_mask.scatter_add_(1, idx, w)
        tube_mask = (tube_mask > 0).to(torch.float32)

        tube_mask = tube_mask.view(-1, 1, Tg, Hg, Wg)
        mask_video = F.interpolate(tube_mask, size=(T, H, W), mode="nearest")
        return mask_video

    # -------------------------------------------------------------
    # Forward Features
    # -------------------------------------------------------------
    def forward_features(self, x, keep_rate=None, tokens=None, get_idx=False):
        """
        Returns:
          act_cls_feat, priv_cls_feat, left_tokens(list), idxs_global(list of (B, K_i) global tubelet ids)
        """
        B, _, T, H, W = x.shape

        # normalize keep_rate/tokens to per-layer sequences
        if keep_rate is None:
            keep_rate = self.keep_rate
        if not isinstance(keep_rate, (tuple, list)):
            keep_rate = (keep_rate,) * self.depth
        if tokens is None or not isinstance(tokens, (tuple, list)):
            tokens = (tokens,) * self.depth
        assert len(keep_rate) == self.depth
        assert len(tokens) == self.depth

        # Patch embedding -> (B, N0, D)
        x_patch = self.patch_embed(x)
        B2, N0, D = x_patch.shape
        assert B2 == B

        # Track global tubelet ids aligned with current non-CLS tokens
        global_ids = torch.arange(N0, device=x.device).unsqueeze(0).expand(B, -1)  # (B, N0)

        # CLS tokens
        cls_act = self.cls_token_act.expand(B, -1, -1)
        cls_priv = self.cls_token_priv.expand(B, -1, -1)

        if self.dist_token is None:
            x_tok = torch.cat((cls_act, cls_priv, x_patch), dim=1)  # (B, 2+N0, D)
        else:
            dist = self.dist_token.expand(B, -1, -1)
            x_tok = torch.cat((cls_act, cls_priv, dist, x_patch), dim=1)  # (B, 3+N0, D)

        # Positional embedding (with your original resize logic)
        pos_embed = self.pos_embed
        if x_tok.shape[1] != pos_embed.shape[1]:
            assert H == W
            real_pos = pos_embed[:, self.num_tokens:]
            hw = int(math.sqrt(real_pos.shape[1]))
            true_hw = int(math.sqrt(x_tok.shape[1] - self.num_tokens))
            real_pos = real_pos.transpose(1, 2).reshape(1, self.embed_dim, hw, hw)
            new_pos = F.interpolate(real_pos, size=true_hw, mode="bicubic", align_corners=False)
            new_pos = new_pos.reshape(1, self.embed_dim, -1).transpose(1, 2)
            pos_embed = torch.cat([pos_embed[:, : self.num_tokens], new_pos], dim=1)

        x_tok = self.pos_drop(x_tok + pos_embed)

        left_tokens = []
        idxs_global = []

        # Force get_idx=True for tracking; your Block returns idx only when pruning triggers
        for i, blk in enumerate(self.blocks):
            x_tok, left_token, idx = blk(x_tok, keep_rate[i], tokens[i], get_idx=True)
            left_tokens.append(left_token)

            if idx is not None:
                # idx indexes into current non-CLS tokens (x_tok[:, 2:] assuming no dist token)
                # If distilled=True in your setup, you must ensure Block/Attention uses correct CLS offset.
                global_ids = torch.gather(global_ids, dim=1, index=idx)  # (B, K)

                if self.fuse_token:
                    # block appended one extra fused token at end of non-CLS; add sentinel -1
                    extra = torch.full((B, 1), -1, device=x.device, dtype=global_ids.dtype)
                    global_ids = torch.cat([global_ids, extra], dim=1)

                idxs_global.append(global_ids)

        x_tok = self.norm(x_tok)

        if self.dist_token is None:
            return self.pre_logits(x_tok[:, 0]), self.pre_logits(x_tok[:, 1]), left_tokens, idxs_global
        else:
            return x_tok[:, 0], x_tok[:, 1], x_tok[:, 2], idxs_global

    # -------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------
    def forward(self, x, keep_rate=None, tokens=None, get_idx=False, lambda_grl=1.0, return_masked_video=False):
        """
        If return_masked_video=True, returns x_masked shaped like input (B,C,T,H,W) for downstream I3D/ResNet.
        If get_idx=True, also returns idxs_global + mask_video.
        """
        if self.dist_token is None:
            x_act_feat, x_priv_feat, _, idxs_global = self.forward_features(x, keep_rate, tokens, get_idx=True)
        else:
            x_act_feat, x_priv_feat, _, idxs_global = self.forward_features(x, keep_rate, tokens, get_idx=True)

        # GRL on privacy branch (training only)
        x_priv_input = grad_reverse(x_priv_feat, lambda_=lambda_grl) if self.training else x_priv_feat

        x_act = self.head_act(x_act_feat)
        x_priv = self.head_priv(x_priv_input)

        mask_video, x_masked = None, None
        if return_masked_video:
            B, C, T, H, W = x.shape
            if len(idxs_global) > 0:
                kept_global = idxs_global[-1]  # (B, K) with possible -1 sentinel
                mask_video = self._kept_ids_to_mask_video(kept_global, T, H, W)
            else:
                mask_video = torch.ones((B, 1, T, H, W), device=x.device, dtype=torch.float32)

            x_masked = x * mask_video  # (B,C,T,H,W)

        # distilled head logic (keep your original behavior)
        if self.head_dist is not None:
            if self.training and not torch.jit.is_scripting():
                if get_idx:
                    return x_act, x_priv, idxs_global, mask_video, x_masked
                return x_act, x_priv
            else:
                _ = (x_act + x_priv) / 2

        if get_idx:
            return x_act, x_priv, idxs_global, mask_video, x_masked

        return x_act, x_priv, x_masked if return_masked_video else None        
    



def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head_act'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)



if __name__ == "__main__":
    import torch

    model = EViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=101,
        num_priv=5,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        keep_rate=(1, 1, 0.7, 0.5),
        fuse_token=True,
    )

    x = torch.randn(4, 3, 16, 224, 224)
    keep_rate = (1, 1, 0.7, 0.5)

    # IMPORTANT: ask for masked video
    y_act, y_priv, idxs_global, mask_video, x_masked = model(x,keep_rate=keep_rate,get_idx=True,return_masked_video=True)

    # token counts per pruning layer (global ids, may include -1 fused token)
    keep_counts = []
    for i, ids in enumerate(idxs_global):
        if ids is not None:
            # count valid tubelets only (exclude fused token sentinel -1)
            keep_counts.append(int((ids[0] >= 0).sum().item()))
    print("Kept tubelets per pruning block:", keep_counts)

    print("Action logits:", y_act.shape)          # (B, num_classes)
    print("Privacy logits:", y_priv.shape)        # (B, num_priv)
    print("Mask video:", mask_video.shape)        # (B, 1, T, H, W)
    print("Masked video:", x_masked.shape)        # (B, 3, T, H, W)

    for i, ids in enumerate(idxs_global):
        s = None if ids is None else tuple(ids.shape)
        print(f"Block {i}: kept_global shape {s}")

