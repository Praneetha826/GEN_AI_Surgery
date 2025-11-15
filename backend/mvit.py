# mvit_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

# ----------------------------
# Basic building blocks
# ----------------------------

def _pair(x):
    return (x, x)

class PatchEmbed(nn.Module):
    """
    Patch embedding via a conv layer. Returns tokens and spatial shape (H, W).
    """
    def __init__(self, img_size=224, patch_size=8, in_chans=3, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # note: output spatial size = img_size // patch_size
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)           # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, N=H'*W', C)
        return tokens, (H, W)

class PatchMerging(nn.Module):
    """
    Reduce spatial resolution by 2x (H, W -> H/2, W/2) and optionally increase channels
    Implemented using conv with kernel=2, stride=2 performed on feature map.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
    def forward(self, tokens, hw):
        # tokens: (B, N, C), hw: (H, W)
        B, N, C = tokens.shape
        H, W = hw
        x = tokens.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        x = self.down(x)                                # (B, C2, H/2, W/2)
        B, C2, H2, W2 = x.shape
        tokens2 = x.flatten(2).transpose(1, 2)          # (B, N2, C2)
        tokens2 = self.norm(tokens2)
        return tokens2, (H2, W2)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# ----------------------------
# Pooling Multi-Head Attention (KV pooling optional)
# ----------------------------
class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=4, kv_pool_stride: int = 1, qkv_bias=True, attn_drop=0., proj_drop=0.):
        """
        kv_pool_stride: if >1 then K and V are spatially pooled by that stride (avgpool)
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.kv_pool_stride = int(kv_pool_stride)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _pool_kv(self, tokens_kv, hw):
        # tokens_kv: (B, N, C), hw: (H, W)
        if self.kv_pool_stride <= 1:
            return tokens_kv, hw
        B, N, C = tokens_kv.shape
        H, W = hw
        x = tokens_kv.transpose(1, 2).reshape(B, C, H, W)    # (B, C, H, W)
        x = F.avg_pool2d(x, kernel_size=self.kv_pool_stride, stride=self.kv_pool_stride)
        B, C, H2, W2 = x.shape
        tokens2 = x.flatten(2).transpose(1, 2)               # (B, N2, C)
        return tokens2, (H2, W2)

    def forward(self, q_tokens, kv_tokens, kv_hw):
        """
        q_tokens: (B, Nq, C)
        kv_tokens: (B, Nk, C)  (may be pooled inside)
        kv_hw: (Hk, Wk)
        returns: (B, Nq, C)
        """
        B, Nq, C = q_tokens.shape
        # optionally pool K/V spatially
        kv_tokens_p, kv_hw_p = self._pool_kv(kv_tokens, kv_hw)
        Nk = kv_tokens_p.shape[1]

        q = self.q(q_tokens).reshape(B, Nq, self.num_heads, self.head_dim).permute(0,2,1,3)  # (B, nh, Nq, hd)
        k = self.k(kv_tokens_p).reshape(B, Nk, self.num_heads, self.head_dim).permute(0,2,1,3) # (B, nh, Nk, hd)
        v = self.v(kv_tokens_p).reshape(B, Nk, self.num_heads, self.head_dim).permute(0,2,1,3) # (B, nh, Nk, hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, nh, Nq, Nk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v)  # (B, nh, Nq, hd)
        out = out.permute(0,2,1,3).reshape(B, Nq, C)  # (B, Nq, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# ----------------------------
# Transformer Block with Pooling-Attn
# ----------------------------
class MViTBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, qkv_bias=True, drop=0., attn_drop=0., kv_pool_stride=1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = PoolingAttention(dim, num_heads=num_heads, kv_pool_stride=kv_pool_stride,
                                     qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_dim, drop=drop)

    def forward(self, x, hw):
        # x: (B, N, C)
        x = x + self.drop_path(self.attn(self.norm1(x), x, hw))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# ----------------------------
# DropPath (stochastic depth)
# ----------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = 0.0 if drop_prob is None else drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# ----------------------------
# The Multiscale Vision Transformer (MViT) Model
# ----------------------------
class MViTForMultiLabel(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=8,
                 in_chans=3,
                 num_classes=4,
                 embed_dims: List[int] = [64, 128, 256, 384],
                 num_blocks: List[int] = [2, 2, 4, 2],
                 num_heads: List[int] = [1, 2, 4, 8],
                 mlp_ratio: float = 2.0,
                 drop_rate: float = 0.1,
                 attn_drop_rate: float = 0.1,
                 drop_path_rate: float = 0.1,
                 use_aux_head: bool = False):
        """
        embed_dims: channels for each stage (will increase after patch merging)
        num_blocks: number of transformer blocks per stage
        num_heads: heads per stage
        """
        super().__init__()
        assert len(embed_dims) == len(num_blocks) == len(num_heads)
        self.num_stages = len(embed_dims)
        self.use_aux_head = use_aux_head
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dims[0])
        # build stages
        total_blocks = sum(num_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        cur = 0
        self.stages = nn.ModuleList()
        self.patch_mergings = nn.ModuleList()
        for stage_idx in range(self.num_stages):
            blocks = []
            for bi in range(num_blocks[stage_idx]):
                # choose kv pooling stride = 1 for earlier layers, maybe >1 in deeper blocks
                # a simple heuristic: use kv_pool_stride = 1 for stage0, 1 for stage1, 2 for stage2, 2 for stage3
                kv_pool_stride = 1
                if stage_idx >= 2:
                    kv_pool_stride = 2
                blk = MViTBlock(dim=embed_dims[stage_idx],
                                num_heads=num_heads[stage_idx],
                                mlp_ratio=mlp_ratio,
                                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                                kv_pool_stride=kv_pool_stride, drop_path=dpr[cur+bi])
                blocks.append(blk)
            cur += num_blocks[stage_idx]
            self.stages.append(nn.ModuleList(blocks))
            # patch merging between stages (except last)
            if stage_idx < self.num_stages - 1:
                in_ch = embed_dims[stage_idx]
                out_ch = embed_dims[stage_idx+1]
                self.patch_mergings.append(PatchMerging(in_ch, out_ch))
            else:
                self.patch_mergings.append(None)

        # final norm
        self.norm = nn.LayerNorm(embed_dims[-1], eps=1e-6)
        # head
        self.head = nn.Sequential(
            nn.Linear(embed_dims[-1], embed_dims[-1] // 2),
            nn.LayerNorm(embed_dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dims[-1] // 2, num_classes)
        )
        if self.use_aux_head:
            self.aux_head = nn.Linear(embed_dims[len(embed_dims)//2], num_classes)
        else:
            self.aux_head = None
        self._init_weights()

    def _init_weights(self):
        # simple initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_aux=False):
        # x: (B, C, H, W)
        tokens, hw = self.patch_embed(x)   # (B, N, C0), hw = (H0, W0)
        aux_out = None
        for s_idx, blocks in enumerate(self.stages):
            for b_idx, blk in enumerate(blocks):
                tokens = blk(tokens, hw)  # each block updates tokens
            # optionally collect aux features from middle stage after blocks
            if self.use_aux_head and s_idx == (self.num_stages // 2):
                # global average pooling of tokens to produce aux features
                feat = tokens.mean(dim=1)  # (B, C_stage)
                aux_out = self.aux_head(feat)
            # patch merging between stages (if present)
            if s_idx < (self.num_stages - 1):
                pm = self.patch_mergings[s_idx]
                tokens, hw = pm(tokens, hw)  # update tokens and new hw

        tokens = self.norm(tokens)    # (B, N, C_final)
        pooled = tokens.mean(dim=1)   # global average pooling across tokens
        out = self.head(pooled)       # logits (B, num_classes)

        if return_aux and (self.aux_head is not None) and (aux_out is not None):
            return out, aux_out
        return out

# ----------------------------
# Example instantiation
# ----------------------------
# if __name__ == "__main__":
#     # Quick sanity check
#     model = MViTForMultiLabel(
#         img_size=224,
#         patch_size=8,
#         in_chans=3,
#         num_classes=4,                    # predict first 4 instruments
#         embed_dims=[64, 128, 256, 384],
#         num_blocks=[2, 2, 4, 2],
#         num_heads=[1, 2, 4, 8],
#         mlp_ratio=2.0,
#         drop_rate=0.1,
#         attn_drop_rate=0.1,
#         drop_path_rate=0.1,
#         use_aux_head=True
#     )
#     print("Model created. Parameter count:", sum(p.numel() for p in model.parameters()))
#     dummy = torch.randn(2,3,224,224)
#     logits, aux = model(dummy, return_aux=True)
#     print("Logits shape:", logits.shape, "Aux shape:", aux.shape)
