# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import List, Optional, Sequence
from util.model_util import VisionRotaryEmbeddingFast, get_2d_sincos_pos_embed, RMSNorm,RotaryEmbedding1DFast
import scipy.stats as stats
from timm.models.vision_transformer import PatchEmbed,Attention

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    
class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x

class BottleneckMaskPatchEmbed(nn.Module):
    """ Image to Patch Embedding + (optional) per-sample random patch masking for t>0.8 """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        pca_dim=768,
        embed_dim=768,
        bias=True,
        mask_ratio=0.75,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.mask_ratio = mask_ratio
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches = num_patches
        self.img_size = img_size
        self.patch_size = patch_size
        pos_embed = get_2d_sincos_pos_embed(embed_dim, int(num_patches ** 0.5))
        self.pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)

        self.x_embedder = PatchEmbed(img_size, patch_size, in_chans, embed_dim, bias=True)

        # self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        # self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)
    def forward(self, x, use_mask=True):
        """
        x: (B, C, H, W)
        Returns:
            tokens: (B, N, D)
            mask:   (B, N) bool
        """
        B, C, H, W = x.shape
        # tokens = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)  # (B, N, D)
        tokens = self.x_embedder(x)
        B, N, D = tokens.shape
        tokens = tokens + self.pos_embed.to(tokens.device)

        # -------- per-sample mask ratio: left-half truncated Gaussian --------
        num_mask = int(self.mask_ratio * N)
        mask = torch.zeros(B, N, device=tokens.device, dtype=torch.bool)
        num_keep = N - num_mask
        if use_mask:
            # random per-sample masking
            scores = torch.rand(B, N, device=tokens.device)
            mask_idx = torch.topk(scores, k=num_mask, dim=1, largest=True).indices
            mask.scatter_(1, mask_idx, True)
            keep_idx = torch.topk(scores, k=num_keep, dim=1, largest=False).indices
            # replace masked tokens with learnable mask token
            tokens = tokens.gather(1, keep_idx.unsqueeze(-1).expand(-1, -1, D))
            return tokens, mask, keep_idx
        else:
            return tokens,None,None



class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


# def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
#     L, S = query.size(-2), key.size(-2)
#     scale_factor = 1 / math.sqrt(query.size(-1))
#     # attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype).cuda()
#     attn_bias = torch.zeros(query.size(0), 1, L, S, dtype=query.dtype, device=query.device)

#     with torch.cuda.amp.autocast(enabled=False):
#         attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
#     attn_weight += attn_bias
#     attn_weight = torch.softmax(attn_weight, dim=-1)
#     attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
#     return attn_weight @ value

def scaled_dot_product_attention(query, key, value, dropout_p=0.0) -> torch.Tensor:
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
        )

    # Fallback for older PyTorch versions.
    scale_factor = 1 / math.sqrt(query.size(-1))
    with torch.cuda.amp.autocast(enabled=False):
        attn_weight = query.float() @ key.float().transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1).to(dtype=value.dtype)
    if dropout_p > 0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = rope(q)
        k = rope(k)

        x = scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(self.ffn_dropout(hidden))


class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class JiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        # )

    @torch.compile
    def forward(self, x,  c, feat_rope=None):
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + self.attn(self.norm1(x), rope=feat_rope)
        x = x + self.mlp(self.norm2(x))
        return x

class JiTBlock_cond(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    @torch.compile
    def forward(self, x,  c, feat_rope=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class ProjLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Sequential(
                nn.Linear(hidden_size, 1024),
                RMSNorm(1024, eps=1e-6),
                nn.Linear(1024, 1024)
            )
        # self.norm = nn.LayerNorm(hidden_size)

    @torch.compile
    def forward(self, x):
        x = self.linear(x)
        # x = self.norm(x)
        return x

class JiT(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=0,
        in_context_start=8,
        teacher_feature_layers=None,
        teacher_cons_layer=7,
        teacher_repa_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        if teacher_feature_layers is None:
            teacher_feature_layers = [5, 12, 19, 26]
        self.teacher_feature_layers = list(teacher_feature_layers)
        self.teacher_cons_layer = teacher_cons_layer
        # Optional 0-based block index for REPA (DINOv2) alignment; None disables.
        self.teacher_repa_layer = teacher_repa_layer

        # time and class embed
        # self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)

        # # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # in-context cls token
        if self.in_context_len > 0:
            self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
            torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        half_head_dim = hidden_size // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        if self.in_context_len > 0:
            self.feat_rope_incontext = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
                num_cls_token=self.in_context_len
            )
        else:
            self.feat_rope_incontext = None

        # transformer
        self.depth = depth
        self.blocks = nn.ModuleList([
            JiTBlock_cond(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict（仅 cosine consistency 分支需要）
        self.proj = ProjLayer(hidden_size) if teacher_cons_layer is not None else None
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, p):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        h = w = int(x.shape[1] ** 0.5)
        # print(h,w,x.shape)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
        self,
        x,
        t,
        y,
        compute_features=True,
        compute_fea_cons=True,
        compute_fea_repa=False,
    ):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        compute_features: 是否收集 DIFT 多层特征（采样时可关）
        compute_fea_cons: 是否跑 ProjLayer 得到 fea_cons（cosine=0 时可关）
        compute_fea_repa: 是否返回指定层的 patch token（REPA；weight=0 时可关）
        """
        # class and time embeddings
        # t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = y_emb

        # forward JiT
        x = self.x_embedder(x)
        B,L,C = x.shape
        x += self.pos_embed
        fea = []
        fea_cons = None
        fea_repa = None
        for i, block in enumerate(self.blocks):
            # in-context
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = y_emb.unsqueeze(1).repeat(1, self.in_context_len, 1)
                in_context_tokens += self.in_context_posemb
                x = torch.cat([in_context_tokens, x], dim=1)
            rope = (
                self.feat_rope_incontext
                if self.in_context_len > 0 and i >= self.in_context_start
                else self.feat_rope
            )
            x = block(x, c, rope)
            if compute_features and i in self.teacher_feature_layers:
                fea.append(x[:, -L:])
            if compute_fea_cons and self.proj is not None and i == self.teacher_cons_layer:
                fea_cons = self.proj(x)
            if (
                compute_fea_repa
                and self.teacher_repa_layer is not None
                and i == self.teacher_repa_layer
            ):
                # Patch tokens only (drop optional in-context tokens), matching DINOv2 T.
                fea_repa = x[:, -L:]
        x = x[:, self.in_context_len:]
        x = self.final_layer(x, c)
        output = self.unpatchify(x, self.patch_size)

        if compute_fea_cons and fea_cons is None:
            raise RuntimeError(
                f"teacher_cons_layer={self.teacher_cons_layer} was not reached "
                f"(depth={self.depth}, proj={'set' if self.proj is not None else 'None'}). "
                f"Check --teacher_cons_layer / distill_cosine_weight."
            )
        if compute_fea_repa and fea_repa is None:
            raise RuntimeError(
                f"teacher_repa_layer={self.teacher_repa_layer} was not reached "
                f"(depth={self.depth}). Check --repa_encoder_depth / repa_loss_weight."
            )
        fea_out = torch.cat(fea, dim=-1) if compute_features else None
        return output, fea_out, fea_cons, fea_repa
    

class JiT_student(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        bottleneck_dim=128,
        in_context_len=0,
        in_context_start=8,
        mask_ratio=0.75,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes

        # time and class embed
        # self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        # linear embed
        self.x_embedder = BottleneckMaskPatchEmbed(
            input_size,
            patch_size,
            in_channels,
            bottleneck_dim,
            hidden_size,
            bias=True,
            mask_ratio=mask_ratio,
        )


        # in-context cls token
        # if self.in_context_len > 0:
        #     self.in_context_posemb = nn.Parameter(torch.zeros(1, self.in_context_len, hidden_size), requires_grad=True)
        #     torch.nn.init.normal_(self.in_context_posemb, std=.02)

        # rope
        self.feat_rope = RotaryEmbedding1DFast(dim=hidden_size // num_heads, max_seq_len=2048, num_cls_token=0)

        # transformer
        self.blocks = nn.ModuleList([
            JiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                     proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0)
            for i in range(depth)
        ])

        # linear predict
        # self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # self.proj = ProjLayer(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # w1 = self.x_embedder.proj1.weight.data
        # nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        # w2 = self.x_embedder.proj2.weight.data
        # nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        # nn.init.constant_(self.x_embedder.proj2.bias, 0)
        w = self.x_embedder.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        # nn.init.constant_(self.final_layer.linear.weight, 0)
        # nn.init.constant_(self.final_layer.linear.bias, 0)
        
        # nn.init.constant_(self.proj.linear.weight, 0)
        # nn.init.constant_(self.proj.linear.bias, 0)

    def _resolve_feature_layers(self, feature_layers: Sequence[int]) -> List[int]:
        num_blocks = len(self.blocks)
        resolved = []
        for fl in feature_layers:
            layer_idx = int(fl)
            if layer_idx < 0:
                layer_idx = num_blocks + layer_idx
            if not (0 <= layer_idx < num_blocks):
                raise IndexError(
                    f"student feature layer index {fl} resolves to {layer_idx}, "
                    f"out of range for student depth {num_blocks}"
                )
            resolved.append(layer_idx)
        return resolved

    def forward(self, x, t, y, use_mask=True, feature_layers: Optional[Sequence[int]] = None):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        If ``feature_layers`` is set, selected block outputs are concatenated on the last dim.
        Masking is applied in patch embed, so all layers share the same kept tokens.
        """
        # class and time embeddings
        # t_emb = self.t_embedder(t)
        # y_emb = self.y_embedder(y)
        c = t

        # forward JiT
        x,mask,keep_idx = self.x_embedder(x,use_mask=use_mask)

        ordered = None
        capture_blocks = None
        if feature_layers is not None and len(feature_layers) > 0:
            ordered = self._resolve_feature_layers(feature_layers)
            capture_blocks = set(ordered)

        last_idx = len(self.blocks) - 1
        layer_feats = {}
        for i, block in enumerate(self.blocks):
            x = block(x, c, self.feat_rope)
            if use_mask and i == last_idx:
                x = self.norm(x)
            if capture_blocks is not None and i in capture_blocks:
                layer_feats[i] = x

        if ordered is not None:
            x = torch.cat([layer_feats[j] for j in ordered], dim=-1)

        return x,mask,keep_idx

class Projector(nn.Module):
    """
    Just image Transformer.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=1000,
        in_context_len=0,
        in_context_start=8,
        block_dim=384,
        num_blocks=8,
        out_dim=4608,
        input_hidden_size=None,
        t_embed_dim=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.in_context_len = in_context_len
        self.in_context_start = in_context_start
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.mlp_ratio = mlp_ratio
        in_hidden = hidden_size if input_hidden_size is None else input_hidden_size
        t_dim = block_dim if t_embed_dim is None else t_embed_dim
        # use fixed sin-cos embedding
        self.num_patches = (input_size // patch_size) * (input_size // patch_size)
        self.block_dim = block_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.block_dim), requires_grad=False)
        self.t_embedder = TimestepEmbedder(t_dim)
        if t_dim == block_dim:
            self.t_to_block = nn.Identity()
        else:
            self.t_to_block = nn.Linear(t_dim, block_dim)

        # rope (block_dim must be divisible by num_heads)
        if block_dim % num_heads != 0:
            raise ValueError(
                f"projector block_dim ({block_dim}) must be divisible by num_heads ({num_heads})."
            )
        half_head_dim = self.block_dim // num_heads // 2
        hw_seq_len = input_size // patch_size
        self.feat_rope = VisionRotaryEmbeddingFast(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0
        )
        drop_depth = max(depth, num_blocks)
        self.blocks = nn.ModuleList([
            JiTBlock(self.block_dim, num_heads, mlp_ratio=mlp_ratio,
                     attn_drop=attn_drop if (drop_depth // 4 * 3 > i >= drop_depth // 4) else 0.0,
                     proj_drop=proj_drop if (drop_depth // 4 * 3 > i >= drop_depth // 4) else 0.0)
            for i in range(num_blocks)
        ])

        # linear predict
        self.in_bottle = nn.Linear(in_hidden, self.block_dim)
        self.out_bottle = nn.Linear(self.block_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, 384))
        # nn.init.normal_(self.mask_token, std=0.02)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # for block in self.blocks:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
    
    def restore_with_mask_token(self,tokens_kept,keep_idx,t):
        
        B, N_keep, D = tokens_kept.shape
        # ---- init full sequence with mask_token ----
        t = t.reshape(B).to(device=tokens_kept.device, dtype=torch.float32)
        cond_embed = self.t_to_block(self.t_embedder(t)).to(tokens_kept.dtype)
        tokens_full = cond_embed.unsqueeze(1).expand(B, self.num_patches, D).clone()
        # ---- scatter kept tokens back ----
        tokens_full.scatter_(1,keep_idx.unsqueeze(-1).expand(-1, -1, D),tokens_kept)

        return tokens_full

        
    def forward(self, x, keep_idx,t):
        """
        x: (N, C, H, W)
        t: (N,)
        y: (N,)
        """

        # forward JiT
        t = t.reshape(x.shape[0]).to(device=x.device, dtype=torch.float32)
        c = self.t_embedder(t).to(x.dtype)
        x = self.in_bottle(x)
        x = self.restore_with_mask_token(x,keep_idx,t)
        x += self.pos_embed

        for i, block in enumerate(self.blocks):
            # in-context
            x = block(x, c, self.feat_rope)
            
        x = self.out_bottle(x)
        x = self.norm(x)

        return x

def _factory_kwargs(defaults, **kwargs):
    """Merge registry defaults with CLI overrides (kwargs wins)."""
    merged = defaults.copy()
    merged.update(kwargs)
    return merged


def JiT_B_16(**kwargs):
    return JiT(**_factory_kwargs(
        dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128,
             in_context_len=32, in_context_start=4, patch_size=16),
        **kwargs))

def JiT_B_32(**kwargs):
    return JiT(**_factory_kwargs(
        dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128,
             in_context_len=32, in_context_start=4, patch_size=32),
        **kwargs))

def JiT_L_16(**kwargs):
    return JiT(**_factory_kwargs(
        dict(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128,
             in_context_len=32, in_context_start=8, patch_size=16),
        **kwargs))

def JiT_L_32(**kwargs):
    return JiT(**_factory_kwargs(
        dict(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128,
             in_context_len=32, in_context_start=8, patch_size=32),
        **kwargs))

def JiT_XL_16(**kwargs):
    return JiT(**_factory_kwargs(
        dict(depth=28, hidden_size=1152, num_heads=16, bottleneck_dim=128,
             in_context_len=32, in_context_start=8, patch_size=16),
        **kwargs))

def JiT_H_16(**kwargs):
    return JiT(**_factory_kwargs(
        dict(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256,
             in_context_len=32, in_context_start=10, patch_size=16),
        **kwargs))

def JiT_H_32(**kwargs):
    return JiT(**_factory_kwargs(
        dict(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256,
             in_context_len=32, in_context_start=10, patch_size=32),
        **kwargs))

def JiT_student_B_16(**kwargs):
    return JiT_student(**_factory_kwargs(
        dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128,
             in_context_len=32, in_context_start=4, patch_size=16),
        **kwargs))

def JiT_student_B_32(**kwargs):
    return JiT_student(**_factory_kwargs(
        dict(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128,
             in_context_len=32, in_context_start=4, patch_size=32),
        **kwargs))

def JiT_student_L_16(**kwargs):
    return JiT_student(**_factory_kwargs(
        dict(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128,
             in_context_len=32, in_context_start=8, patch_size=16),
        **kwargs))

def JiT_student_L_32(**kwargs):
    return JiT_student(**_factory_kwargs(
        dict(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128,
             in_context_len=32, in_context_start=8, patch_size=32),
        **kwargs))

def JiT_student_H_16(**kwargs):
    return JiT_student(**_factory_kwargs(
        dict(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256,
             in_context_len=32, in_context_start=10, patch_size=16),
        **kwargs))

def JiT_student_H_32(**kwargs):
    return JiT_student(**_factory_kwargs(
        dict(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256,
             in_context_len=32, in_context_start=10, patch_size=32),
        **kwargs))


def Projector_B_16(**kwargs):
    return Projector(**_factory_kwargs(
        dict(depth=12, hidden_size=768, num_heads=12,
             in_context_len=32, in_context_start=4, patch_size=16),
        **kwargs))


def Projector_L_16(**kwargs):
    return Projector(**_factory_kwargs(
        dict(depth=12, hidden_size=1024, num_heads=16,
             in_context_len=32, in_context_start=4, patch_size=16),
        **kwargs))
JiT_models = {
    'JiT-B/16': JiT_B_16,
    'JiT-B/32': JiT_B_32,
    'JiT-L/16': JiT_L_16,
    'JiT-L/32': JiT_L_32,
    'JiT-XL/16': JiT_XL_16,
    'JiT-H/16': JiT_H_16,
    'JiT-H/32': JiT_H_32,
}
JiT_models_student = {
    'JiT-B/16': JiT_student_B_16,
    'JiT-B/32': JiT_student_B_32,
    'JiT-L/16': JiT_student_L_16,
    'JiT-L/32': JiT_student_L_32,
    'JiT-XL/16': JiT_student_L_16,
    'JiT-H/16': JiT_student_H_16,
    'JiT-H/32': JiT_student_H_32,
}
Projector_models = {
    'JiT-B/16': Projector_B_16,
    'JiT-B/32': Projector_B_16,
    'JiT-L/16': Projector_B_16,
    'JiT-L/32': Projector_B_16,
    'JiT-XL/16': Projector_L_16,
    'JiT-H/16': Projector_B_16,
    'JiT-H/32': Projector_B_16,
}