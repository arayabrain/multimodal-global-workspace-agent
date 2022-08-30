import torch
import torch as th
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat

# Helper classes
class SelfAttention(nn.Module):
    def __init__(self, h_size):
        super(SelfAttention, self).__init__()
        self.h_size = h_size
        self.mha = nn.MultiheadAttention(
            h_size, 4, dropout=0.0, add_zero_attn=False, batch_first=True
        )
        self.ln = nn.LayerNorm([h_size])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([h_size]),
            nn.Linear(h_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, h_size),
        )

    def forward(self, x):
        x_ln = self.ln(x)
        attention_value, attn_weighting = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value, attn_weighting


class CrossAttention(nn.Module):
    def __init__(self, q_size, kv_size, skip_q=False):
        super(CrossAttention, self).__init__()
        self.h_size = q_size
        self.skip_q = skip_q
        self.mha = nn.MultiheadAttention(
            q_size,
            4,
            dropout=0.0,
            add_zero_attn=False,
            batch_first=True,
            kdim=kv_size,
            vdim=kv_size,
        )
        self.ln_q = nn.LayerNorm([q_size])
        self.ln_kv = nn.LayerNorm([kv_size])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([q_size]),
            nn.Linear(q_size, q_size),
            nn.GELU(),
            nn.Linear(q_size, q_size),
        )

    def forward(self, q, kv):
        q_ln = self.ln_q(q)
        kv_ln = self.ln_kv(kv)
        attention_value, attn_weighting = self.mha(q_ln, kv_ln, kv_ln)
        if not self.skip_q:
            attention_value = attention_value + q
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value, attn_weighting

# Main class(es)
class Perceiver_GWT_GWWM(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        latent_type = "randn",
        latent_learned = True,
        num_latents = 8,
        latent_dim = 64,
        # cross_heads = 1,
        # latent_heads = 8,
        # cross_dim_head = 64,
        # latent_dim_head = 64,
        # self_per_cross_attn = 1, # Number of self attention blocks per cross attn.
        # Modality embeddings realted
        hidden_size = 512, # Dim of the visual / audio encoder outputs
        mod_embed = 0, # Dimensio of learned modality embeddings
        use_sa = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_latents = num_latents # N
        self.latent_dim = latent_dim # D

        self.mod_embed = mod_embed
        self.hidden_size = hidden_size
        self.use_sa = use_sa
        
        # Cross Attention
        self.ca = CrossAttention(num_latents * latent_dim, input_dim + mod_embed * 2, skip_q=False) # skip_q if not using SA
        # Self Attention
        if self.use_sa:
            self.sa = SelfAttention(num_latents * latent_dim)
        # self.decoder = CrossAttention(self.h_size, self.s_size, skip_q=True)

        # Modality embedding
        if self.mod_embed:
            self.modality_embeddings = nn.Parameter(0.1 * torch.randn(1, mod_embed * 2))
        
        # Latent vector, supposedly equivalent to an RNN's hidden state
        if latent_type == "randn":
            self.latents = torch.randn(num_latents, latent_dim)
            # As per original paper
            with th.no_grad():
                self.latents.normal_(0.0, 0.02).clamp_(-2.0,2.0)
        elif latent_type == "zeros":
            self.latents = torch.zeros(num_latents, latent_dim)
        
        self.latents = nn.Parameter(self.latents, requires_grad=latent_learned)

    def seq_forward(self, data, prev_latents, masks):
        # TODO: a more optimal method to process sequences of same length together ?
        x_list, latents_list = [], []

        B_T, feat_dim = data.shape
        B = prev_latents.shape[0]
        T = B_T // B # TODO: assert that B * T == B_T
        latents = prev_latents.clone()

        data = data.reshape(B, T, feat_dim)
        masks = masks.reshape(B, T, 1)

        for t in range(T):
            x, latents = self.single_forward(data[:, t], latents, masks[:, t])

            x_list.append(x)
            latents_list.append(latents)
        
        # TODO: debug
        x_list = th.stack(x_list, dim=0).flatten(start_dim=0, end_dim=1) # [B * T, feat_dim]
        latents_list = th.stack(latents_list, dim=0).flatten(start_dim=0, end_dim=1) # [B * T, num_latents, latent_dim]

        return x_list, latents_list

    def single_forward(self, data, prev_latents, masks):
        if self.mod_embed:
            b = data.shape[0]
            data = th.cat([
                data[:, :self.hidden_size], self.modality_embeddings[:, :self.mod_embed].repeat(b, 1), # Audio feats and embeddings
                data[:, self.hidden_size:], self.modality_embeddings[:, self.mod_embed:].repeat(b, 1), # Video feats and embeddings
            ], dim=-1)

        b, device, dtype = data.shape[0], data.device, data.dtype
        
        # If the current step is the start of a new episode,
        # the the mask will contain 0
        prev_latents = masks[:, :, None] * prev_latents + \
            (1. - masks[:, :, None]) * repeat(self.latents, 'n d -> b n d', b = b)

        x = prev_latents.flatten(start_dim=1) # [B, N, D] -> [B, N * D]
        
        # Cross Attention
        x, _ = self.ca(x, data) # x: [B, N * D], x_weights: [B, ???]

        # Self Attention
        if self.use_sa:
            x, _ = self.sa(x) # x: [B, N * D]

        return x, x.view(b, self.num_latents, self.latent_dim)

    def forward(self, data, prev_latents, masks):
        """
            - data: observation features [NUM_ENVS, feat_dim] or [NUM_ENVS, NUM_STEPS, feat_dim]
            - prev_latents: previous latents [B, num_latents, latent_dim]
            - masks: not Perceiver mask, but end-of-episode signaling mask
                - shape of [NUM_ENVS, 1] if single step forward
                - shape of [NUM_ENVS, NUM_STEPS, 1] if sequence forward
        """
        if data.size(0) == prev_latents.size(0):
            return self.single_forward(data, prev_latents, masks)
        else:
            return self.seq_forward(data, prev_latents, masks)
