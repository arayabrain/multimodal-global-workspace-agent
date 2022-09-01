import torch
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

STR_TO_ACTIVATION = {
    "identity": nn.Identity(),
    "tanh": nn.Tanh(),
    "ln": nn.LayerNorm(512)
}

# Helper classes
class SelfAttention(nn.Module):
    def __init__(self, h_size, n_heads=4):
        super().__init__()
        self.h_size = h_size
        self.n_heads = n_heads
        
        self.mha = nn.MultiheadAttention(
            h_size, n_heads, dropout=0.0, add_zero_attn=False, batch_first=True
        )
        self.ln = nn.LayerNorm([h_size])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([h_size]),
            nn.Linear(h_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, h_size)
        )

    def forward(self, x):
        x_ln = self.ln(x)
        attention_value, attn_weighting = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value, attn_weighting

class CrossAttention(nn.Module):
    def __init__(self, q_size, kv_size, n_heads=4, skip_q=False):
        super().__init__()
        self.h_size = q_size
        self.skip_q = skip_q
        self.n_heads = n_heads

        self.mha = nn.MultiheadAttention(
            q_size,
            n_heads,
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
            nn.Linear(q_size, q_size)
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
        cross_heads = 1, # LucidRains's implm. uses 1 by default
        latent_heads = 4, # LucidRains's implm. uses 8 by default
        # cross_dim_head = 64,
        # latent_dim_head = 64,
        # self_per_cross_attn = 1, # Number of self attention blocks per cross attn.
        # Modality embeddings realted
        hidden_size = 512, # Dim of the visual / audio encoder outputs
        mod_embed = 0, # Dimensio of learned modality embeddings
        use_sa = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_latents = num_latents # N
        self.latent_dim = latent_dim # D
        self.latent_type = latent_type
        self.latent_learned = latent_learned

        self.mod_embed = mod_embed
        self.hidden_size = hidden_size
        self.use_sa = use_sa
        # Cross Attention
        self.ca = CrossAttention(latent_dim, input_dim + mod_embed,
            n_heads=cross_heads, skip_q=True) # If not skipping, usually blows
        # Self Attention
        if self.use_sa:
            self.sa = SelfAttention(latent_dim, n_heads=latent_heads)
        # self.decoder = CrossAttention(self.h_size, self.s_size, skip_q=True)

        # Modality embedding
        if self.mod_embed:
            self.modality_embeddings = nn.Parameter(torch.randn(1, 2, mod_embed))
        
        # Latent vector, supposedly equivalent to an RNN's hidden state
        if latent_type == "randn":
            self.latents = torch.randn(1, num_latents, latent_dim)
            # As per original paper
            with th.no_grad():
                self.latents.normal_(0.0, 0.02).clamp_(-2.0,2.0)
        elif latent_type == "zeros":
            self.latents = torch.zeros(1, num_latents, latent_dim)
        
        self.latents = nn.Parameter(self.latents, requires_grad=latent_learned)

    def seq_forward(self, data, prev_latents, masks):
        # TODO: a more optimal method to process sequences of same length together ?
        x_list, latents_list = [], []

        T_B, feat_dim = data.shape
        B = prev_latents.shape[0]
        T = T_B // B # TODO: assert that T * B == T_B exactly
        latents = prev_latents.clone()

        data = data.reshape(T, B, feat_dim)
        masks = masks.reshape(T, B, 1)

        for t in range(T):
            x, latents = self.single_forward(data[t], latents, masks[t])

            x_list.append(x.clone())
            latents_list.append(latents.clone())
        
        x_list = th.stack(x_list, dim=0).flatten(start_dim=0, end_dim=1) # [B * T, feat_dim]
        latents_list = th.stack(latents_list, dim=0).flatten(start_dim=0, end_dim=1) # [B * T, num_latents, latent_dim]

        return x_list, latents_list

    def single_forward(self, data, prev_latents, masks):
        b = data.shape[0] # Batch size

        if data.dim() == 2:
            data = data.reshape(b, 2, self.hidden_size) # [B,1024] -> [B,2,512]
            
        if self.mod_embed:
            data = th.cat([data, self.modality_embeddings.repeat(b, 1, 1)], dim=2)
        
        # If the current step is the start of a new episode,
        # the the mask will contain 0
        prev_latents = masks[:, :, None] * prev_latents + \
            (1. - masks[:, :, None]) * self.latents.repeat(b, 1, 1)

        x = prev_latents
        
        # Cross Attention
        x, _ = self.ca(x, data) # x: [B, N * D], x_weights: [B, ???]

        # Self Attention
        if self.use_sa:
            x, _ = self.sa(x) # x: [B, N * D]
        
        return x.flatten(start_dim=1), x

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

# Old version that cats both modality, but with fix sequence processing
class Perceiver_GWT_GWWM_Legacy(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        latent_type = "randn",
        latent_learned = True,
        num_latents = 8,
        latent_dim = 64,
        cross_heads = 1, # LucidRains's implm. uses 1 by default
        latent_heads = 4, # LucidRains's implm. uses 8 by default
        # cross_dim_head = 64,
        # latent_dim_head = 64,
        # self_per_cross_attn = 1, # Number of self attention blocks per cross attn.
        # Modality embeddings realted
        hidden_size = 512, # Dim of the visual / audio encoder outputs
        mod_embed = 0, # Dimensio of learned modality embeddings
        use_sa = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_latents = num_latents # N
        self.latent_dim = latent_dim # D
        self.latent_type = latent_type
        self.latent_learned = latent_learned

        self.mod_embed = mod_embed
        self.hidden_size = hidden_size
        self.use_sa = use_sa
        # Cross Attention
        self.ca = CrossAttention(num_latents * latent_dim, (input_dim + mod_embed) * 2, 
            n_heads=cross_heads, skip_q=not use_sa) # skip_q if not using SA
        # Self Attention
        if self.use_sa:
            self.sa = SelfAttention(num_latents * latent_dim, n_heads = latent_heads)
        # self.decoder = CrossAttention(self.h_size, self.s_size, skip_q=True)

        # Modality embedding
        if self.mod_embed:
            self.modality_embeddings = nn.Parameter(torch.randn(1, 2, mod_embed))
        
        # Latent vector, supposedly equivalent to an RNN's hidden state
        if latent_type == "randn":
            self.latents = torch.randn(1, num_latents, latent_dim)
            # As per original paper
            with th.no_grad():
                self.latents.normal_(0.0, 0.02).clamp_(-2.0,2.0)
        elif latent_type == "zeros":
            self.latents = torch.zeros(1, num_latents, latent_dim)
        
        self.latents = nn.Parameter(self.latents, requires_grad=latent_learned)

    def seq_forward(self, data, prev_latents, masks):
        # TODO: a more optimal method to process sequences of same length together ?
        x_list, latents_list = [], []

        T_B, feat_dim = data.shape
        B = prev_latents.shape[0]
        T = T_B // B # TODO: assert that T * B == T_B exactly
        latents = prev_latents.clone()

        data = data.reshape(T, B, feat_dim)
        masks = masks.reshape(T, B, 1)

        for t in range(T):
            x, latents = self.single_forward(data[t], latents, masks[t])

            x_list.append(x.clone())
            latents_list.append(latents.clone())
        
        x_list = th.stack(x_list, dim=0).flatten(start_dim=0, end_dim=1) # [B * T, feat_dim]
        latents_list = th.stack(latents_list, dim=0).flatten(start_dim=0, end_dim=1) # [B * T, num_latents, latent_dim]

        return x_list, latents_list

    def single_forward(self, data, prev_latents, masks):
        b = data.shape[0] # Batch size

        if self.mod_embed:
            data = data.reshape(b, 2, self.hidden_size) # [B,1024] -> [B,2,512]
            data = th.cat([data, self.modality_embeddings.repeat(b, 1, 1)], dim=2)
            data = data.reshape(b, -1) # [B, 2, 512 + 128] -> [B, 2 * (512 + 128)]
        
        # If the current step is the start of a new episode,
        # the the mask will contain 0
        prev_latents = masks[:, :, None] * prev_latents + \
            (1. - masks[:, :, None]) * self.latents.repeat(b, 1, 1)

        x = prev_latents.flatten(start_dim=1)
        
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


class Perceiver_GWT_AttGRU(nn.Module):
    def __init__(
        self,
        *,
        input_dim,
        latent_type = "randn",
        latent_learned = True,
        num_latents = 8,
        latent_dim = 64,
        cross_heads = 1, # LucidRains's implm. uses 1 by default
        latent_heads = 4, # LucidRains's implm. uses 8 by default
        # cross_dim_head = 64,
        # latent_dim_head = 64,
        # self_per_cross_attn = 1, # Number of self attention blocks per cross attn.
        # Modality embeddings realted
        hidden_size = 512, # Dim of the visual / audio encoder outputs
        mod_embed = 0, # Dimensio of learned modality embeddings
        use_sa = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_latents = num_latents # N
        self.latent_dim = latent_dim # D
        self.latent_type = latent_type
        self.latent_learned = latent_learned

        self.mod_embed = mod_embed
        self.hidden_size = hidden_size
        self.use_sa = use_sa

        # Modality embedding
        if self.mod_embed:
            self.modality_embeddings = nn.Parameter(torch.randn(1, 2, mod_embed))
        
        # Latent vector, supposedly equivalent to an RNN's hidden state
        if latent_type == "randn":
            self.latents = torch.randn(1, num_latents, latent_dim)
            # As per original paper
            with th.no_grad():
                self.latents.normal_(0.0, 0.02).clamp_(-2.0,2.0)
        elif latent_type == "zeros":
            self.latents = torch.zeros(1, num_latents, latent_dim)
        
        self.latents = nn.Parameter(self.latents, requires_grad=latent_learned)

        # Cross Attention
        self.q_ln, self.kv_ln = nn.LayerNorm(latent_dim), nn.LayerNorm(input_dim + mod_embed)
        self.r_ca = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=cross_heads,
            kdim=input_dim + mod_embed, vdim=input_dim + mod_embed, batch_first=True)
        self.z_ca = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=cross_heads,
            kdim=input_dim + mod_embed, vdim=input_dim + mod_embed, batch_first=True)
        self.n_ca = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=cross_heads,
            kdim=input_dim + mod_embed, vdim=input_dim + mod_embed, batch_first=True)
        
        # Self Attention
        # if self.use_sa:
        #     self.sa = SelfAttention(latent_dim, n_heads=latent_heads)
        # self.decoder = CrossAttention(self.h_size, self.s_size, skip_q=True)

    def seq_forward(self, data, prev_latents, masks):
        # TODO: a more optimal method to process sequences of same length together ?
        x_list, latents_list = [], []

        T_B, feat_dim = data.shape
        B = prev_latents.shape[0]
        T = T_B // B # TODO: assert that T * B == T_B exactly
        latents = prev_latents.clone()

        data = data.reshape(T, B, feat_dim)
        masks = masks.reshape(T, B, 1)

        for t in range(T):
            x, latents = self.single_forward(data[t], latents, masks[t])

            x_list.append(x.clone())
            latents_list.append(latents.clone())
        
        x_list = th.stack(x_list, dim=0).flatten(start_dim=0, end_dim=1) # [B * T, feat_dim]
        latents_list = th.stack(latents_list, dim=0).flatten(start_dim=0, end_dim=1) # [B * T, num_latents, latent_dim]

        return x_list, latents_list

    def single_forward(self, data, prev_latents, masks):
        b = data.shape[0] # Batch size

        if data.dim() == 2:
            data = data.reshape(b, 2, self.hidden_size) # [B,1024] -> [B,2,512]
            
        if self.mod_embed:
            data = th.cat([data, self.modality_embeddings.repeat(b, 1, 1)], dim=2)
        
        # If the current step is the start of a new episode,
        # the the mask will contain 0
        prev_latents = masks[:, :, None] * prev_latents + \
            (1. - masks[:, :, None]) * self.latents.repeat(b, 1, 1)

        x = prev_latents
        
        # Cross Attention
        norm_input, norm_hidden = self.kv_ln(data), self.q_ln(x)  # Concatenate embeddings and apply normalisation
        reset = torch.sigmoid(self.r_ca(norm_hidden, norm_input, norm_input, need_weights=False)[0])  # Apply attention and unlike typical Pre-LN Transformer, do not apply residual
        update = torch.sigmoid(self.z_ca(norm_hidden, norm_input, norm_input, need_weights=False)[0])
        new = torch.tanh(self.n_ca(reset * norm_hidden, norm_input, norm_input, need_weights=False)[0])
        x = (1 - update) * new + update * x

        # Self Attention
        # if self.use_sa:
        #     x, _ = self.sa(x) # x: [B, N * D]

        return x.flatten(start_dim=1), x

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
