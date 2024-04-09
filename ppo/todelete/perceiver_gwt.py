# Based on https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
# This variant aims to expose the Perceiver's internal latent as the hidden state of an RNN
# - does not support direct feed of 2D images, audio etc... Only pre-processed features
import torch
import torch as th
from torch import nn, einsum
import torch.nn.functional as F

from functools import wraps

from einops import rearrange, repeat

from math import pi

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = dict()
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if key in cache:
            return cache[key]
        result = f(*args, **kwargs)
        cache[key] = result
        return result
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# Main class(es)
class Perceiver_GWT(nn.Module):
    def __init__(
        self,
        *,
        depth,
        input_channels,
        latent_type = "randn",
        latent_learned = True,
        num_latents = 8,
        latent_dim = 64,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        self_per_cross_attn = 1, # Number of self attention blocks per cross attn.
        weight_tie_layers = False,
        # FF related
        max_freq = 10.,
        num_freq_bands = 6,
        fourier_encode_data = False,
        input_axis = 1,
        # Modality embeddings realted
        hidden_size = 512, # Dim of the visual / audio encoder outputs
        mod_embed = 0, # Dimensio of learned modality embeddings
        use_sa = True
    ):
        super().__init__()
        self.input_channels = input_channels
        self.mod_embed = mod_embed
        self.hidden_size = hidden_size
        self.use_sa = use_sa

        # NOTE: Using FF and modality embedding together ?

        # Fourier Encode related
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands
        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

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

        # Defines the cross-attention and self-attention layers
        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        # Populate cross-attention and self-attention layers
        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])
            
            if self.use_sa:
                for block_ind in range(self_per_cross_attn):
                    self_attns.append(nn.ModuleList([
                        get_latent_attn(**cache_args, key = block_ind),
                        get_latent_ff(**cache_args, key = block_ind)
                    ]))
            else:
                self_attns.append(nn.Identity())

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

    def seq_forward(self, data, prev_latents, masks):
        # TODO: a more optimal method to process sequences of same length together ?
        x_list, latents_list = [], []

        B_T, feat_dim = data.shape
        B = prev_latents.shape[0]
        T = B_T // B # TODO: assert that B * T == B_T
        latents = prev_latents

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
        
        if data.dim() == 2:
            # data = data[:, :, None] # From [B, feat_dim] -> [B ,feat_dim, 1]
            # data = data[:, None, :] # [B, feat_dim] -> [B, 1, feat_dim]
            pass
        
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        # assert len(axis) == self.input_axis, 'input data must have the right number of axis'
        
        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)

            data = torch.cat((data, enc_pos), dim = -1)

        # concat to channels of data and flatten axis
        # data = rearrange(data, 'b ... d -> b (...) d')

        # If the current step is the start of a new episode,
        # the the mask will contain 0
        prev_latents = masks[:, :, None] * prev_latents + \
            (1. - masks[:, :, None]) * repeat(self.latents, 'n d -> b n d', b = b)
        
        x = prev_latents

        # Apply cross-attention and self-attention layers successively
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = None) + x
            x = cross_ff(x) + x

            if self.use_sa:
                for self_attn, self_ff in self_attns:
                    x = self_attn(x) + x
                    x = self_ff(x) + x
        
        return x.flatten(start_dim=1), x # state_feat, latents
    
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
