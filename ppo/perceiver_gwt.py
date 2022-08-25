# Based on https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
# This variant aims to expose the Perceiver's internal latent as the hidden state of an RNN
# - does not support direct feed of 2D images nor 2D audio
# - disregards fourier encoding, as we use VisualEncoder and AudioEncoder from the SS baseline
import torch
import torch as th
from torch import nn, einsum
import torch.nn.functional as F

from functools import wraps

from einops import rearrange, repeat

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
        input_dim,
        latent_type = "randn",
        latent_learned = True,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        attn_dropout = 0.,
        ff_dropout = 0.,
        self_per_cross_attn = 1, # Number of self attention blocks per cross attn.
        weight_tie_layers = False
    ):
        super().__init__()
        self.input_dim = input_dim

        # Latent vector, supposedly equivalent to an RNN's hidden state
        if latent_type == "randn":
            self.latents = torch.randn(num_latents, latent_dim)
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

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

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
        b, device, dtype = data.shape[0], data.device, data.dtype
        
        if data.dim() == 2:
            data = data[:, None, :] # [NUM_ENVS, feat_dim] -> [NUM_ENVS, 1, feat_dim] as expected by Perceiver
            masks = masks[:, :, None] # [NUM_ENVS, 1] -> [NUM_ENVS, 1, 1] to match prev_latnets.shape
        # If the current step is the start of a new episode,
        # the the mask will contain 0
        prev_latents = masks * prev_latents + \
            (1. - masks) * repeat(self.latents.clone(), 'n d -> b n d', b = b)
        
        x = prev_latents

        # Apply cross-attention and self-attention layers successively
        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = None) + x
            x = cross_ff(x) + x

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


