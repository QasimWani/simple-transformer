import numpy as np
import torch
import torch.functional as F
import torch.nn as nn

# Simple implementation of decoder-only transformer, such as GPT
global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SingleHeadedAttention(nn.Module):
    
    def __init__(self, d_embed):
        super().__init__()
        self.query_proj = nn.Linear(d_embed, d_embed, bias=False)
        self.key_proj = nn.Linear(d_embed, d_embed, bias=False)
        self.value_proj = nn.Linear(d_embed, d_embed, bias=False)
        
        self.w_out = nn.Linear(d_embed, d_embed, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_embed = x.shape
        
        Q = self.query_proj(x)  # (batch_size, seq_len, d_embed)
        K = self.key_proj(x)  # (batch_size, seq_len, d_embed)
        V = self.value_proj(x)  # (batch_size, seq_len, d_embed)
        
        K = K.transpose(1, 2)  # (batch_size, d_embed, seq_len)
        
        scores = (Q @ K) / d_embed ** 0.5  # (batch_size, seq_len, seq_len)
        attention = torch.softmax(scores, dim=-1) @ V  # (batch_size, seq_len, d_embed)
        
        out = self.w_out(attention)  # (batch_size, seq_len, d_embed)
        return out


class MultiHeadedAttention(nn.Module):

    def __init__(self, d_embed, num_heads, max_seq_len):
        super().__init__()
        self.head_size = d_embed // num_heads
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        self.query_proj = nn.Linear(d_embed, d_embed, bias=False)
        self.key_proj = nn.Linear(d_embed, d_embed, bias=False)
        self.value_proj = nn.Linear(d_embed, d_embed, bias=False)

        self.w_out = nn.Linear(d_embed, d_embed, bias=False)

        self.register_buffer("causal_mask", None, persistent=False)  # Updated dynamically

    def get_causal_mask(self, seq_len):
        """
            A   B     C
        A   x   inf  inf

        B   x   x    inf

        C   x   x    x

        This is akin to the upper triangular matrix
        """
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            size = 1
            while size < seq_len:  # Exponential scaling to minimize frequent reallocations
                size = size * 2
            size = min(size, self.max_seq_len)

            mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)

            self.register_buffer("causal_mask", mask, persistent=False)  # caches the mask
        return self.causal_mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_embed = x.shape

        Q = self.query_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_size)
        K = self.key_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_size)
        V = self.value_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_size)

        K = K.transpose(-2, -1)  # (batch_size, num_heads, head_size, seq_len)

        scores = (Q @ K) / self.head_size ** 0.5  # (batch_size, num_heads, seq_len, seq_len)
        attention_mask = self.get_causal_mask(seq_len).view(1, 1, seq_len, seq_len).to(device)  # (1, 1, seq_len, seq_len)
        masked_scores = scores.masked_fill(attention_mask, float('-inf'))  # (batch_size, num_heads, seq_len, seq_len)

        attention = torch.softmax(masked_scores, dim=-1) @ V  # (batch_size, num_heads, seq_len, head_size)
        attention = attention.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_size)
        attention = attention.view(batch_size, seq_len, d_embed)  # (batch_size, seq_len, num_heads * head_size)

        out = self.w_out(attention)
        return out

class LayerNorm(nn.Module):
    def __init__(self, d_embed, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_embed))
        self.beta = nn.Parameter(torch.zeros(d_embed))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_mean = x.mean(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        layer_var = x.var(dim=-1, keepdim=True, unbiased=False)  # (batch_size, seq_len, 1)
        # unbiased=True:  var = sum((x - mean)^2) / (N-1)  # sample variance
        # unbiased=False: var = sum((x - mean)^2) / N  # population variance
        # But honestly the difference is so negligble at scale that it doesn't even matter
        
        norm = (x - layer_mean) / torch.sqrt(layer_var + self.eps)  # (batch_size, seq_len, 1)
        out = norm * self.gamma + self.beta  # (batch_size, seq_len, d_embed) Automatic broadcasting
        return out
        

class FFN(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        """ 2-layer MLP """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    
    def __init__(self, max_seq_len, d_embed, num_heads):
        super().__init__()
        
        self.d_embed = d_embed
        self.num_heads = num_heads
        
        self.mha = MultiHeadedAttention(d_embed, num_heads, max_seq_len)
        self.ln_1 = LayerNorm(d_embed)
        self.ln_2 = LayerNorm(d_embed)
        self.mlp = FFN(d_embed, d_embed * 4, d_embed)  # For final transformer block, replace d_embed with vocab_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.ln_1(x))  # (batch_size, seq_len, d_embed )
        x = x + self.mlp(self.ln_2(x))  # (batch_size, seq_len, d_embed )
        return x


class SimpleGPT(nn.Module):
    """
    GPT model architecture
    """
    
    def __init__(self, **kwargs):
        super(SimpleGPT, self).__init__()
        self.max_seq_len = kwargs.get('max_seq_len', 1024)
        self.vocab_size = kwargs.get('vocab_size', 50_257)
        self.n_transformer_blocks = kwargs.get('n_transformer_blocks', 12)
        self.num_attention_heads = kwargs.get('num_attention_heads', 8)
        self.d_embed = kwargs.get('d_embed', 768)
        
        self.network = nn.Sequential(*[TransformerBlock(self.max_seq_len, self.d_embed, self.num_attention_heads) for _ in range(self.n_transformer_blocks)])

        self.w_out = nn.Linear(self.d_embed, self.vocab_size)
        self.ln = LayerNorm(self.d_embed)
        
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_embed)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_embed)
        
    def forward(self, input_ids: torch.Tensor):
        batch_size, seq_len = input_ids.shape
        
        token_emb = self.token_embedding(input_ids)  # (batch_size, seq_len, d_embed)
        pos_ids = torch.arange(seq_len).to(device).unsqueeze(0)  # (1, seq_len)
        pos_emb = self.pos_embedding(pos_ids)  # (1, seq_len, d_embed)
        
        x = token_emb + pos_emb  # (batch_size, seq_len, d_embed)
        
        x = self.ln(self.network(x))  # (batch_size, seq_len, d_embed)
        x = self.w_out(x)  # (batch_size, seq_len, vocab_size)
        # x = torch.softmax(x, dim=-1)

        return x
    