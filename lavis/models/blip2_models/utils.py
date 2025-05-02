import torch
import torch.nn as nn


class Prompt_Enhence(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size, dropout=0.1):
        super(Prompt_Enhence, self).__init__()

        self.num_heads = num_heads
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.query_fc = nn.Linear(embed_size, hidden_size)
        self.key_fc = nn.Linear(embed_size, hidden_size)
        self.value_fc = nn.Linear(embed_size, hidden_size)
        self.out_fc = nn.Linear(hidden_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, query, key, value, attention_mask=None):
        B, T_q, _ = query.size()
        _, T_k, _ = key.size()
        Q = self.query_fc(query)  # [B, T_q, embed_size]
        K = self.key_fc(key)  # [B, T_k, embed_size]
        V = self.value_fc(value)  # [B, T_k, embed_size]
        assert self.hidden_size % self.num_heads == 0
        head_dim = self.hidden_size // self.num_heads
        Q = Q.view(B, T_q, self.num_heads, head_dim).transpose(1, 2)  # [B,h,T_q,D]
        K = K.view(B, T_k, self.num_heads, head_dim).transpose(1, 2)  # [B,h,T_k,D]
        V = V.view(B, T_k, self.num_heads, head_dim).transpose(1, 2)  # [B,h,T_k,D]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # [B,h,T_q,D] * [B,h,D,T_k] -> Q[B, h, T_q, T_k]
        attention_scores = attention_scores / head_dim ** 0.5
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # [B, h, T_q, T_k]
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)  # [B, h, T_q, D]
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, T_q, self.hidden_size)
        attention_output = self.out_fc(attention_output)  # [B, T_q, embed_size]

        return attention_output