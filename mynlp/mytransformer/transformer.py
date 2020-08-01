import torch
import torch.nn as nn
import numpy as np

"""
Position Embedding:
PE(pos, 2i)=sin(pos/10000^(2i/d_model))
PE(pos, 2i+1)=cos(pos/10000^(2i/d_model))
"""


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_length, model_dim):
        super(PositionalEncoding, self).__init__()

        # 给定词语的位置pos，把它编码成d_model维的向量 (max_seq_length, d_model)
        pe = np.array(
            [[(pos / np.power(10000, 2 * (i // 2) / model_dim)) for i in range(model_dim)] for pos in
             range(max_seq_length)])
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])
        pe = torch.FloatTensor(pe).unsqueeze(0)  # (1, max_seq_length, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :].clone().detach()


"""
Attention:
Attention(Q, K, V) = softmax(QK^T/sqrt(d_model))*V
"""


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, scale=None, mask=None):
        # q,k,v (batch size, max seq length, embedding size)
        # attention (batch size, max seq length, max seq length)
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if mask is not None:
            attention = attention.masked_fill_(mask == 1, -np.inf)
        attention = self.dropout(self.softmax(attention))
        # context (batch size, max seq length, embedding size)
        context = torch.bmm(attention, v)

        return context, attention


# Multi-head attention, Add & Norm
class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.attention = ScaledDotProductAttention(dropout)
        self.linear = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, q, k, v, mask=None):
        residual = q

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = q.size(0)

        # linear projection
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # split by num_heads
        q = q.view(batch_size * num_heads, -1, dim_per_head)
        k = k.view(batch_size * num_heads, -1, dim_per_head)
        v = v.view(batch_size * num_heads, -1, dim_per_head)

        if mask is not None:
            mask = mask.repeat(num_heads, 1, 1)

        scale = (q.size(-1) // num_heads) ** -0.5
        context, attention = self.attention(q, k, v, scale, mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        output = self.dropout(self.linear(context))
        output = self.layer_norm(residual + output)

        return output, attention


"""
Feed Forward:
FFN(x) = max(0, xW1+b1)W2+b2
"""


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(model_dim, ffn_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        residual = x
        x = self.relu(self.linear1(x))
        x = self.dropout(self.linear2(x))
        output = self.layer_norm(x + residual)

        return output


"""
Encoder:
    Input embedding, Position embedding
    Multi-head attention, Add & Norm
    Feed Forward, Add & Norm
"""


# EncoderLayer
class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, enc_inputs, enc_self_attn_mask=None):
        context, attention = self.attention(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        output = self.feed_forward(context)

        return output, attention


# EncoderLayer叠加成Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_length, num_layers=6, model_dim=512,
                 num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Encoder, self).__init__()

        self.word_embed = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.pos_embed = PositionalEncoding(max_seq_length, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.layer_stack = nn.ModuleList([EncoderLayer(model_dim, num_heads, ffn_dim, dropout)
                                          for _ in range(num_layers)])

    def forward(self, inputs, slf_attn_mask):
        embed = self.word_embed(inputs) + self.pos_embed(inputs)
        output = self.layer_norm(self.dropout(embed))

        attentions = []
        for encoder in self.layer_stack:
            output, attention = encoder(output, slf_attn_mask)
            attentions.append(attention)

        return output, attentions


"""
Decoder:
    Input embedding, Position embedding
    Multi-head attention, Add & Norm
    Multi-head attention, Add & Norm
    Feed Forward, Add & Norm
"""


# DecoderLayer
class DecoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.dec_enc_attn = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, dec_inputs, enc_outputs, slf_attn_mask=None, dec_enc_attn_mask=None):
        # self attention: all inputs are decoder inputs
        dec_outputs, self_attn = self.self_attn(dec_inputs, dec_inputs, dec_inputs, slf_attn_mask)

        # context attention: query is decoder's outputs, key and value are encoder's inputs
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)

        dec_outputs = self.feed_forward(dec_outputs)

        return dec_outputs, self_attn, dec_enc_attn


# DecoderLayer叠加成Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_seq_length, num_layers=6, model_dim=512,
                 num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Decoder, self).__init__()

        self.word_embed = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.pos_embed = PositionalEncoding(max_seq_length, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.layer_stack = nn.ModuleList([DecoderLayer(model_dim, num_heads, ffn_dim, dropout)
                                          for _ in range(num_layers)])

    def forward(self, tgt_seq, tgt_mask, enc_outputs, src_mask):
        embed = self.word_embed(tgt_seq) + self.pos_embed(tgt_seq)
        dec_outputs = self.layer_norm(self.dropout(embed))

        self_attns, dec_enc_attns = [], []
        for decoder in self.layer_stack:
            dec_outputs, self_attn, dec_enc_attn = decoder(dec_outputs, enc_outputs, slf_attn_mask=tgt_mask,
                                                           dec_enc_attn_mask=src_mask)
            self_attns.append(self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, self_attns, dec_enc_attns


"""
Transformer
"""


def get_pad_mask(seq):
    mask = (seq == 0).unsqueeze(-2)
    return mask


def get_sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    return mask


class Transformer(nn.Module):
    # model_dim: Embedding Size, ffn_dim: FeedForward dimension
    # n_layers: number of Encoder of Decoder Layer, n_heads: number of heads in Multi-Head Attention
    def __init__(self, src_vocab_size, src_max_len, tgt_vocab_size, tgt_max_len,
                 num_layers=6, num_heads=8, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, tgt_seq):
        src_mask = get_pad_mask(src_seq).int()
        tgt_mask = torch.gt((get_pad_mask(tgt_seq).int() + get_sequence_mask(tgt_seq).int()), 0).int()

        enc_output, enc_self_attn = self.encoder(src_seq, src_mask)
        dec_output, dec_self_attn, dec_enc_attn = self.decoder(tgt_seq, tgt_mask, enc_output, src_mask)
        output = self.softmax(self.linear(dec_output))

        return output, enc_self_attn, dec_self_attn, dec_enc_attn
