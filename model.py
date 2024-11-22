
import torch
import torch.nn as nn
import math
device = 'cuda'

class MultiHeadAttention(nn.Module):
  def __init__(self, heads=4,d_model=256):
    super(MultiHeadAttention, self).__init__()
    self.heads = heads
    self.d_model = d_model
    self.W_q = nn.Linear(d_model,d_model)
    self.W_k = nn.Linear(d_model,d_model)
    self.W_v = nn.Linear(d_model,d_model)

    self.W_o = nn.Linear(d_model,d_model)

  def forward(self, q,k,v,mask):
    d_k = self.d_model // self.heads
    q,k,v = self.W_q(q),self.W_k(k),self.W_v(v)

    q = q.view(q.shape[0],q.shape[1],self.heads,-1).transpose(1,2) # (batch,h,seq_len,d_k)
    k = k.view(k.shape[0],k.shape[1],self.heads,-1).transpose(1,2) # (batch,h,seq_len,d_k)
    v = v.view(v.shape[0],v.shape[1],self.heads,-1).transpose(1,2) # (batch,h,seq_len,d_k)

    res = torch.matmul(q,k.transpose(-2,-1))  /  math.sqrt(d_k)

    if mask is not None:
      mask = mask.unsqueeze(1)
      res = torch.masked_fill(res,mask==0,float('-inf'))

    attention = nn.functional.softmax(res,dim=-1)

    # (batch,h,seq_len,d_k) -> (batch,seq_len,h,d_k) -> (batch,h,seq_len,d_model)
    output = torch.matmul(attention,v).transpose(1,2).contiguous().view(q.shape[0],-1,self.d_model)
    output = self.W_o(output)


    return output


class EncoderBlock(nn.Module):
  def __init__(self,d_model=256,heads=4,hidden=1024,dropout=0.1):
    super(EncoderBlock, self).__init__()


    self.multi_head_attention = MultiHeadAttention(heads=heads,d_model=d_model)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.mlp = nn.Sequential(nn.Linear(d_model,hidden),
                             nn.GELU(),
                             nn.Linear(hidden,d_model,
                             nn.Dropout(dropout)
                             ))

  def forward(self,src,mask):
    att_output = self.multi_head_attention(src,src,src,mask)
    skip_con = att_output + src
    norm_output = self.norm1(skip_con)
    mlp_out = self.mlp(norm_output)
    return self.norm2(mlp_out + norm_output)


class Encoder(nn.Module):
  def __init__(self,num_layers=6,d_model=256,heads=4,hidden=1024):
      super(Encoder, self).__init__()
      self.layers = nn.ModuleList([EncoderBlock(d_model=d_model,heads=heads,hidden=hidden) for i in range(num_layers)])

  def forward(self,src,mask):
    for layer in self.layers:
      src = layer(src,mask)
    return src


class DecoderBlock(nn.Module):
  def __init__(self,d_model=256,heads=4,hidden=1024,dropout=0.1):
    super(DecoderBlock, self).__init__()


    self.masked_multi_head_attention = MultiHeadAttention(heads=heads,d_model=d_model)
    self.cross_multi_head_attention = MultiHeadAttention(heads=heads,d_model=d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)



    self.mlp = nn.Sequential(nn.Linear(d_model,hidden),
                             nn.GELU(),
                             nn.Linear(hidden,d_model),
                             nn.Dropout(dropout))

  def forward(self,tgt,memory,tgt_padding_mask,tgt_causal_mask,memory_padding_mask):
    tgt_padding_mask = (tgt_causal_mask & tgt_padding_mask).int()

    att_output = self.masked_multi_head_attention(tgt,tgt,tgt,tgt_padding_mask)
    norm_output = self.norm1(att_output + tgt )
    cross_attn_output = self.cross_multi_head_attention(norm_output,memory,memory,memory_padding_mask)
    norm_output = self.norm2(cross_attn_output + norm_output)
    mlp_out = self.mlp(norm_output)
    return self.norm3(mlp_out + norm_output)

class Decoder(nn.Module):
  def __init__(self,num_layers=6,d_model=256,heads=4,hidden=1024):
      super(Decoder, self).__init__()
      self.layers = nn.ModuleList([DecoderBlock(d_model=d_model,heads=heads,hidden=hidden) for i in range(num_layers)])

  def forward(self,tgt,memory,tgt_padding_mask,tgt_causal_mask,memory_padding_mask):
    for layer in self.layers:
      tgt = layer(tgt,memory,tgt_padding_mask,tgt_causal_mask,memory_padding_mask)
    return tgt

class TranslationModel(nn.Module):
  def __init__(self,vocab_size,d_model,src_seq_len,trg_seq_len):
    super(TranslationModel, self).__init__()
    self.src_seq_len = src_seq_len
    self.trg_seq_len = trg_seq_len

    self.encoder = Encoder(d_model=d_model,)
    self.decoder = Decoder(d_model=d_model,)

    self.embedding_src = nn.Embedding(vocab_size,d_model)
    self.embedding_tgt = nn.Embedding(vocab_size,d_model)

    self.positional_encoding_src = nn.Embedding(src_seq_len,d_model)
    self.positional_encoding_tgt = nn.Embedding(trg_seq_len,d_model)

    self.src_pos = torch.arange(0,src_seq_len).unsqueeze(0).to(device)
    self.trg_pos = torch.arange(0,trg_seq_len).unsqueeze(0).to(device)

    self.ffn = nn.Linear(d_model,vocab_size)

  def forward(self,src,tgt,src_padding_mask,tgt_padding_mask):

    src_padding_mask = src_padding_mask.unsqueeze(1).repeat(1, self.src_seq_len, 1)
    tgt_padding_mask = tgt_padding_mask.unsqueeze(1).repeat(1, self.trg_seq_len, 1)
    tgt_causal_mask = torch.tril(torch.ones_like(tgt_padding_mask)).to(device)



    src_embed = self.embedding_src(src) + self.positional_encoding_src(self.src_pos)
    tgt_embed = self.embedding_tgt(tgt) + self.positional_encoding_tgt(self.trg_pos)

    enc_out = self.encoder(src_embed,src_padding_mask)
    dec_out = self.decoder(tgt = tgt_embed,
                           memory=enc_out,
                           tgt_padding_mask = tgt_padding_mask,
                           tgt_causal_mask = tgt_causal_mask,
                           memory_padding_mask = src_padding_mask)

    return self.ffn(dec_out)
