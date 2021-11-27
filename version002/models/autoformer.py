import torch
import torch.nn as nn
import torch.nn.functional as F
from version002.tools.Autoformer_Embed import DataEmbedding, DataEmbedding_wo_pos
from version002.tools.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from version002.tools.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class autoformer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self,x_enc, x_mark_enc, x_dec, x_mark_dec,opt):
        super(autoformer, self).__init__()
        self.seq_len = opt.seq_len
        self.label_len = opt.label_len
        self.pred_len = opt.pred_len
        self.output_attention = opt.output_attention

        # Decomp
        kernel_size = opt.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(opt.enc_in, opt.d_model, opt.embed, opt.freq,
                                                  opt.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(opt.dec_in, opt.d_model, opt.embed, opt.freq,
                                                  opt.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, opt.factor, attention_dropout=opt.dropout,
                                        output_attention=opt.output_attention),
                        opt.d_model, opt.n_heads),
                    opt.d_model,
                    opt.d_ff,
                    moving_avg=opt.moving_avg,
                    dropout=opt.dropout,
                    activation=opt.activation
                ) for l in range(opt.e_layers)
            ],
            norm_layer=my_Layernorm(opt.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, opt.factor, attention_dropout=opt.dropout,
                                        output_attention=False),
                        opt.d_model, opt.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, opt.factor, attention_dropout=opt.dropout,
                                        output_attention=False),
                        opt.d_model, opt.n_heads),
                    opt.d_model,
                    opt.c_out,
                    opt.d_ff,
                    moving_avg=opt.moving_avg,
                    dropout=opt.dropout,
                    activation=opt.activation,
                )
                for l in range(opt.d_layers)
            ],
            norm_layer=my_Layernorm(opt.d_model),
            projection=nn.Linear(opt.d_model, opt.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
