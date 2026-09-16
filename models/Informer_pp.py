import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.masking import TriangularCausalMask, ProbMask
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp
from layers.RevIN import RevIN


class Mlp_time(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.2, n_vars=1):
        super(Mlp_time, self).__init__()
        self.n_vars = n_vars
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc_layers = nn.ModuleList([nn.Linear(in_features, hidden_features) for _ in range(n_vars)])
        self.act_layers = nn.ModuleList([nn.ReLU() for _ in range(n_vars)])
        self.drop_layers = nn.ModuleList([nn.Dropout(drop) for _ in range(n_vars)])
    def forward(self, x):
        outputs = []
        for i in range(self.n_vars):
            tmp = x[:, i, :]
            z = self.fc_layers[i](x[:, i, :])
            z = self.act_layers[i](z)
            z = self.drop_layers[i](z)
            outputs.append(z+tmp) 
        return torch.stack(outputs, dim=1) 
    
class Mlp_feat(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(Mlp_feat, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        tmp = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x) + tmp
        return x


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=2 * i, dilation=2))
        self.kernels = nn.ModuleList(kernels)
        init_weight = False
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)

        return res


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class AttentionMechanism(nn.Module):
    def __init__(self, n=16):
        super(AttentionMechanism, self).__init__()
        self.attention_layer = nn.Linear(n*2, 1, bias=False)
    def forward(self, tensor_a, tensor_b):
        combined = torch.cat((tensor_a, tensor_b), dim=-1)
        attention_scores = self.attention_layer(combined)
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_tensor_a = tensor_a * attention_weights
        output = weighted_tensor_a + tensor_b
        return output

class series_decomp_multi(nn.Module):
    def __init__(self, kernel_size, seq_len=48):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.attention_mechanism_sea = AttentionMechanism(seq_len)
        self.attention_mechanism_mov = AttentionMechanism(seq_len)
    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)
            sea = x - moving_avg
            res.append(sea)
        sea = self.attention_mechanism_sea(res[0], res[1])
        moving_mean = self.attention_mechanism_mov(moving_mean[0], moving_mean[1])
        return sea, moving_mean




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.d_model = configs.d_model
        self.output_attention = configs.output_attention
        self.dct = {'48': 14, '96':26, '168':44}
        nnn = self.dct[str(self.seq_len)]
        self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        self.Linear_Sea = nn.Linear(nnn, nnn)
        self.Linear_Sea.weight = nn.Parameter((1 / nnn) * torch.ones([nnn, nnn]))
        self.MLP_feat = Mlp_feat(configs.enc_in, configs.enc_in)
        self.MLP_feat2 = Mlp_feat(configs.enc_in, configs.enc_in)
        self.MLP_feat3 = Mlp_feat(configs.enc_in, configs.enc_in)
        self.MLP_time = Mlp_time(self.seq_len, self.seq_len, n_vars=configs.enc_in)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.MLP_time2 = Mlp_time(self.seq_len, self.seq_len, n_vars=configs.enc_in)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.MLP_time3 = Mlp_time(self.seq_len, self.seq_len, n_vars=configs.enc_in)
        self.dropout3 = nn.Dropout(0.1)
        self.l1 = nn.Linear(self.seq_len, self.seq_len)
        self.l2 = nn.Linear(self.seq_len, self.seq_len)
        self.l3 = nn.Linear(self.seq_len, self.seq_len)
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for l in range(configs.e_layers - 1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        kernel_size = [5, 7]
        self.decomposition = series_decomp_multi(kernel_size, self.seq_len)
        self.input_projection_trend = nn.Linear(self.seq_len, self.seq_len) 
        self.input_projection_sea = nn.Linear(self.seq_len, self.seq_len) 
        n = self.pred_len + self.label_len
        self.projection_ = nn.Linear(self.seq_len, n)
        self.k = configs.top_k
        d_ff = 32
        d_model_ = 4
        num_kernels = 6 
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model_, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model_,
                               num_kernels=num_kernels)
        )

        self.predict_in = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.predict_out1 = nn.Linear(self.pred_len + self.seq_len, nnn)
        self.predict_out2 = nn.Linear(4, self.d_model)
        self.attention_mechanism = AttentionMechanism(self.d_model)
        self.dlinear_output = nn.Linear(self.pred_len, self.pred_len + self.label_len)
        self.fuse_trend = nn.Linear(self.seq_len, self.seq_len)
        self.revin_layer = RevIN(configs.enc_in, affine=False, subtract_last=False)

    def times_net(self, x_enc):
        x = self.predict_in(x_enc.permute(0, 2, 1)).permute(0, 2, 1)

        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)

                out = torch.cat([x, padding], dim=1)

            else:
                length = (self.seq_len + self.pred_len)
                out = x

            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        res = res + x
        res = self.predict_out1(res.permute(0, 2, 1)).permute(0, 2, 1)
        res = self.predict_out2(res)
        return res


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        decomposed_flag = True  
        timesnet_flag = True   
        dlinear_flag = True    
        shift_flag = True       
        revin_flag = False    
        time_mixer_flag = True
        if shift_flag:
            seq_last = x_enc[:, -1:, :].detach()
            x_enc = x_enc - seq_last
        if revin_flag:
            x_enc = self.revin_layer(x_enc, 'norm')
        if decomposed_flag:
            src_conditioner_series = x_enc.permute(0, 2, 1)
            seasonal, trend = self.decomposition(src_conditioner_series)
            seasonal = self.input_projection_sea(seasonal)
            trend = self.input_projection_trend(trend)           
            res = self.times_net(seasonal.permute(0, 2, 1))
            if time_mixer_flag:
                t1 = self.MLP_time(trend)
                t2 = self.MLP_feat(trend.permute(0,2,1)).permute(0,2,1)
               
                t = self.l1(t1+t2) +x_enc.permute(0,2,1)
                t3 = self.MLP_time2(t)
                t4 = self.MLP_feat2(t.permute(0,2,1)).permute(0,2,1)
                
                t = self.l2(t3+t4) + x_enc.permute(0,2,1)
                t5 = self.MLP_time3(t)
                t6 = self.MLP_feat3(t.permute(0,2,1)).permute(0,2,1)
                
                trend = self.l3(t5+t6)                 
            if dlinear_flag:
                dlinear_trend = self.Linear_Trend(trend)        
                res = self.Linear_Sea(res.permute(0,2,1)).permute(0,2,1)
        if not decomposed_flag:
            trend, seasonal = torch.zeros_like(x_enc), torch.zeros_like(x_dec)  
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        if timesnet_flag:
            assert enc_out.shape == res.shape
            enc_out = self.attention_mechanism(enc_out, res)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        if dlinear_flag:
            dlinear_output = self.dlinear_output(dlinear_trend).permute(0, 2, 1)
            dec_out = dlinear_output + dec_out
        if shift_flag:
            dec_out = dec_out + seq_last
        if revin_flag:       
            dec_out = self.revin_layer(dec_out, 'denorm')
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
