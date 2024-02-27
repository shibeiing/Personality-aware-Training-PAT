import torch
import torch.nn as nn
import torch.nn.functional as F
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L, ReVGG2L
from espnet.nets.pytorch_backend.transducer.blocks import build_transformer_block
from espnet.nets.pytorch_backend.transformer.repeat import MultiSequential
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class ConvSanEncoder(nn.Module):
    def __init__(self, idim, odim, btn_dim, n_layer):
        super().__init__()
        self.odim = odim
        self.idim = idim
        self.pre_encoder = VGG2L(idim, odim, None)

        # config for self-attention network (SAN)
        san_config = {
            "d_hidden": odim,
            "d_ff": odim * 2,
            "heads": 4,
            "dropout-rate": 0.0,
            "pos-dropout-rate": 0.0,
            "att-dropout-rate": 0.0,
            "return_att_scores": False,
        }
        san_att_layer_config = {
            "d_hidden": odim,
            "d_ff": odim * 2,
            "heads": 1,
            "dropout-rate": 0.0,
            "pos-dropout-rate": 0.0,
            "att-dropout-rate": 0.0,
            "return_att_scores": False,
        }
        # get san_block function
        san_block_fn = build_transformer_block("encoder", san_config, "linear", "swish")
        san_att_layer_fn = build_transformer_block("encoder", san_att_layer_config, "linear", "swish")
        # build `n_layer` san blocks
        # 3 layer 4-head self-att and 1 layer 1-head self-att
        self.block_list = [san_block_fn() for _ in range(n_layer - 1)]
        self.san_blocks = MultiSequential(*self.block_list)
        self.last_san = san_att_layer_fn()
        self.bottleneck = torch.nn.Linear(odim, btn_dim)

        self.conv_subsampling_factor = 4
        self.time_subsampling_factor = 6

        self.att_scores = None

    def forward(self, xs, ilens):
        # xs in shape (Batch, Time, idim)
        # ilens in shape (Batch)
        # out in shape (Batch, Time/6, odim)

        masks = make_non_pad_mask(ilens.tolist(), xs[:, :, 0]).to(xs.device).unsqueeze(-2)
        xs, masks = self.pre_encoder(xs, masks)
        out, out_mask = self.san_blocks(xs, masks)
        out, out_mask = self.last_san(out, out_mask)
        out = self.bottleneck(out)

        return out, out_mask

    def get_att_scores(self):
        # gy20220901
        # get top-layer att_scores
        self.att_scores = self.block_list[-1].get_att_scores()
        return self.att_scores


class ConvSanDecoder(nn.Module):
    def __init__(self, enc_dim, odim, hdim, h_ling_dim, h_spk_dim, n_layer):
        super().__init__()
        self.zqx_conv = nn.Conv1d(
            in_channels=enc_dim,
            out_channels=hdim,
            kernel_size=1,
            stride=1,
        )
        self.ling_conv = nn.Conv1d(
            in_channels=h_ling_dim,
            out_channels=hdim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.spk_conv = nn.Conv1d(
            in_channels=h_spk_dim,
            out_channels=hdim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # config for self-attention network (SAN)
        san_config = {
            "d_hidden": hdim,
            "d_ff": hdim * 2,
            "heads": 4,
            "dropout-rate": 0.0,
            "pos-dropout-rate": 0.0,
            "att-dropout-rate": 0.0,
        }
        # get san_block function
        san_block_fn = build_transformer_block("encoder", san_config, "linear", "swish")
        # build `n_layer` san blocks
        self.san_blocks = MultiSequential(*[san_block_fn() for _ in range(n_layer)])

        self.re_vgg2l = ReVGG2L(hdim, odim, None)

    def forward(self, zqx, h_ling, h_spk, xs_mask):
        # zqx in shape (Batch, Time, enc_dim)
        # h_ling in shape (Batch, Time, h_ling_dim)
        # h_spk in shape (Batch, h_spk_dim)
        # xs_mask in shape (Batch, 1, Time)

        zqx = torch.transpose(zqx.contiguous(), 1, 2)
        zqx = self.zqx_conv.forward(zqx)  # B x hdim x T
        h_ling = self.ling_conv.forward(torch.transpose(h_ling, 1, 2))  # B x hdim x T
        h_spk = self.spk_conv.forward(h_spk.unsqueeze(2))  # B x hdim x 1
        xs = zqx + h_ling + h_spk
        xs, xs_mask = self.san_blocks(torch.transpose(xs, 1, 2), xs_mask)
        xs, xs_mask = self.re_vgg2l(xs, xs_mask)

        return xs, xs_mask
