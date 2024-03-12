import torch
import torch.nn as nn
import torch.nn.functional as F

from code.model.archs.local import Local_Base
from code.model.archs.arch_util import LayerNorm2d, SkipConnection, IdentityMod, get_activation, get_attention


class EBokehNet(nn.Module):
    def __init__(self, img_channel=3, width=16, drop_out_rate=0., skip_connections=None, dw_expand=1., ffn_expand=2,
                 attention_type="CA", activation_type="GELU",
                 enc_blk_nums=None, middle_blk_num=0, dec_blk_nums=None,
                 enc_blks_apply_strength=None, middle_blk_apply_strength=False, dec_blks_apply_strength=None,
                 enc_blks_apply_lens_factor=None, middle_blk_apply_lens_factor=False, dec_blks_apply_lens_factor=None,
                 enc_blks_use_pos_map=None, middle_blks_use_pos_map=False, dec_blks_use_pos_map=None,
                 in_stage_use_pos_map=False, out_stage_use_pos_map=False,
                 inverted_conv=True, kernel_size=3):
        super(EBokehNet, self).__init__()

        if enc_blk_nums is None:
            enc_blk_nums = [0]
        if dec_blk_nums is None:
            dec_blk_nums = [0]

        if enc_blks_apply_strength is None:
            enc_blks_apply_strength = [False for _ in range(len(enc_blk_nums))]
        if dec_blks_apply_strength is None:
            dec_blks_apply_strength = [False for _ in range(len(dec_blk_nums))]

        if enc_blks_apply_lens_factor is None:
            enc_blks_apply_lens_factor = [False for _ in range(len(enc_blk_nums))]
        if dec_blks_apply_lens_factor is None:
            dec_blks_apply_lens_factor = [False for _ in range(len(dec_blk_nums))]

        if enc_blks_use_pos_map is None:
            enc_blks_use_pos_map = [False for _ in range(len(enc_blk_nums))]
        if dec_blks_use_pos_map is None:
            dec_blks_use_pos_map = [False for _ in range(len(dec_blk_nums))]

        if skip_connections is None:
            skip_connections = [True for _ in range(len(dec_blk_nums))]

        # add 2 channels for position map if used
        self.in_stage = nn.Conv2d(in_channels=img_channel + 2 if in_stage_use_pos_map else img_channel,
                                  out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.out_stage = nn.Conv2d(in_channels=width + 2 if out_stage_use_pos_map else width,
                                   out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.skips = nn.ModuleList()

        self.middle_blks = nn.ModuleList()

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        block_config = {'drop_out_rate': drop_out_rate, 'attention_type': attention_type,
                        'activation_type': activation_type, 'inverted_conv': inverted_conv, 'kernel_size': kernel_size}

        depths = range(0, len(enc_blk_nums))

        chan = width
        for num, apply_strength, apply_lens_factor, use_pos_map, depth in \
                zip(enc_blk_nums, enc_blks_apply_strength, enc_blks_apply_lens_factor, enc_blks_use_pos_map, depths):
            self.encoders.append(
                nn.ModuleList(
                    [BaselineBlockMod(chan, dw_expand, ffn_expand, **block_config, use_pos_map=use_pos_map, depth=depth,
                                      apply_strength=True if x == num-1 and apply_strength else False,
                                      apply_lens_factor=True if x == num-1 and apply_lens_factor else False)
                     for x in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.ModuleList(
                [BaselineBlockMod(chan, dw_expand, ffn_expand, **block_config, use_pos_map=middle_blks_use_pos_map,
                                  depth=depths[-1]+1,
                                  apply_strength=True if x == middle_blk_num - 1 and middle_blk_apply_strength else False,
                                  apply_lens_factor=True if x == middle_blk_num - 1 and middle_blk_apply_lens_factor else False)
                 for x in range(middle_blk_num)]
            )

        for num, apply_strength, apply_lens_factor, use_pos_map, skip_connection, depth in \
                zip(dec_blk_nums, dec_blks_apply_strength, dec_blks_apply_lens_factor, dec_blks_use_pos_map,
                    skip_connections, depths.__reversed__()):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            self.skips.append(SkipConnection() if skip_connection else IdentityMod())
            chan = chan // 2
            self.decoders.append(
                nn.ModuleList(
                    [BaselineBlockMod(chan, dw_expand, ffn_expand, **block_config, use_pos_map=use_pos_map, depth=depth,
                                      apply_strength=True if x == num - 1 and apply_strength else False,
                                      apply_lens_factor=True if x == num - 1 and apply_lens_factor else False)
                     for x in range(num)]
                )
            )

        self.in_stage_use_pos_map = in_stage_use_pos_map
        self.out_stage_use_pos_map = out_stage_use_pos_map

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, x_in, bokeh_str=None, lens_factor=None, pos_map=None):

        x = torch.cat([x_in, pos_map], dim=1) if self.in_stage_use_pos_map else x_in
        x = self.in_stage(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            for blk in encoder:
                x = blk(x, bokeh_str=bokeh_str, pos_map=pos_map, lens_factor=lens_factor)
            encs.append(x)
            x = down(x)

        for blk in self.middle_blks:
            x = blk(x, bokeh_str=bokeh_str, pos_map=pos_map, lens_factor=lens_factor)

        for decoder, up, skip, enc_skip in zip(self.decoders, self.ups, self.skips, encs[::-1]):
            x = up(x)
            x = skip(x, enc_skip)
            for blk in decoder:
                x = blk(x, bokeh_str=bokeh_str, pos_map=pos_map, lens_factor=lens_factor)

        x = torch.cat([x, pos_map], dim=1) if self.in_stage_use_pos_map else x
        x = self.out_stage(x)

        return torch.clamp(x + x_in, 0., 1.)


class EBokehNetLocal(Local_Base, EBokehNet):
    def __init__(self, train_size=(1, 3, 512, 512), fast_impl=False, **kwargs):
        Local_Base().__init__()
        EBokehNet.__init__(self, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_impl=fast_impl)


class BaselineBlockMod(nn.Module):
    def __init__(self, c, DW_Expand=1., FFN_Expand=2, apply_strength=False, drop_out_rate=0., attention_type='CA',
                 activation_type='GELU', apply_lens_factor=False, use_pos_map=False,
                 inverted_conv=True, kernel_size=3, depth=None):
        super().__init__()
        dw_channel = int(c * DW_Expand) if int(c * DW_Expand) % 2 == 0 else int(c * DW_Expand) + 1

        # add posi
        if inverted_conv:
            self.conv1 = InvertedConvolution(in_channels=c + 2 if use_pos_map else c, out_channels=dw_channel,
                                             kernel_size=kernel_size, padding='same', bias=True)
        else:
            self.conv1 = nn.Conv2d(in_channels=c + 2 if use_pos_map else c, out_channels=dw_channel,
                                   kernel_size=kernel_size, padding='same', stride=1, bias=True)

        # Activation
        self.activation = get_activation(activation_type)

        # Channel Attention
        self.attention = get_attention(attention_type)(
            dw_channel // 2 if activation_type == 'Simple_Gate' else dw_channel, apply_lens_factor=apply_lens_factor)

        self.conv2 = nn.Conv2d(in_channels=dw_channel // 2 if activation_type == 'Simple_Gate' else dw_channel,
                               out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        # sub block 1 done
        ffn_channel = FFN_Expand * c
        self.conv3 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        # second activation call
        self.conv4 = nn.Conv2d(in_channels=ffn_channel // 2 if activation_type == 'Simple_Gate' else ffn_channel,
                               out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        # sub block 2 done

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.apply1 = ApplyChannelWeights() if apply_strength else IdentityMod()

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # save use_pos_map and depth metadata for forward
        self.use_pos_map = use_pos_map
        self.depth = depth

    def forward(self, x_in, bokeh_str, lens_factor=None, pos_map=None) -> torch.Tensor:
        pos_map = F.interpolate(pos_map, scale_factor=1 / 2 ** self.depth, mode='bilinear', align_corners=False) \
            if self.use_pos_map else None

        x = x_in

        x = self.norm1(x)

        x = torch.cat([x, pos_map], dim=1) if self.use_pos_map else x
        x = self.conv1(x)
        x = self.activation(x)

        att = self.attention(x, lens_factor)

        x = x * att
        x = self.conv2(x)

        x = self.dropout1(x)

        x = self.apply1(x, bokeh_str)

        y = x_in + x * self.beta

        x = self.conv3(self.norm2(y))
        x = self.activation(x)
        x = self.conv4(x)

        x = self.dropout2(x)

        ret = y + x * self.gamma

        return ret


class ApplyChannelWeights(nn.Module):
    def __init__(self):
        super(ApplyChannelWeights, self).__init__()

    def forward(self, x: torch.Tensor, weights) -> torch.Tensor:
        b, _, _, _ = x.shape
        if x.is_cuda:
            x_base = torch.zeros(x.shape).cuda()
        else:
            x_base = torch.zeros(x.shape)

        for i in range(b):
            x_base[i, :, :, :] = weights[i] * x[i, :, :, :]

        return x_base


class InvertedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding='same', bias=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=bias)
        # add positional argument
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                               padding=padding, stride=1, groups=out_channels, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
