from code.utils.dataloading_utils import get_loss
from code.model.archs.EBokehNet_arch import EBokehNet, EBokehNetLocal

import pytorch_lightning as pl


class DeBokehLn(pl.LightningModule):
    def __init__(self, img_channel=3, width=16, dw_expand=1, ffn_expand=2, drop_out_rate=0.,
                 attention_type="CA", activation_type="GELU", inverted_conv=True, kernel_size=3,
                 enc_blk_nums=None, middle_blk_num=8, dec_blk_nums=None, skip_connections=None,
                 in_stage_use_pos_map=False, out_stage_use_pos_map=False,
                 enc_blks_apply_strength=None, middle_blk_apply_strength=True, dec_blks_apply_strength=None,
                 enc_blks_apply_lens_factor=None, middle_blk_apply_lens_factor=False, dec_blks_apply_lens_factor=None,
                 enc_blks_use_pos_map=None, middle_blks_use_pos_map=False, dec_blks_use_pos_map=None,
                 loss='MSELoss', loss_args=None, loss_func="std",
                 optimizer='Adam', optimizer_args=None, lr=1e-4, lr_scheduler=None, lr_scheduler_args=None,
                 log_dest: str = None, log_train_metrics=False, log_val_metrics=True, log_test_metrics=True,
                 train_out_dir='../train_debug', val_out_dir='../val_out', test_out_dir='../test_out',
                 tlc=False, train_size=None,
                 mask_foreground=False, version="v1.3"):
        # change version number if incompatible changes are made!!!!

        super().__init__()

        self.img_channel = img_channel
        self.width = width

        self.dw_expand = dw_expand
        self.ffn_expand = ffn_expand

        self.drop_out_rate = drop_out_rate

        self.attention_type = attention_type
        self.activation_type = activation_type
        self.inverted_conv = inverted_conv
        self.kernel_size = kernel_size

        self.skip_connections = skip_connections or [True, True, True]

        self.in_stage_use_pos_map = in_stage_use_pos_map
        self.out_stage_use_pos_map = out_stage_use_pos_map

        self.enc_blks_nums = enc_blk_nums or [1, 1, 1]
        self.middle_blk_num = middle_blk_num
        self.dec_blks_nums = dec_blk_nums or [1, 2, 4]

        self.enc_blks_apply_strength = enc_blks_apply_strength or [False, False, False]
        self.middle_blk_apply_strength = middle_blk_apply_strength
        self.dec_blks_apply_strength = dec_blks_apply_strength or [False, False, False]

        self.enc_blks_apply_lens_factor = enc_blks_apply_lens_factor or [False, False, False]
        self.middle_blk_apply_lens_factor = middle_blk_apply_lens_factor
        self.dec_blks_apply_lens_factor = dec_blks_apply_lens_factor or [False, False, False]

        self.enc_blks_use_pos_map = enc_blks_use_pos_map or [False, False, False]
        self.middle_blks_use_pos_map = middle_blks_use_pos_map
        self.dec_blks_use_pos_map = dec_blks_use_pos_map or [False, False, False]

        self.loss_args = loss_args
        self.loss = get_loss(loss, self.loss_args)
        self.loss_func = loss_func

        self.optimizer = optimizer
        self.optimizer_args = optimizer_args or {}

        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_args = lr_scheduler_args or {}

        self.log_dest = log_dest
        self.log_train_metrics = log_train_metrics
        self.train_out_dir = train_out_dir
        self.log_val_metrics = log_val_metrics
        self.val_out_dir = val_out_dir
        self.log_test_metrics = log_test_metrics
        self.test_out_dir = test_out_dir

        self.tlc = tlc
        self.train_size = train_size

        self.mask_foreground = mask_foreground
        self.version = version

        network_args = {'img_channel': self.img_channel, 'width': self.width, 'skip_connections': self.skip_connections,
                        'dw_expand': self.dw_expand, 'ffn_expand': self.ffn_expand, 'drop_out_rate': self.drop_out_rate,
                        'attention_type': self.attention_type, 'activation_type': self.activation_type,
                        'inverted_conv': self.inverted_conv, 'kernel_size': self.kernel_size,
                        'in_stage_use_pos_map': self.in_stage_use_pos_map,
                        'out_stage_use_pos_map': self.out_stage_use_pos_map,
                        'enc_blk_nums': self.enc_blks_nums,
                        'enc_blks_apply_strength': self.enc_blks_apply_strength,
                        'enc_blks_apply_lens_factor': self.enc_blks_apply_lens_factor,
                        'enc_blks_use_pos_map': self.enc_blks_use_pos_map,
                        'middle_blk_num': self.middle_blk_num,
                        'middle_blk_apply_strength': self.middle_blk_apply_strength,
                        'middle_blk_apply_lens_factor': self.middle_blk_apply_lens_factor,
                        'middle_blks_use_pos_map': self.middle_blks_use_pos_map,
                        'dec_blk_nums': self.dec_blks_nums,
                        'dec_blks_apply_strength': self.dec_blks_apply_strength,
                        'dec_blks_apply_lens_factor': self.dec_blks_apply_lens_factor,
                        'dec_blks_use_pos_map': self.dec_blks_use_pos_map}

        self.model = \
            EBokehNet(**network_args) if not self.tlc else EBokehNetLocal(train_size=self.train_size, fast_impl=False,
                                                                          **network_args)

        self.w_alex = None
