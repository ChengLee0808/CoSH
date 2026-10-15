# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default=75, help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--save_per_img', action='store_true', help='if specified, save per image')
        parser.add_argument('--show_corr', action='store_true', help='if specified, save bilinear upsample correspondence')
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=512, load_size=512, display_winsize=512)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
