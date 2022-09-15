import os
import argparse
import datetime


class BaseOptions:
    def __init__(self):
        self.parser = None

    def initialize(self, parser):
        parser.add_argument("--project", type=str, default="Reverse-ISP2022")
        parser.add_argument("--trained_model", type=str, default="p20", choices=["p20", "s7"])
        parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
        parser.add_argument("--gpu", type=int, help="GPU id, None: all", default=None)
        parser.add_argument('--amp', action='store_true', default=False, help='Enables mixed precision')
        parser.add_argument("--model", type=str, help="Model to use for reverse ISP", default="base")
        parser.add_argument('--base_channels', type=int, default=36, help='Base channels')
        parser.add_argument("--dataset_dir", type=str, default="/mnt/D/data/aim-reverse-isp/p20/test", help='Dataset directory')
        parser.add_argument("--output_folder", type=str, default="./output", help='Dataset directory')
        parser.add_argument("--checkpoints", type=str, default="./checkpoints", help='Checkpoints directory')
        parser.add_argument('--workers', type=int, default=4,  help='number of workers to fetch data (default: 4)')

        
        return parser
    
    def init(self):
        if self.parser is None:
            parser = argparse.ArgumentParser(description='Reverse-ISP2022', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            return parser
        return self.parser

    def parse(self, args=None):
        self.parser = self.init()
            
        # get the basic options
        opt, unknown = self.parser.parse_known_args(args=args)
        opt = self.parser.parse_args(args=args)
        return opt
    
    def notebook(self, args=""):
        return self.parse(args)