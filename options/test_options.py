from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--test_type', type=str, default='real', help='syn, real')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=5000, help='how many test images to run')
        parser.add_argument('--R_Dehazing_premodel', type=str, default=" ", help='pretrained Dehazing model')
        parser.add_argument('--S_Dehazing_premodel', type=str, default=" ", help='pretrained Dehazing model')
        parser.add_argument('--g_s2r_premodel', type=str, default=" ", help='pretrained G_s2r model')
        parser.add_argument('--g_r2s_premodel', type=str, default=" ", help='pretrained G_r2s model')
        self.isTrain = False

        return parser
