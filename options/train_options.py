from .base_options import BaseOptions


class TrainOptions(BaseOptions):
	def initialize(self, parser):
		parser = BaseOptions.initialize(self, parser)
		# training epoch

		parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
		parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
		parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
		parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
		parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate')
		parser.add_argument('--niter_decay', type=int, default=200, help='# of iter to linearly decay learning rate to zero')
		parser.add_argument('--unlabel_decay', type=float, default=1.0, help='decay the unlabel loss weights every epoch')
		parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
		parser.add_argument('--patch_size', type=int, default=35, help='the size of image patch used in Dark Channel')
		parser.add_argument('--semi', action='store_true', help='semi-supervised training scheme')
		parser.add_argument('--all_loss', action='store_true', help='adding all loss(EC+TV) to supervised pair')
		parser.add_argument('--update_ratio', type=int, default=5, help='the update ratio of the supervised and unsupervised')

		# learning rate and loss weight
		parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
		parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
		parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy[lambda|step|plateau]')
		parser.add_argument('--lr_task', type=float, default=1e-4, help='initial learning rate for adam')
		parser.add_argument('--lr_trans', type=float, default=5e-5, help='initial learning rate for discriminator')
		parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')

		# display the results
		parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
		parser.add_argument('--show_freq', type=int, default=100, help='frequency of showing training results on plot')
		parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
		parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
		parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
		parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
		self.isTrain = True
		return parser