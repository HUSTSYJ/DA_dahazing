import torch
import itertools
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool
import torch.nn.functional as F
import util.task as task
from ECLoss.ECLoss import BCLoss, DCLoss
from TVLoss.TVLossL1 import TVLossL1
from TVLoss.L1_TVLoss import L1_TVLoss_Charbonnier
from . import losses

class RDehazingnetModel(BaseModel):
	def name(self):
		return 'RDehazingnetModel'

	@staticmethod
	def modify_commandline_options(parser, is_train=True):

		parser.set_defaults(no_dropout=True)
		if is_train:
			parser.add_argument('--lambda_Dehazing', type=float, default=10.0, help='weight for reconstruction loss (dehazing)')

			parser.add_argument('--lambda_Dehazing_DC', type=float, default=1e-4, help='weight for dark channel loss')
			parser.add_argument('--lambda_Dehazing_TV', type=float, default=1e-4, help='weight for TV loss')
			parser.add_argument('--lambda_gan', type=float, default=0, help='weight for gan loss')
			parser.add_argument('--which_model_netG_A', type=str, default='resnet_9blocks_depth',
								help='selects model to use for netG_A')
			parser.add_argument('--which_model_netG_B', type=str, default='resnet_9blocks',
								help='selects model to use for netG_B')
			parser.add_argument('--g_s2r_premodel', type=str, default=" ", help='pretrained G_s2r model')

			parser.add_argument('--R_Dehazing_premodel', type=str, default=" ", help='pretrained Dehazing model')
		else:
			parser.add_argument('--which_model_netG_A', type=str, default='resnet_9blocks_depth',
							   help='selects model to use for netG_A')
			parser.add_argument('--which_model_netG_B', type=str, default='resnet_9blocks',
								help='selects model to use for netG_B')
		return parser

	def initialize(self, opt):
		BaseModel.initialize(self, opt)

		if self.isTrain:
			self.loss_names = ['R_Dehazing_TV', 'R_Dehazing_DC', 'S2R_Dehazing', 'G']

		if self.isTrain:
			self.visual_names = ['syn_haze_img', 'img_s2r', 'clear_img', 's2r_dehazing_img','real_haze_img', 'r_dehazing_img']

		else:
			# self.visual_names = ['syn_haze_img', 'img_s2r', 'clear_img', 's2r_dehazing_img', 'real_haze_img', 'r_dehazing_img']
			# self.visual_names = ['syn_haze_img', 'img_s2r']
			self.visual_names = ['r_dehazing_img']

		if self.isTrain:
			self.model_names = ['R_Dehazing']

		else:
			self.model_names = ['R_Dehazing']

		use_parallel = False
		self.netS2R = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
										opt.which_model_netG_A, opt.norm, not opt.no_dropout, self.gpu_ids, use_parallel,
										opt.learn_residual)

		self.netR_Dehazing = networks.define_Gen(opt.input_nc, opt.output_nc, opt.ngf, opt.task_layers, opt.norm,
												 opt.activation, opt.task_model_type, opt.init_type, opt.drop_rate,
												 False, opt.gpu_ids, opt.U_weight)
		if self.isTrain:
			use_sigmoid = False
			self.netD = networks.define_D(opt.output_nc, opt.ndf,
											opt.which_model_netD,
											opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)
		if self.isTrain:
			self.init_with_pretrained_model('S2R', self.opt.g_s2r_premodel)
			self.init_with_pretrained_model('R_Dehazing', self.opt.R_Dehazing_premodel)
			self.netS2R.eval()
		else:
			self.init_with_pretrained_model('S2R', self.opt.g_s2r_premodel)
			self.init_with_pretrained_model('R_Dehazing', self.opt.R_Dehazing_premodel)

		if self.isTrain:
			self.fake_pool = ImagePool(opt.pool_size)
			# define loss functions
			self.criterionDehazing = torch.nn.MSELoss()
			self.TVLoss = L1_TVLoss_Charbonnier()
			self.criterionGAN = losses.GANLoss(use_ls=True).to(self.device)
			self.optimizer_G_task = torch.optim.Adam(itertools.chain(self.netR_Dehazing.parameters()),
													 lr=opt.lr_task, betas=(0.9, 0.999))
			self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()),
												lr=opt.lr, betas=(opt.beta1, 0.999))
			self.optimizers = []
			self.optimizers.append(self.optimizer_G_task)
			self.optimizers.append(self.optimizer_D)

	def set_input(self, input):

		if self.isTrain:
			AtoB = self.opt.which_direction == 'AtoB'
			input_A = input['A' if AtoB else 'B']
			input_B = input['B' if AtoB else 'A']
			input_C = input['C']
			self.depth = input['D'].to(self.device)
			self.syn_haze_img = input_A.to(self.device)
			self.real_haze_img = input_C.to(self.device)
			self.clear_img = input_B.to(self.device)
			self.image_paths = input['A_paths' if AtoB else 'B_paths']

			self.num = self.syn_haze_img.shape[0]
		else:
			self.real_haze_img = input['C'].to(self.device)
			# self.syn_haze_img = input['A'].to(self.device)
			# self.depth = input['D'].to(self.device)
			# self.clear_img = input['B'].to(self.device)
			self.image_paths = input['C_paths']

	def forward(self):

		if self.isTrain:

			self.img_s2r = self.netS2R(self.syn_haze_img, self.depth, True).detach()
			self.out = self.netR_Dehazing(torch.cat((self.img_s2r, self.real_haze_img), 0))
			self.s2r_dehazing_img = self.out[-1].narrow(0, 0, self.num)
			self.r_dehazing_img = self.out[-1].narrow(0, self.num, self.num)

		else:
			# self.img_s2r = self.netS2R(self.syn_haze_img, self.depth, True)
			# self.s2r_dehazing_img = self.netR_Dehazing(self.img_s2r)[-1]
			self.r_dehazing_img = self.netR_Dehazing(self.real_haze_img)[-1]

	def backward_D_basic(self, netD, real, fake):
		# Real
		pred_real = netD(real)
		loss_D_real = self.criterionGAN(pred_real, True)
		# Fake
		pred_fake = netD(fake.detach())
		loss_D_fake = self.criterionGAN(pred_fake, False)
		# Combined loss
		loss_D = (loss_D_real + loss_D_fake) * 0.5
		# backward
		loss_D.backward()
		return loss_D

	def backward_D(self):
		r_dehazing_img = self.fake_pool.query(self.r_dehazing_img)
		self.loss_D = self.backward_D_basic(self.netD, self.clear_img, r_dehazing_img)

	def backward_G(self):

		lambda_Dehazing = self.opt.lambda_Dehazing

		size = len(self.out)
		clear_imgs = task.scale_pyramid(self.clear_img, size - 1)
		self.loss_S2R_Dehazing = 0.0
		for (dehazing_img, clear_img) in zip(self.out[1:], clear_imgs):
			self.loss_S2R_Dehazing += self.criterionDehazing(dehazing_img[:self.num, :, :, :], clear_img) * lambda_Dehazing

		# if iter % 2000 == 1 and iter > 2000:
		# 	self.opt.lambda_Dehazing_DC *= self.opt.unlabel_decay
		# 	self.opt.lambda_Dehazing_TV *= self.opt.unlabel_decay
		# 	print('unlabel loss decay {}, is {}'.format(self.opt.unlabel_decay, self.opt.lambda_DC))

		# self.loss_R_Dehazing_TV = torch.Tensor([0.0])
		# self.loss_R_Dehazing_DC = torch.Tensor([0.0])

		# TV LOSS
		# self.loss_R_Dehazing_TV = TVLossL1(self.r_dehazing_img) * self.opt.lambda_Dehazing_TV
		self.loss_R_Dehazing_TV = self.TVLoss(self.r_dehazing_img) * self.opt.lambda_Dehazing_TV

		# DC LOSS
		self.loss_R_Dehazing_DC = DCLoss((self.r_dehazing_img + 1) / 2, self.opt.patch_size) * self.opt.lambda_Dehazing_DC
		# GAN LOSS
		self.loss_G = self.criterionGAN(self.netD(self.r_dehazing_img), True) * self.opt.lambda_gan
		self.loss_GR_Dehazing = self.loss_S2R_Dehazing + self.loss_R_Dehazing_TV + self.loss_R_Dehazing_DC + self.loss_G


		self.loss_GR_Dehazing.backward()

	def optimize_parameters(self):

		#without GAN loss
		self.forward()
		self.optimizer_G_task.zero_grad()
		self.backward_G()
		self.optimizer_G_task.step()

        #with GAN LOSS
		# # G
		# self.forward()
		# self.set_requires_grad(self.netD, False)
		# self.optimizer_G_task.zero_grad()
		# self.backward_G()
		# self.optimizer_G_task.step()
		# # D
		# self.set_requires_grad(self.netD, True)
		# self.optimizer_D.zero_grad()
		# self.backward_D()
		# self.optimizer_D.step()
