import numpy as n
import torch
import os
import itertools
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
import util.task as task
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .losses import init_loss
import pdb
from ECLoss.ECLoss import BCLoss, DCLoss
from TVLoss.TVLossL1 import TVLossL1
from TVLoss.L1_TVLoss import L1_TVLoss_Charbonnier
from . import losses

try:
	xrange  # Python2
except NameError:
	xrange = range  # Python 3


class DAnetmodel(BaseModel):
	def name(self):
		return 'DAnetModel'

	@staticmethod
	def modify_commandline_options(parser, is_train=True):

		parser.set_defaults(no_dropout=True)
		if is_train:
			parser.add_argument('--lambda_Dehazing', type=float, default=10.0, help='weight for reconstruction loss (dehazing)')
			parser.add_argument('--lambda_Dehazing_Con', type=float, default=50.0, help='weight for consistency')

			parser.add_argument('--lambda_Dehazing_DC', type=float, default=0.01, help='weight for dark channel loss')
			parser.add_argument('--lambda_Dehazing_TV', type=float, default=0.01, help='weight for TV loss')

			parser.add_argument('--lambda_gan_feat', type=float, default=0.1, help='weight for feature GAN loss')

			# cyclegan
			parser.add_argument('--lambda_S', type=float, default=1.0, help='weight for cycle loss (A -> B -> A)')
			parser.add_argument('--lambda_R', type=float, default=1.0, help='weight for cycle loss (B -> A -> B)')
			parser.add_argument('--lambda_identity', type=float, default=30.0,
								help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

			parser.add_argument('--which_model_netG_A', type=str, default='resnet_9blocks_depth',
								help='selects model to use for netG_A')
			parser.add_argument('--which_model_netG_B', type=str, default='resnet_9blocks',
								help='selects model to use for netG_B')
			parser.add_argument('--S_Dehazing_premodel', type=str, default=" ",
								help='pretrained dehazing model')
			parser.add_argument('--R_Dehazing_premodel', type=str, default=" ",
								help='pretrained dehazing model')

			parser.add_argument('--g_s2r_premodel', type=str, default=" ",
								help='pretrained G_s2r model')
			parser.add_argument('--g_r2s_premodel', type=str, default=" ",
								help='pretrained G_r2s model')
			parser.add_argument('--d_s_premodel', type=str, default=" ",
								help='pretrained D_s model')
			parser.add_argument('--d_r_premodel', type=str, default=" ",
								help='pretrained D_r model')

			parser.add_argument('--freeze_bn', action='store_true', help='freeze the bn in mde')
			parser.add_argument('--freeze_in', action='store_true', help='freeze the in in cyclegan')
		return parser
	def initialize(self, opt):
		BaseModel.initialize(self, opt)


		# specify the training losses you want to print out. The program will call base_model.get_current_losses
		if self.isTrain:
			self.loss_names = ['S2R_Dehazing', 'S_Dehazing', 'R2S_Dehazing_DC', 'R_Dehazing_DC']
			self.loss_names +=['R2S_Dehazing_TV', 'R_Dehazing_TV', 'Dehazing_Con']
			self.loss_names +=['idt_R', 'idt_S', 'D_R', 'D_S', 'G_S2R', 'G_R2S', 'cycle_S', 'cycle_R', 'G_Rfeat', 'G_Sfeat', 'D_Rfeat', 'D_Sfeat']

		# specify the images you want to save/display. The program will call base_model.get_current_visuals
		if self.isTrain:
			visual_names_S = ['syn_haze_img', 'img_s2r', 'clear_img', 's2r_dehazing_img', 's_dehazing_img'] #, 's_rec_img']
			visual_names_R = ['real_haze_img', 'img_r2s', 'r2s_dehazing_img', 'r_dehazing_img'] #, 'r_rec_img']
			# if self.opt.lambda_identity > 0.0:
			# 	visual_names_S.append('idt_S')
			# 	visual_names_R.append('idt_R')
			self.visual_names = visual_names_S + visual_names_R
		else:
			self.visual_names = ['pred', 'img', 'img_trans']

		# specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
		if self.isTrain:
			self.model_names = ['S_Dehazing', 'R_Dehazing']
			self.model_names += ['S2R', 'R2S', 'D_R', 'D_S', 'D_Sfeat', 'D_Rfeat']
		else:
			self.model_names = ['S_Dehazing', 'R_Dehzaing', 'S2R', 'R2S']

		# Temp Fix for nn.parallel as nn.parallel crashes oc calculating gradient penalty
		# use_parallel = not opt.gan_type == 'wgan-gp'
		use_parallel = False
		# define the transform network
		self.netS2R = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
										 opt.which_model_netG_A, opt.norm, not opt.no_dropout, self.gpu_ids, use_parallel,
										 opt.learn_residual)
		self.netR2S = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
										 opt.which_model_netG_A, opt.norm, not opt.no_dropout, self.gpu_ids, use_parallel,
										 opt.learn_residual)


		# define the image dehazing network
		self.netR_Dehazing = networks.define_Gen(opt.input_nc, opt.output_nc, opt.ngf, opt.task_layers, opt.norm,
												   opt.activation, opt.task_model_type, opt.init_type, opt.drop_rate,
												   False, opt.gpu_ids, opt.U_weight)

		self.netS_Dehazing = networks.define_Gen(opt.input_nc, opt.output_nc, opt.ngf, opt.task_layers, opt.norm,
													 opt.activation, opt.task_model_type, opt.init_type, opt.drop_rate,
													 False, opt.gpu_ids, opt.U_weight)
		# define the discriminator
		if self.isTrain:
			use_sigmoid = False

			self.netD_R = networks.define_D(opt.input_nc, opt.ndf,
											   opt.which_model_netD,
											   opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)

			self.netD_S = networks.define_D(opt.input_nc, opt.ndf,
										opt.which_model_netD,
										opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids, use_parallel)


			self.netD_Sfeat = networks.define_featureD(opt.image_feature, n_layers=2, norm='batch', activation='PReLU',
													   init_type='xavier', gpu_ids=self.gpu_ids)

			self.netD_Rfeat = networks.define_featureD(opt.image_feature, n_layers=2, norm='batch', activation='PReLU',
													   init_type='xavier', gpu_ids=self.gpu_ids)
		if self.isTrain and not opt.continue_train:

			self.init_with_pretrained_model('S2R', self.opt.g_s2r_premodel)
			self.init_with_pretrained_model('R2S', self.opt.g_r2s_premodel)
			self.init_with_pretrained_model('R_Dehazing', self.opt.R_Dehazing_premodel)
			self.init_with_pretrained_model('S_Dehazing', self.opt.S_Dehazing_premodel)
			self.init_with_pretrained_model('D_R', self.opt.d_r_premodel)
			self.init_with_pretrained_model('D_S', self.opt.d_s_premodel)

		if opt.continue_train:
			self.load_networks(opt.which_epoch)

		if self.isTrain:
			self.fake_s_pool = ImagePool(opt.pool_size)
			self.fake_r_pool = ImagePool(opt.pool_size)

			# define loss functions
			self.criterionGAN = losses.GANLoss(use_ls=not opt.no_lsgan).to(self.device)
			self.l1loss = torch.nn.L1Loss()
			self.criterionCycle = torch.nn.L1Loss()
			self.criterionIdt = torch.nn.L1Loss()
			self.criterionDehazing = torch.nn.MSELoss()
			self.criterionCons = torch.nn.L1Loss()
			self.nonlinearity = torch.nn.ReLU()
			self.TVLoss = L1_TVLoss_Charbonnier()
			# initialize optimizers
			self.optimizer_G_task = torch.optim.Adam(itertools.chain(self.netS_Dehazing.parameters(),
																	 self.netR_Dehazing.parameters()),
													 lr=opt.lr_task, betas=(0.95, 0.999))
			self.optimizer_G_trans = torch.optim.Adam(itertools.chain(self.netS2R.parameters(),
																	  self.netR2S.parameters()),
													  lr=opt.lr_trans, betas=(0.5, 0.9))
			self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_S.parameters(),
																self.netD_R.parameters(),
																self.netD_Sfeat.parameters(),
																self.netD_Rfeat.parameters()),
												lr=opt.lr_trans, betas=(0.5, 0.9))
			self.optimizers = []
			self.optimizers.append(self.optimizer_G_task)
			self.optimizers.append(self.optimizer_G_trans)
			self.optimizers.append(self.optimizer_D)
			if opt.freeze_bn:
				self.netS_Dehazing.apply(networks.freeze_bn)
				self.netR_Dehazing.apply(networks.freeze_bn)
			if opt.freeze_in:
				self.netS2R.apply(networks.freeze_in)
				self.netR2S.apply(networks.freeze_in)



	def set_input(self, input):

		if self.isTrain:
			AtoB = self.opt.which_direction == 'AtoB'
			input_A = input['A' if AtoB else 'B']
			input_B = input['B' if AtoB else 'A']
			input_C = input['C']
			self.syn_haze_img = input_A.to(self.device)
			self.real_haze_img = input_C.to(self.device)
			self.clear_img = input_B.to(self.device)
			self.depth = input['D'].to(self.device)
			self.real_depth = input['E'].to(self.device)
			self.image_paths = input['A_paths' if AtoB else 'B_paths']
		else:
			self.img = input['A'].to(self.device)

	def forward(self):

		if self.isTrain:
			pass

		# else:
		# 	if self.opt.phase == 'test':
		# 		self.pred_s = self.netS_Dehazing(self.img)[-1]
		# 		self.img_trans = self.netS2R(self.img)
		# 		self.pred_r = self.netR_Dehazing(self.img_trans)[-1]
		# 		self.pred = 0.5 * (self.pred_s + self.pred_r)
		# 	else:
		# 		self.pred_r = self.netR_Dehazing(self.img)[-1]
		# 		self.img_trans = self.netR2S(self.img)
		# 		self.pred_s = self.netS_Dehazing(self.img_trans)[-1]
		# 		self.pred = 0.5 * (self.pred_s + self.pred_r)


	# get image paths
	def get_image_paths(self):
		return self.image_paths

	def backward_D_basic(self, netD, real, fake):
		# Real
		pred_real = netD(real.detach())
		loss_D_real = self.criterionGAN(pred_real, True)
		# Fake
		pred_fake = netD(fake.detach())
		loss_D_fake = self.criterionGAN(pred_fake, False)
		# Combined loss
		loss_D = (loss_D_real + loss_D_fake) * 0.5
		# backward
		loss_D.backward()
		return loss_D

	def backward_D_S(self):
		img_r2s = self.fake_s_pool.query(self.img_r2s)
		self.loss_D_S = self.backward_D_basic(self.netD_S, self.syn_haze_img, img_r2s)

	def backward_D_R(self):
		img_s2r = self.fake_r_pool.query(self.img_s2r)
		self.loss_D_R = self.backward_D_basic(self.netD_R, self.real_haze_img, img_s2r)

	def backward_D_Sfeat(self):

		self.loss_D_Sfeat = self.backward_D_basic(self.netD_Sfeat, self.s_dehazing_feat, self.r2s_dehazing_feat)

	def backward_D_Rfeat(self):

		self.loss_D_Rfeat = self.backward_D_basic(self.netD_Rfeat, self.r_dehazing_feat, self.s2r_dehazing_feat)


	def backward_G(self):



		lambda_Dehazing = self.opt.lambda_Dehazing
		lambda_Dehazing_Con = self.opt.lambda_Dehazing_Con
		lambda_gan_feat = self.opt.lambda_gan_feat
		lambda_idt = self.opt.lambda_identity
		lambda_S = self.opt.lambda_S
		lambda_R = self.opt.lambda_R

		# =========================== synthetic ==========================
		self.img_s2r = self.netS2R(self.syn_haze_img, self.depth, True)
		self.idt_S = self.netR2S(self.syn_haze_img, self.depth, True)
		self.s_rec_img = self.netR2S(self.img_s2r, self.depth, True)
		self.out_r = self.netR_Dehazing(self.img_s2r)
		self.out_s = self.netS_Dehazing(self.syn_haze_img)
		self.s2r_dehazing_feat = self.out_r[0]
		self.s_dehazing_feat = self.out_s[0]
		self.s2r_dehazing_img = self.out_r[-1]
		self.s_dehazing_img = self.out_s[-1]
		self.loss_G_S2R = self.criterionGAN(self.netD_R(self.img_s2r), True)
		self.loss_G_Rfeat = self.criterionGAN(self.netD_Rfeat(self.s2r_dehazing_feat), True) * lambda_gan_feat
		self.loss_cycle_S = self.criterionCycle(self.s_rec_img, self.syn_haze_img) * lambda_S
		self.loss_idt_S = self.criterionIdt(self.idt_S, self.syn_haze_img) * lambda_S * lambda_idt
		size = len(self.out_s)
		self.loss_S_Dehazing = 0.0
		clear_imgs = task.scale_pyramid(self.clear_img, size - 1)
		for (s_dehazing_img, clear_img) in zip(self.out_s[1:], clear_imgs):
			self.loss_S_Dehazing += self.criterionDehazing(s_dehazing_img, clear_img) * lambda_Dehazing
		self.loss_S2R_Dehazing = 0.0
		for (s2r_dehazing_img, clear_img) in zip(self.out_r[1:], clear_imgs):
			self.loss_S2R_Dehazing += self.criterionDehazing(s2r_dehazing_img, clear_img) * lambda_Dehazing
		self.loss = self.loss_G_S2R + self.loss_G_Rfeat + self.loss_cycle_S + self.loss_idt_S + self.loss_S_Dehazing + self.loss_S2R_Dehazing
		self.loss.backward()

		# ============================= real =============================
		self.img_r2s = self.netR2S(self.real_haze_img, self.real_depth, True)
		self.idt_R = self.netS2R(self.real_haze_img, self.real_depth, True)
		self.r_rec_img = self.netS2R(self.img_r2s, self.real_depth, True)
		self.out_s = self.netS_Dehazing(self.img_r2s)
		self.out_r = self.netR_Dehazing(self.real_haze_img)
		self.r_dehazing_feat = self.out_r[0]
		self.r2s_dehazing_feat = self.out_s[0]
		self.r_dehazing_img = self.out_r[-1]
		self.r2s_dehazing_img = self.out_s[-1]
		self.loss_G_R2S = self.criterionGAN(self.netD_S(self.img_r2s), True)
		self.loss_G_Sfeat = self.criterionGAN(self.netD_Sfeat(self.r2s_dehazing_feat), True) * lambda_gan_feat
		self.loss_cycle_R = self.criterionCycle(self.r_rec_img, self.real_haze_img) * lambda_R
		self.loss_idt_R= self.criterionIdt(self.idt_R, self.real_haze_img) * lambda_R * lambda_idt

		# TV LOSS

		self.loss_R2S_Dehazing_TV = self.TVLoss(self.r2s_dehazing_img) * self.opt.lambda_Dehazing_TV
		self.loss_R_Dehazing_TV = self.TVLoss(self.r_dehazing_img) * self.opt.lambda_Dehazing_TV


		# DC LOSS

		self.loss_R2S_Dehazing_DC = DCLoss((self.r2s_dehazing_img + 1) / 2, self.opt.patch_size) * self.opt.lambda_Dehazing_DC
		self.loss_R_Dehazing_DC = DCLoss((self.r_dehazing_img + 1) / 2, self.opt.patch_size) * self.opt.lambda_Dehazing_DC

		# dehazing consistency
		self.loss_Dehazing_Con = 0.0
		for (out_s1, out_r2) in zip(self.out_s, self.out_r):
			self.loss_Dehazing_Con += self.criterionCons(out_s1, out_r2) * lambda_Dehazing_Con

		self.loss_G = self.loss_G_R2S + self.loss_G_Sfeat + self.loss_cycle_R + self.loss_idt_R + self.loss_R2S_Dehazing_TV \
					  + self.loss_R_Dehazing_TV + self.loss_R2S_Dehazing_DC + self.loss_R_Dehazing_DC + self.loss_Dehazing_Con
		self.loss_G.backward()
		self.real_dehazing_img = (self.r_dehazing_img + self.r2s_dehazing_img) / 2.0
		self.syn_dehazing_img = (self.s_dehazing_img + self.s2r_dehazing_img) / 2.0


	def optimize_parameters(self):

		self.forward()
		self.set_requires_grad([self.netD_S, self.netD_R, self.netD_Sfeat, self.netD_Rfeat], False)
		self.optimizer_G_trans.zero_grad()
		self.optimizer_G_task.zero_grad()
		self.backward_G()
		self.optimizer_G_trans.step()
		self.optimizer_G_task.step()

		self.set_requires_grad([self.netD_S, self.netD_R, self.netD_Sfeat, self.netD_Rfeat], True)
		self.optimizer_D.zero_grad()
		self.backward_D_S()
		self.backward_D_R()
		self.backward_D_Sfeat()
		self.backward_D_Rfeat()
		self.optimizer_D.step()

