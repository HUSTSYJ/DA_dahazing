import argparse
import os
from util import util
import torch
import models
class BaseOptions():
	def __init__(self):
		self.initialized = False

	def initialize(self, parser):
		parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
		parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
		parser.add_argument('--loadSizeX', type=int, default=400, help='scale images to this size')
		parser.add_argument('--loadSizeY', type=int, default=400, help='scale images to this size')
		parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
		parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
		parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
		parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
		parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
		parser.add_argument('--image_feature', type=int, default=512, help='the max channels for image features')
		parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
		parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
		parser.add_argument('--task_model_type', type=str, default='UNet', help='select model for task network [UNet] |[ResNet]')
		parser.add_argument('--trans_model_type', type=str, default='ResNet', help='select model for transform network [UNet] |[ResNet]')
		parser.add_argument('--transform_layers', type=int, default=9,help='# of number of the down sample layers for transform network')
		parser.add_argument('--image_D_layers', type=int, default=3, help='# of number of the down layers for image discriminator')
		parser.add_argument('--num_D', type=int, default=1, help='# of number of the discriminator')
		parser.add_argument('--task_layers', type=int, default=4, help='# of number of the down sample layers for task network')
		parser.add_argument('--norm', type=str, default='batch', help='batch normalization or instance normalization')
		parser.add_argument('--activation', type=str, default='PReLU', help='ReLu, LeakyReLU, PReLU, or SELU')
		parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal|xavier|kaiming]')
		parser.add_argument('--drop_rate', type=float, default=0, help='# of drop rate')
		parser.add_argument('--U_weight', type=float, default=0.1, help='weight for Unet')
		parser.add_argument('--learn_residual', action='store_true', help='if specified, model would learn only the residual to the input')
		parser.add_argument('--gan_type', type=str, default='wgan-gp', help='wgan-gp : Wasserstein GAN with Gradient Penalty, lsgan : Least Sqaures GAN, gan : Vanilla GAN')
		parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
		parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
		parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
		parser.add_argument('--model', type=str, default='CycleGanmodel', help='chooses which model to use. DA_model, S_dehazing_model, R_dehazing_model, CycleGAN_model Seminetmodel, test')
		parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
		parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
		parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
		parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
		parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
		parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
		parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
		parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
		parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
		parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
		parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
		parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
		# parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
		self.initialized = True
		return parser

	def gather_options(self):
		# initialize parser with basic options
		if not self.initialized:
			parser = argparse.ArgumentParser(
				formatter_class=argparse.ArgumentDefaultsHelpFormatter)
			parser = self.initialize(parser)

		# get the basic options
		opt, _ = parser.parse_known_args()

		# modify model-related parser options
		model_name = opt.model
		model_option_setter = models.get_option_setter(model_name)
		parser = model_option_setter(parser, self.isTrain)
		opt, _ = parser.parse_known_args()  # parse again with the new defaults

		self.parser = parser

		return parser.parse_args()

	def print_options(self, opt):
		message = ''
		message += '----------------- Options ---------------\n'
		for k, v in sorted(vars(opt).items()):
			comment = ''
			default = self.parser.get_default(k)
			if v != default:
				comment = '\t[default: %s]' % str(default)
			message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
		message += '----------------- End -------------------'
		print(message)

		# save to the disk
		if self.isTrain:
			expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
			util.mkdirs(expr_dir)
			file_name = os.path.join(expr_dir, 'opt.txt')
			with open(file_name, 'wt') as opt_file:
				opt_file.write(message)
				opt_file.write('\n')

	def parse(self):

		opt = self.gather_options()
		opt.isTrain = self.isTrain

		# opt.expr_name = opt.src_dataset + '2' + opt.tgt_dataset + '_' + opt.model
		# # process opt.suffix
		# if opt.suffix:
		# 	suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
		# 	opt.expr_name = opt.expr_name + suffix

		self.print_options(opt)

		# set gpu ids
		str_ids = opt.gpu_ids.split(',')
		opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				opt.gpu_ids.append(id)
		if len(opt.gpu_ids) > 0:
			torch.cuda.set_device(opt.gpu_ids[0])

		self.opt = opt
		return self.opt