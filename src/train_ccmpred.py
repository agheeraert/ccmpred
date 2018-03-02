import os
import sys
import torch
import argparse
from Training import CCMPredTrainer
from Dataset.MSASampler import get_msa_stream
from Dataset.MSASamplerKKp import get_msa_streamKKp
from tqdm import tqdm

import matplotlib.pylab as plt
import numpy as np

def plot_mat(filename):
	mat = []
	with open(filename, 'r') as fin:
		for line in fin:
			line = line.replace(',', '.')
			vec = []
			for dig in line.split('\t')[:-1]:
				vec.append(float(dig))
			mat.append(vec)
	mat = np.array(mat)

	f = plt.figure()
	plt.title("CCMPred result")
	plt.imshow(mat)
	plt.colorbar()
	plt.savefig('../results/ccmpred.png')


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='CCMPred training')	
	
	parser.add_argument('-lr', default=0.0003 , help='Learning rate')
	parser.add_argument('-lrd', default=0.0001, help='Learning rate decay')
	parser.add_argument('-wd', default=0.0, help='Weight decay')
	parser.add_argument('-max_epoch', default=50, help='Max epoch')
	parser.add_argument('-gpu', default=None, help='Use gpu')
	parser.add_argument('-gpu_num', default=1, help='GPU number')
	parser.add_argument('-method', default='K', help='Use the method with J or K.K\'')


	
	args = parser.parse_args()
	if args.gpu is None:
		gpu = False
	else:
		gpu = True
	
	torch.cuda.set_device(int(args.gpu_num))
	loss_list = []
	
	if args.method == 'K':
		stream_train = get_msa_streamKKp("../database/1BDO_A.aln", shuffle=True)
	else:
		stream_train = get_msa_stream("../database/1BDO_A.aln", shuffle=True)
	
	trainer = CCMPredTrainer.CCMPredTrainer(L = stream_train.dataset.L, q = stream_train.dataset.q, lr = args.lr, weight_decay=args.wd, lr_decay=args.lrd, gpu=gpu, method=args.method)

	# trainer.model.load()
	# cmat = trainer.model.contact_matrix().cpu().numpy()
	# plt.title("This algorithm result")
	# plt.imshow(cmat)
	# plt.colorbar()
	# plt.show()
	# plot_mat('../database/1BDO_A.mat')
	# aaint = trainer.model.aa_interactions().cpu().numpy()
	# plt.title("Amino-acid interactions")
	# plt.imshow(aaint)
	# plt.colorbar()
	# plt.show()		
	# sys.exit()


	for epoch in tqdm(range(args.max_epoch)):
		av_loss = 0.0
		for data in stream_train:
			loss = trainer.optimize(data)
			av_loss += loss

		loss_list.append(loss)
	
	
	trainer.model.save()
	if not gpu:
		cmat = trainer.model.contact_matrix().numpy()
	else:
		cmat = trainer.model.contact_matrix().cpu().numpy()
	
	f = plt.figure()
	plt.title("This algorithm result")
	plt.imshow(cmat)
	plt.colorbar()
	plt.savefig('../results/J.png')
	
	if args.method == 'K':
		if not gpu:
			aaint = trainer.model.aa_interactions().numpy()
		else:
			aaint = trainer.model.aa_interactions().cpu().numpy()
	
		f = plt.figure()
		plt.title("Amino-acid interactions")
		plt.imshow(aaint)
		plt.colorbar()
		plt.savefig('../results/K.png')		

	f = plt.figure()
	plt.title("Loss vs epoch")
	plt.plot(loss_list)
	plt.savefig('../results/loss.png')
	