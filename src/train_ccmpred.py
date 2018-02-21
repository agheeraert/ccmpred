import os
import sys
import torch
import argparse
from Training import CCMPredTrainer
from Dataset import get_msa_stream
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
	plt.show()


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='CCMPred training')	
	
	parser.add_argument('-lr', default=0.001 , help='Learning rate')
	parser.add_argument('-lrd', default=0.0001 , help='Learning rate decay')
	parser.add_argument('-wd', default=0.0, help='Weight decay')
	parser.add_argument('-max_epoch', default=200, help='Max epoch')
	parser.add_argument('-gpu', default=None, help='Use gpu')
	
	args = parser.parse_args()
	if args.gpu is None:
		gpu = False
	else:
		gpu = True
	
	# 
	# sys.exit()

	# torch.cuda.set_device(1)
	loss_list = []
	stream_train = get_msa_stream("../database/1BDO_A.aln", shuffle=True)
	trainer = CCMPredTrainer(L = stream_train.dataset.L, q = stream_train.dataset.q, lr = args.lr, weight_decay=args.wd, lr_decay=args.lrd, gpu=gpu)
	for epoch in tqdm(xrange(args.max_epoch)):
		
		av_loss = 0.0
		for data in stream_train:
			loss = trainer.optimize(data)
			av_loss += loss

		loss_list.append(loss)
	

	plot_mat('../database/1BDO_A.mat')

	f = plt.figure()
	cmat = trainer.model.contact_matrix().numpy()
	plt.title("This algorithm result")
	plt.imshow(cmat)
	plt.colorbar()
	plt.show()

	f = plt.figure()
	plt.title("Loss vs epoch")
	plt.plot(loss_list)
	plt.show()
	