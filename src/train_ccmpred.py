import os
import sys
import torch
import argparse
from Training import CCMPredTrainer
from Dataset import get_msa_stream
from tqdm import tqdm


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep qa')	
	parser.add_argument('-experiment', default='QA3', help='Experiment name')
	parser.add_argument('-dataset', default='CASP', help='Dataset name')
	
	parser.add_argument('-lr', default=0.001 , help='Learning rate')
	parser.add_argument('-lrd', default=0.0001 , help='Learning rate decay')
	parser.add_argument('-wd', default=0.0, help='Weight decay')
	parser.add_argument('-tm_score_threshold', default=0.1, help='GDT-TS score threshold')
	parser.add_argument('-gap_weight', default=0.1, help='Gap weight')
	parser.add_argument('-max_epoch', default=150, help='Max epoch')
	
	args = parser.parse_args()

	# torch.cuda.set_device(0)
	stream_train = get_msa_stream("../database/1atzA.aln", shuffle=False)
	trainer = CCMPredTrainer(L = stream_train.dataset.L, lr = args.lr, weight_decay=args.wd, lr_decay=args.lrd)

	for epoch in xrange(args.max_epoch):
		
		av_loss = 0.0
		for data in tqdm(stream_train):
			loss = trainer.optimize(data)
			av_loss += loss
		
		av_loss/=len(stream_train)
		print 'Loss training = ', av_loss
		