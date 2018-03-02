import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import atexit
import numpy as np
import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Models import LogLossRB, LogLossFactorized



class CCMPredTrainer:
	def __init__(self, L, q, lr=0.001, weight_decay=0.0, lr_decay=0.0001, gpu = False, method = 'J'):
		self.wd = weight_decay
		self.lr = lr
		self.lr_decay = lr_decay
		self.gpu = gpu
		self.method = method
		if method == 'K':
			self.model = LogLossFactorized(L, q, gpu=gpu)
		else:
			self.model = LogLossRB.LogLossRB(L, q, gpu=gpu)
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
		if gpu:
			self.model.cuda()
		self.log = None
		# self.lr_sceduler = GeometricLR(self.optimizer, self.lr_decay)

		atexit.register(self.cleanup)
	
	def new_log(self, log_file_name):
		if not self.log is None:
			self.log.close()
		self.log = open(log_file_name, "w")

	def cleanup(self):
		if not self.log is None:
			self.log.close()

	def optimize(self, data):
		"""
		Optimization step. 
		Input: s_r, s_i, all_aa_si, r
		Output: loss
		"""
		self.optimizer.zero_grad()
		if self.method == 'K':
			s_r, w_b = data
			s_r = torch.squeeze(s_r)
			if self.gpu:
				s_r = s_r.cuda()
			s_r = Variable(s_r)
				
			model_out = self.model(s_r, w_b)
			
		else:
			s_r, s_i, all_aa_si, r, w_b = data
			s_r, s_i, all_aa_si = torch.squeeze(s_r), torch.squeeze(s_i), torch.squeeze(all_aa_si)
			if self.gpu:
				s_r, s_i, all_aa_si = s_r.cuda(), s_i.cuda(), all_aa_si.cuda()
			s_r, s_i, all_aa_si = Variable(s_r), Variable(s_i), Variable(all_aa_si)
				
			model_out = self.model(s_r, s_i, all_aa_si, r, w_b)

		model_out.backward()
		
		self.optimizer.step()
		self.model.symmetrize()

		return model_out.data