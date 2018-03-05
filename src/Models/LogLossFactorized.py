import os
import torch
from torch.autograd import Function, Variable
import torch.nn as nn
import sys
import math

def initialize(layer):
	for parameter in layer.parameters():
		parameter.data.normal_()/math.sqrt(parameter.size(0)*parameter.size(1))

class LogLossFactorized(nn.Module):
	'''
	Negative log loss given position and sequence
	'''
	def __init__(self, L, q=21, gpu=False, lambda_h=0.001, lambda_J=0.00075):
		super(LogLossFactorized, self).__init__()
		self.L = L
		self.q = q
		self.K = nn.Parameter(torch.ones(self.q, self.q).cuda())
		self.H = nn.Parameter(torch.ones(self.q, self.L).cuda())
		self.J = nn.Parameter(torch.ones(self.L, self.L).cuda())
		
		self.gpu = gpu
		self.lambda_h = lambda_h
		self.lambda_J = lambda_J
		
		self.all_aa = torch.LongTensor([i for i in range(0, q)])
		if gpu:
			self.all_aa = self.all_aa.cuda()
		self.all_aa = Variable(self.all_aa)

		self.mask = torch.ones(self.L, self.L).cuda()
		for i in range(self.L):
			self.mask[i,i]=0.0
		self.mask = Variable(self.mask)

		self.apply(initialize)
		self.symmetrize()

	def contact_matrix(self):
		"""
		Returns the contact matrix
		"""		
		Jp = self.J.view(self.L, self.L)
		S_FN = torch.sqrt(Jp*Jp)

		return S_FN.data

	def aa_interactions(self):
		"""
		Shows the interactions between amino acids
		"""
		return self.K.data

     
	def forward(self, sigma_i, w_b):

		Kir = self.K[sigma_i,:]
		Jir = nn.ReLU()(self.J)*self.mask
        #nominator
		N = -((Kir[:,sigma_i] * Jir).sum(dim=0) + self.H[sigma_i,:].diag()).sum(dim=0)
		#denominator
		Kli = self.K[:, sigma_i]
		D = torch.log(torch.exp(torch.matmul(Kli,Jir) + self.H).sum(dim=0)).sum()
		Lpseudo = w_b[0]*(N + D)        
		
		#regularization
		Lpseudo += self.lambda_h*torch.sum(self.H*self.H)
		Lpseudo += self.lambda_J*torch.sum(self.J*self.J)
		
		return Lpseudo
		
	def symmetrize(self):
		"""
		Computes the symmetric K and K'
		"""
		
		Kpt = self.K.data.t()
		Kps = 0.5*(Kpt+self.K.data)
		self.K.data.copy_(Kps)

		Jpt = self.J.data.t()
		Jps = 0.5*(Jpt+self.J.data)
		self.J.data.copy_(Jps)		

	def save(self):
		"""
		Creates an output file containing the computed H and J
		"""
		if not os.path.isdir('../results/'):
			os.mkdir('../results/')

		torch.save(self.H, '../results/1BDO_A_H.out')
		torch.save(self.K, '../results/1BDO_A_K.out')
		torch.save(self.J, '../results/1BDO_A_Kp.out')

	def load(self):
		"""
		Creates an output file containing the computed H and J
		"""
		
		self.H = torch.load('../results/1BDO_A_H.out')
		self.K = torch.load('../results/1BDO_A_K.out')
		self.J = torch.load('../results/1BDO_A_Kp.out')
	
	def final_loss(self):
		Kir = self.K[sigma_i,:]
		Jir = self.J*self.mask
		#nominator
		N = -((Kir[:,sigma_i] * Jir).sum(dim=0) + self.H[sigma_i,:].diag()).sum(dim=0)
		#denominator
		Kli = self.K[:, sigma_i]
		D = torch.log(torch.exp(torch.matmul(Kli,Jir) + self.H ).sum(dim=0)).sum()
		Lpseudo = w_b[0]*(N + D)

		with open("../results/values.txt", 'a') as output:
			output.write('lambda_h: ' + self.lambda_h, + 'lambda_J: ' + self.lambda_J, + 'Lpseudo: '+ Lpseudo)
