import os
import torch
from torch.autograd import Function, Variable
import torch.nn as nn

class LogLossKKp(nn.Module):
	'''
	Negative log loss given position and sequence
	'''
	def __init__(self, L, q=21, gpu=False, lambda_h=0.01, lambda_k=0.01, lambda_kp=0.01 ):
		super(LogLossKKp, self).__init__()
		self.L = L
		self.q = q
		self.H = nn.Embedding(q, L)
		self.K = nn.Embedding(q, q)
		self.Kp = nn.Embedding(L, L)
		self.gpu = gpu
		self.lambda_h = lambda_h
		self.lambda_k = lambda_k
		self.lambda_kp = lambda_kp

		if gpu:
			self.H = self.H.cuda()
			self.K = self.K.cuda()			
			self.Kp = self.Kp.cuda()

		self.all_aa = torch.LongTensor([i for i in range(0, q)])
		if gpu:
			self.all_aa = self.all_aa.cuda()
		self.all_aa = Variable(self.all_aa)

	def contact_matrix(self):
		"""
		Returns the contact matrix
		"""
		for Kp in self.Kp.parameters():
			return Kp.data

	def aa_interactions(self):
		"""
		Shows the interactions between amino acids
		"""
		for K in self.K.parameters():
			return K.data

	def forward(self, sigma_i, w_b):
		Kl = self.K(sigma_i)		
		positions = Variable(torch.arange(self.L).long())
		if self.gpu:
		    positions = positions.cuda()

		Kpi = self.Kp(positions)
		Kpi = Kpi - Kpi.diag().diag()
		
		Jl = torch.FloatTensor(self.q, self.L)
		if self.gpu:
			Jl = Jl.cuda()
        
		Jl = torch.matmul(Kpi, Kl).t()
		dl = self.H(self.all_aa).add(Jl)

		Lpseudo = w_b[0]*((-dl[sigma_i] + dl.exp().sum(dim=0).log())).sum()

		# lpseudo = 0
		# for r in range(self.L):
		# 	s1 = 0
		# 	for i in range(self.L):
		# 		if i != r:
		# 			s1 += self.K(sigma_i)[sigma_i[r], sigma_i[i]]*self.Kp(positions)[r, i]
		# 	nominator = torch.exp(self.H(sigma_i)[r] + s1)
		# 	denominator = 0
		# 	for l in range(self.q):
		# 		s2 = 0
		# 		for i in range(self.L):
		# 			if i != r:
		# 				s2 += self.K(self.all_aa)[sigma_i[r], sigma_i[i]]*self.Kp(positions)[r, i]
		# 		denominator += torch.exp(self.H(self.all_aa)[l]+ s2)
		# 	lpseudo += nominator/denominator
		# print(lpseudo - Lpseudo)



        #regularization
		for H in self.H.parameters():
			Lpseudo += self.lambda_h*torch.sum(H*H)
		for K in self.K.parameters():
			Lpseudo += self.lambda_k*torch.sum(K*K)
		for Kp in self.Kp.parameters():
			Lpseudo += self.lambda_kp*torch.sum(Kp*Kp)

		return Lpseudo
		
	def symmetrize(self):
		"""
		Computes the symmetric K and K'
		"""
		for K in self.K.parameters():
			Kt = K.data.t()
			Ks = 0.5*(Kt+K.data)
			K.data.copy_(Ks)
		for Kp in self.Kp.parameters():
			Kpt = Kp.data.t()
			Kps = 0.5*(Kpt+Kp.data)
			Kp.data.copy_(Kps)		

	def create_output(self):
		"""
		Creates an output file containing the computed H and J
		"""
		if not os.path.isdir('../results/'):
			os.mkdir('../results/')

		torch.save(self.H, '../results/1BDO_A_H.out')
		torch.save(self.K, '../results/1BDO_A_K.out')
		torch.save(self.Kp, '../results/1BDO_A_Kp.out')