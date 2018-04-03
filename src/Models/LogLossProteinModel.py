import os
import torch
from torch.autograd import Function, Variable
import torch.nn as nn
import sys
import math

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../ProteinClassesLibrary"))
from PythonInterface import Angles2CoordsAB,Coords2Pairs, Angles2CoordsDihedral
from matplotlib import pylab as plt

def initialize(layer):
	for parameter in layer.parameters():
		if len(parameter.size())>1:
			parameter.data.normal_()/math.sqrt(parameter.size(0)*parameter.size(1))

class LogLossProteinModel(nn.Module):
	'''
	Negative log loss given position and sequence
	'''
	def __init__(self, L, q=21, lambda_h=0.0014):
		super(LogLossProteinModel, self).__init__()
		self.L = L
		self.q = q
		self.K = nn.Parameter(torch.ones(self.q, self.q).cuda())
		self.H = nn.Parameter(torch.ones(self.q, self.L).cuda())
		
		self.cos_alpha = nn.Parameter(torch.randn(L-1).cuda())
		self.sin_alpha = nn.Parameter(torch.randn(L-1).cuda())
		self.cos_beta = nn.Parameter(torch.randn(L-1).cuda())
		self.sin_beta = nn.Parameter(torch.randn(L-1).cuda())

		
		self.length = Variable(torch.IntTensor(1).fill_(self.L-1))
		self.a2c = Angles2CoordsAB(self.L-1)
		# self.a2c = Angles2CoordsDihedral(self.L-1)
		self.c2p = Coords2Pairs(self.L-1)
		
		self.lambda_h = lambda_h
		
		self.all_aa = torch.LongTensor([i for i in range(0, q)])
		self.all_aa = self.all_aa.cuda()
		self.all_aa = Variable(self.all_aa)

		self.mask = torch.ones(self.L, self.L).cuda()
		for i in range(self.L):
			self.mask[i,i]=0.0
		self.mask = Variable(self.mask)
		self.sigmoid = nn.Sigmoid()
		self.apply(initialize)
		self.symmetrize()

	def contact_matrix(self):
		"""
		Returns the contact matrix
		"""		
		
		return self.get_cmap().data

	def aa_interactions(self):
		"""
		Shows the interactions between amino acids
		"""
		return self.K.data

	def get_cmap(self, threshold=3.5):
		alpha = torch.atan(self.sin_alpha/self.cos_alpha+0.001*torch.sign(self.cos_alpha))
		beta = torch.atan(self.sin_beta/self.cos_beta+0.001*torch.sign(self.cos_beta))
		angles = torch.stack([alpha,beta]).contiguous()
		
		coords = self.a2c(angles, self.length).contiguous()
		pairs = self.c2p(coords, self.length).contiguous()

		pairs2d = pairs.resize(3, self.L, self.L)
		dist2d = torch.sqrt(pairs2d[0,:,:]*pairs2d[0,:,:] + pairs2d[1,:,:]*pairs2d[1,:,:] + pairs2d[2,:,:]*pairs2d[2,:,:] + 0.0001)
		cmap = 1.0 - self.sigmoid(dist2d-threshold)
		return cmap


	def forward(self, sigma_i, w_b):
		Kir = self.K[sigma_i,:]
		Jir = self.get_cmap()
		Jir = Jir*self.mask
		#nominator
		N = -((Kir[:,sigma_i] * Jir).sum(dim=0) + self.H[sigma_i,:].diag()).sum(dim=0)
		#denominator
		Kli = self.K[:, sigma_i]
		D = torch.log(torch.exp(torch.matmul(Kli,Jir) + self.H).sum(dim=0)).sum()
		Lpseudo = w_b[0]*(N + D)
		Lpseudo_noreg = Lpseudo        
		
		#regularization
		Lpseudo += self.lambda_h*torch.sum(self.H*self.H)
		
		return Lpseudo, Lpseudo_noreg
		
	def symmetrize(self):
		"""
		Computes the symmetric K and K'
		"""
		
		Kpt = self.K.data.t()
		Kps = 0.5*(Kpt+self.K.data)
		self.K.data.copy_(Kps)

	def save(self):
		"""
		Creates an output file containing the computed H and J
		"""
		if not os.path.isdir('../results/'):
			os.mkdir('../results/')

		torch.save(self.H, '../results/1BDO_A_H.out')
		torch.save(self.K, '../results/1BDO_A_K.out')
		torch.save(self.cos_alpha, '../results/1BDO_A_cosa.out')
		torch.save(self.sin_alpha, '../results/1BDO_A_sina.out')
		torch.save(self.cos_beta, '../results/1BDO_A_cosb.out')
		torch.save(self.sin_beta, '../results/1BDO_A_sinb.out')


	def load(self):
		"""
		Creates an output file containing the computed H and J
		"""
		
		self.H = torch.load('../results/1BDO_A_H.out')
		self.K = torch.load('../results/1BDO_A_K.out')
		self.cos_alpha = torch.load('../results/1BDO_A_cosa.out')
		self.sin_alpha = torch.load('../results/1BDO_A_sina.out')
		self.cos_beta = torch.load('../results/1BDO_A_cosb.out')
		self.sin_beta = torch.load('../results/1BDO_A_sinb.out')

	def lambdas(self):
		return self.lambda_h
	
if __name__=="__main__":
	L = 100
	ll = LogLossProteinModel(L)
	cmap = ll.get_cmap()
	f = plt.figure()
	plt.imshow(cmap .cpu().data.numpy())
	plt.show()
