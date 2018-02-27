import os
import torch
from torch.autograd import Function, Variable
import torch.nn as nn

class LogLossRB(nn.Module):
	'''
	Negative log loss given position and sequence
	'''
	def __init__(self, L, q=21, gpu=False, lambda_h=0.01, lambda_j=0.01 ):
		super(LogLossRB, self).__init__()
		self.L = L
		self.q = q
		self.H = nn.Embedding(q, L)
		self.J = nn.Embedding(q*q, L*L)
		self.gpu = gpu
		self.lambda_h = lambda_h
		self.lambda_j = lambda_j

		if gpu:
			self.H = self.H.cuda()
			self.J = self.J.cuda()
		self.all_aa = torch.LongTensor([i for i in range(0, q)])
		if gpu:
			self.all_aa = self.all_aa.cuda()
		self.all_aa = Variable(self.all_aa)
	
	def symmetrize(self):
		"""
		Computes the symmetric J: J_{ij}(sigma_k,sigma_l) = J_{ji}(sigma_l, sigma_k)
		"""
		for J in self.J.parameters():
			if self.gpu:
				Jp = torch.FloatTensor(self.q, self.q, self.L, self.L).cuda()
			else:
				Jp = torch.FloatTensor(self.q, self.q, self.L, self.L)

			Jp.copy_(J.view(self.q, self.q, self.L, self.L).data)
			Jp = torch.transpose(torch.transpose(Jp, dim0=0, dim1=1), dim0=2, dim1=3)
			Jp = (J.view(self.q, self.q, self.L, self.L).data + Jp)*0.5
		J.data.copy_(Jp.view(self.q*self.q, self.L*self.L))
	
	def create_output(self):
		"""
		Creates an output file containing the computed H and J
		"""
		if not os.path.isdir('../results/'):
			os.mkdir('../results/')

		torch.save(self.H, '../results/1BDO_A_H.out')
		torch.save(self.J, '../results/1BDO_A_J.out')	
	
	def contact_matrix(self):
		"""
		Computes renormalized matrix
		"""
		for J in self.J.parameters():
			Jp = J.data
		
		Jp = Jp.view(self.q, self.q, self.L, self.L)
		Jp = Jp - torch.mean(Jp, dim=0) - torch.mean(Jp, dim=1) + torch.mean(torch.mean(Jp, dim=0), dim=0)
		S_FN = Jp*Jp
		S_FN = torch.sqrt(torch.sum(torch.sum(S_FN, dim=0), dim=0))

		S_CN = S_FN - torch.mean(S_FN, dim=0).resize_(1,self.L) * torch.mean(S_FN, dim=1).resize_(self.L,1) / torch.mean(S_FN)
		return S_CN


	def forward(self, sigma_r, sigma_i, sigma_ri, r, w_b):
		r = r[0]
		mask = torch.eye(self.L)
		if self.gpu:
			mask = mask.cuda()
		mask[r,r] = 0.0
		mask = Variable(mask)

		mask_extended = torch.FloatTensor(self.q,self.L,self.L)
		if self.gpu:
			mask_extended = mask_extended.cuda()

		for i in range(0,self.q):
			mask_extended[i, :, :].copy_(mask.data)
		mask_extended = Variable(mask_extended)

		J_rij = self.J(sigma_i).resize(self.L, self.L, self.L)[:,r,:]
		nominator = torch.exp(self.H(sigma_r)[0,r] + (J_rij*mask).sum())
		
		J_rili = self.J(sigma_ri).resize(self.q, self.L, self.L, self.L)
		J_ili = torch.squeeze(J_rili[:,:,r,:])
		J_l = (J_ili*mask_extended).sum(dim=1).sum(dim=1)
		denominator = torch.exp(self.H(self.all_aa)[:,r] + J_l).sum()

		Lpseudo = (-torch.log(nominator) + torch.log(denominator))*w_b[0]

		#regularization
		for H in self.H.parameters():
			Lpseudo += self.lambda_h*torch.sum(H*H)
		for J in self.J.parameters():
			Lpseudo += self.lambda_j*torch.sum(J*J)

		return Lpseudo


if __name__=='__main__':
	from Bio.PDB.Polypeptide import aa1
	aa = list(aa1 + '-')
	aa_to_idx = {}
	for i, a in enumerate(aa):
		aa_to_idx[a] = i

	print(aa_to_idx)

	with open("../../database/1BDO_A.aln", "r") as msa_file:
		msa = []
		for line in msa_file.readlines():
			msa.append(list(line.split()[0]))

	L = len(msa[0]) #sequence length
	q = len(aa) #number of aa
	b = 1
	r = 2

	s_r = Variable(torch.LongTensor([aa_to_idx[msa[b][i]]]))
	s_i = []
	for i, aa in enumerate(msa[b]):
		s_i.append(aa_to_idx[msa[b][r]]*q + aa_to_idx[aa])
	s_i = Variable(torch.LongTensor(s_i))

	all_aa_si = torch.LongTensor(q,L)
	for i in range(0,q):
		for j, aa in enumerate(msa[b]):
			all_aa_si[i,j] = i*q + aa_to_idx[aa]
	all_aa_si = Variable(all_aa_si)

	loss = LogLossRB(L)
	for parameter in loss.parameters():
		print(parameter)

	y = loss(s_r, s_i, all_aa_si, r)
	y.backward()