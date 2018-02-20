import torch
from torch.autograd import Function, Variable
import torch.nn as nn

class LogLossRB(nn.Module):
	'''
	Negative log loss given position and sequence
	'''
	def __init__(self, L, q=22):
		super(LogLossRB, self).__init__()
		self.L = L
		self.q = q
		self.H = nn.Embedding(q, L)
		self.J = nn.Embedding(q*q, L*L)

		self.all_aa = Variable(torch.LongTensor([i for i in range(0, q)]))
	
	def symmetrize(self):
		
		for J in self.J.parameters():
			Jp = torch.FloatTensor(self.q, self.q, self.L, self.L).copy_(J.data)
			Jp = torch.transpose(torch.transpose(Jp, dim0=0, dim1=1), dim0=2, dim1=3)
			Jp = (J.data + Jp)*0.5
			J.data.copy_(Jp)
			

	def forward(self, sigma_r, sigma_i, sigma_ri, r):
		
		mask = torch.eye(self.L)
		mask[r,r] = 0.0
		mask = Variable(mask)

		mask_extended = torch.FloatTensor(self.q,self.L,self.L)
		for i in range(0,self.q):
			mask_extended[i, :, :].copy_(mask.data)
		mask_extended = Variable(mask_extended)
		

		J_irij = self.J(sigma_i)
		J_rij = self.J(sigma_i).resize(self.L, self.L, self.L)[:,r,:]
		J_r = (J_rij*mask).sum()
		H_r = self.H(sigma_r)
		H_r = H_r[:,r]
		nominator = torch.exp( H_r + J_r)
		
		J_rili = self.J(sigma_ri).resize(self.q, self.L, self.L, self.L)
		J_ili = torch.squeeze(J_rili[:,:,r,:])
		J_l = (J_ili*mask_extended).sum(dim=1).sum(dim=1)

		denominator = torch.exp(self.H(self.all_aa)[:,r] + J_l).sum()
		
		Lpseudo = -torch.log(nominator) + torch.log(denominator)
		return Lpseudo


if __name__=='__main__':
	from Bio.PDB.Polypeptide import aa1
	aa = list(aa1 + '-')
	aa_to_idx = {}
	for i, a in enumerate(aa):
		aa_to_idx[a] = i

	print aa_to_idx

	with open("../../database/1atzA.aln", "r") as msa_file:
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
		print parameter

	y = loss(s_r, s_i, all_aa_si, r)
	y.backward()
