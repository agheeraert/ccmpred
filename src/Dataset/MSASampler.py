import os
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import atexit
import numpy as np
import _pickle as pkl
#import cPickle as pkl

from os import listdir
from os.path import isfile
import random
random.seed(42)

# sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from Bio.PDB.Polypeptide import aa1

class MSASampler(Dataset):
	"""
	The dataset that loads msa and samples b and r
	"""
	def __init__(self, filename, max_iter = 1000):
		"""
		"""
		self.filename = filename
		aa = list(aa1 +'-')
		self.aa_to_idx = {}
		for i, a in enumerate(aa):
			self.aa_to_idx[a] = i

		with open(self.filename, "r") as msa_file:
			self.msa = []
			for line in msa_file.readlines():
				self.msa.append(list(line.split()[0]))

		self.L = len(self.msa[0]) #sequence length
		self.M = len(self.msa)
		self.q = len(aa) #number of aa
		for m in range(self.M):
			for l in range(self.L):
				if self.msa[m][l] in self.aa_to_idx:
					self.msa[m][l] = self.aa_to_idx[self.msa[m][l]]
				else:
					self.msa[m][l] = 20
		self.msa = torch.from_numpy(np.asarray(self.msa))

		self.indexing = []
		for i in range(0,self.M):
			for j in range(0,self.L):
				self.indexing.append((i,j))

		if max_iter < len(self.indexing):
			random.shuffle(self.indexing)
			self.indexing = self.indexing[:max_iter]

		self.dataset_size = len(self.indexing)

	def __getitem__(self, index):
		"""
		"""
		b, r = self.indexing[index]
		s_r = torch.LongTensor([self.msa[b, r]])
		s_i = []
		for i, aa in enumerate(self.msa[b]):
			s_i.append(self.msa[b, r]*self.q + aa)
	    
		s_i = torch.LongTensor(s_i)
		all_aa_si = torch.LongTensor(self.q,self.L)
		for i in range(0,self.q):
			for j, aa in enumerate(self.msa[b]):
				all_aa_si[i,j] = i*self.q + aa

		#Reweighting		
		sims = 0
		for m in range(self.M):
			if torch.sum(torch.eq(self.msa[b], self.msa[m])) > 0.9*self.L:
		 		sims += 1
		w_b = 1./sims

		return s_r, s_i, all_aa_si, r, w_b

	def __len__(self):
		"""
		Returns length of the dataset
		"""
		return self.dataset_size

def get_msa_stream(filename, batch_size = 1, shuffle = True):
	dataset = MSASampler(filename)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0)
	return trainloader
