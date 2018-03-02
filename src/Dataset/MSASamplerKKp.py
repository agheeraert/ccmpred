import os
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import atexit
import numpy as np
import _pickle as pkl

from os import listdir
from os.path import isfile
import random
random.seed(42)

from Bio.PDB.Polypeptide import aa1


class MSASamplerKKp(Dataset):
	"""
	The dataset that loads msa and samples b and r
	"""
	def __init__(self, filename, max_iter = 1):
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

		self.indexing = [self.msa]

		if max_iter < len(self.indexing):
			random.shuffle(self.indexing)
			self.indexing = self.indexing[:max_iter]

		self.dataset_size = len(self.indexing)

	def __getitem__(self, index):
		"""
		"""
		msa = self.indexing[index].long()

		#Reweighting		
		sims_b = torch.ones(self.M)
		for b in range(self.M):
			for m in range(b+1, self.M):
				if torch.sum(torch.eq(self.msa[b].long(), self.msa[m].long())) > 0.9*self.L:
					sims_b[b] += 1
					sims_b[m] += 1
		w_b = 1./sims_b

		return msa, w_b

	def __len__(self):
		"""
		Returns length of the dataset
		"""
		return self.dataset_size

def get_msa_streamKKp(filename, batch_size = 1, shuffle = True):
	dataset = MSASamplerKKp(filename)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=0)
	return trainloader
