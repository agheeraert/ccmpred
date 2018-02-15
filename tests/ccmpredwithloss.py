#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:02:15 2018

@author: Aria
"""

import reweighting
import convert_msa
import scoring
import loss_function

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from Bio.PDB.Polypeptide import aa1


#Parameters
x = 0.9 #Threshold to consider two proteins similar
lambdah = 0.01 #l2 regularization parameters
lambdaj = 0.01
learningRate = 0.001
NStep=250 #Number of optimization steps
a = 1.5 #Factor used to show the a.N strongest ranked pairs
toggle_reweight = False


with open("../database/1atzA.aln", "r") as msa_file:
    msa = []
    for line in msa_file.readlines():
        msa.append(list(line.split()[0]))
aa = list(aa1 + '-')
aa_to_idx = {}
for i, a in enumerate(aa):
    aa_to_idx[a] = i

B = len(msa)
N = len(msa[0]) #sequence length
q = len(aa) #number of aa

#parameters of the model
H = nn.Embedding(q, N)
J = nn.Embedding(q*q, N*N)

#Reweighting part
if toggle_reweight:
    weightsList = Variable(reweighting.reweight(msa, B, N, x))
    Beff = torch.sum(weightsList)

optimizer = optim.SGD([{'params': H.parameters(), 'weight_decay': lambdah}, #the l2 regularization are applied here
                       {'params': J.parameters(), 'weight_decay': lambdaj}
                       ], lr=learningRate) 
    
print('Starting optimization')    
for epoch in range(NStep):
    optimizer.zero_grad()
    loss = Variable(torch.zeros(B, N))
    for b in range(B):
        print(b)
        for r in range(N):
            loss[b, r] = loss_function.loss(msa, H, J, b, r)
    lpseudo = torch.sum(loss)      
    lpseudo.backward()    
    optimizer.step()
    
correctedNorm = scoring.corrected_norm(J, N)

# Getting the top a.N strongest ranked pairs 
# We use a trick because topk only works in one dimension, we flatten the tensor first
topPairsFlattened = torch.topk(correctedNorm.view(-1), int(a*N))[1]
# And then get the position... Euclidean division doesn't seem to be supported by Pytorch
topPairs = torch.zeros(2, int(a*N))
for i in range(int(a*N)):
    topPairs[0][i] = topPairsFlattened[i]//N
    topPairs[1][i] = topPairsFlattened[i]%N
    
"""
with open("../database/contactmap", "r") as contact_map:
    
"""

# Then plotting the contact map   
plt.title("Contact map")
plt.xlabel("Position of the amino acid i") 
plt.ylabel("Position of the amino acid j")
plt.scatter(topPairs[0], topPairs[1], c='red', edgecolor='black')
plt.scatter(topPairs[1], topPairs[0], c='red', edgecolor='black')
plt.plot(range(N), c='black')
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()