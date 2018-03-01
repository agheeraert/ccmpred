#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:02:15 2018

@author: Aria
"""

import reweighting
import convert_msa
import scoring
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim as optim


#Parameters
x = 0.9 #Threshold to consider two proteins similar
lambdah = 0.01 #l2 regularization parameters
lambdaj = 0.01
learningRate = 0.001
NStep=250 #Number of optimization steps
a = 1.5 #Factor used to show the a.N strongest ranked pairs

# Conversion of the aln file into a list of lists
with open("../database/1atzA.aln", "r") as msa_file:
    msa = []
    for line in msa_file.readlines():
        msa.append(list(line))
        

#Conversion of the list of lists into an array and tensor with shared memory
msa_np = np.asarray(convert_msa.to_numbers(msa), np.int32)
msa = torch.from_numpy(msa_np)

#""" Moving the tensor on the GPU
if torch.cuda.is_available():
    msa = msa.cuda()
#"""

#Getting B and N
B, N = msa.size()[0], msa.size()[1]
  
#Reweighting part
weightsList = Variable(reweighting.reweight(msa, B, N, x))
Beff = torch.sum(weightsList)


#Defining the tensors to optimize
h = Variable(torch.zeros(N, 21), requires_grad=True)
J = Variable(torch.zeros(N, N, 21, 21), requires_grad=True)
# Moving the tensors on the GPU
if torch.cuda.is_available():
    h = h.cuda()
    J = J.cuda()

optimizer = optim.SGD([{'params': h, 'weight_decay': lambdah}, #the l2 regularization are applied here
                       {'params': J, 'weight_decay': lambdaj}
                       ], lr=learningRate) 
    
print('Starting optimization')    
for tour in range(NStep):
    optimizer.zero_grad()
    h_cap = torch.zeros(B, N)
    s1 = torch.zeros(B, N, N) #generation of the first sum
    s3 = torch.zeros(B, N, 21) #generation of the third sum
    for b in range(B):
        print(b)
        for r in range(N):
            aminoAcid = msa[b][r]
            h_cap[b][r] = float(h[r][aminoAcid])
            for i in range(N):
                if i != r:
                    s1[b][r][i] = float(J[r][i][aminoAcid][msa[b][i]])
            for l in range(21):
                for i in range(N):
                    if i != r:
                        s3[b][r][l] += float(J[r][i][l][msa[b][i]])                                   
    s2 = torch.sum(
            torch.addcmul(
            Variable(torch.zeros(B, N, 21)),
            torch.exp(h),
            Variable(torch.exp(s3))
            ), 2)
    
    P = torch.addcmul(
            Variable(torch.zeros(B, N)),
            torch.exp(h_cap + torch.sum(s1, 2)),
            1./s2)
    
    #lpseudo = -1./B*torch.sum(torch.log(P))
    lpseudo = -1./Beff*torch.sum(
            torch.addcmul(
            Variable(torch.zeros(B, N)),
            weightsList.unsqueeze(1), 
            torch.log(P)))
    
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

"""


for tour in range(NStep):
    optimizer.zero_grad()
    #Computation of lpseudo
    lpseudo = 0 
    for b in range(B):
        sumr = 0 # may be replaced by -Beff*gr in the future
        for r in range(N): #Computation of P(sigma_r = sigma_r^(b) | sigma_\r = sigma_\r^(b))
           sum1, sum2, sum3 = 0, 0, 0 #corresponds to the sums in (14) (left to right, up to bottom)
           aminoAcid = msa[b][r]
           for i in range(N):
               correlatedAminoAcid = msa[b][i]
               if i != r:
                   sum1 += J[r][i][aminoAcid][correlatedAminoAcid]
           for l in range(21):
               for i in range(N):
                   if i != r:
                       sum3 += J[r][i][l][msa[b][i]]
               sum2 += np.exp(h[r][l]+sum3)
           P = np.exp(h[r][aminoAcid])
           sumr += np.log(P)  
        #lpseudo += listOfWeights[b]*sumr 
        lpseudo += sumr
    #lpseudo /= -1./Beff 
    lpseudo /= -1./B
    #optimizing
    lpseudo.backward()    
    optimizer.step()
    
    
                for l in range(21):
                for i in range(N):
                    if i != r:
                        s3[b][r][i] += float(J[r][i][l][msa[b][i]])

"""



    
    


