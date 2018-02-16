import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from Bio.PDB.Polypeptide import aa1

import numpy as np

def getQ(seq, i):
    return torch.LongTensor([aa_to_idx[seq[i]]])

def getQQ(seq, i, j):
    L = len(seq)
    return torch.LongTensor([aa_to_idx[seq[i]]*L + aa_to_idx[seq[j]]])

if __name__=='__main__':

    aa = list(aa1 + '-')
    aa_to_idx = {}
    for i, a in enumerate(aa):
        aa_to_idx[a] = i

    print aa_to_idx

    with open("../database/1atzA.aln", "r") as msa_file:
        msa = []
        for line in msa_file.readlines():
            msa.append(list(line.split()[0]))

    L = len(msa[0]) #sequence length
    q = len(aa) #number of aa

    #parameters of the model
    H = nn.Embedding(q, L)
    J = nn.Embedding(q*q, L*L)
    
    b = 0 #num of sequence in MSA
    r = 2 #num of amino-acid in sequence

    s_r = Variable(getQ(msa[b], r))
    s_i = []
    for i, aa in enumerate(msa[b]):
        s_i.append(aa_to_idx[msa[b][r]]*q + aa_to_idx[aa])
    s_i = Variable(torch.LongTensor(s_i))

    #mask to sum over repeting indexes in 2d
    mask = torch.eye(L)
    mask[r,r] = 0.0
    mask = Variable(mask)

    #mask to sum over repeting indexes in 3d
    mask_extended = torch.FloatTensor(q,L,L)
    for i in range(0,q):
        mask_extended[i, :, :].copy_(mask.data)
    mask_extended = Variable(mask_extended)

    all_aa = Variable(torch.LongTensor([i for i in range(0, q)]))
    all_aa_si = torch.LongTensor(q,L)
    for i in range(0,q):
        for j, aa in enumerate(msa[b]):
            all_aa_si[i,j] = i*q + aa_to_idx[aa]
    all_aa_si = Variable(all_aa_si)

    #Computing nominator
    J_rij = J(s_i).resize(L, L, L)[:,r,:]
    nominator = torch.exp(H(s_r)[0,r] + (J_rij*mask).sum())

    #Checking if nominator is correct
    Hparam = list(H.parameters())[0]
    Hparam = torch.FloatTensor(q,L).copy_(Hparam.data)
    Jparam = list(J.parameters())[0]
    Jparam = torch.FloatTensor(q,q,L,L).copy_(Jparam.data)

    sigma_r = aa_to_idx[msa[b][r]]
    nominator_naive = Hparam[sigma_r, r]
    for i in range(0,L):
        sigma_i = aa_to_idx[msa[b][i]]
        if i != r:
            nominator_naive += Jparam[sigma_r, sigma_i, r, i]
    nominator_naive = np.exp(nominator_naive)

    print 'Nominator check = ', nominator_naive, nominator.data[0]

    
    #Computing denominator
    J_rili = J(all_aa_si).resize(q,L,L,L)
    J_ili = J_rili[:,:,r,:]
    J_l = (J_ili*mask_extended).sum(dim=1).sum(dim=1)
    denominator = torch.exp(H(all_aa)[:,r] + J_l).sum()

    #Checking if denominator is correct
    denominator_naive = 0.0
    for l in range(0, q):
        denominator_naive_l = Hparam[l,r]
        for i in range(0,L):
            sigma_i = aa_to_idx[msa[b][i]]
            if i != r:
                denominator_naive_l += Jparam[l, sigma_i, r, i]
        denominator_naive += np.exp(denominator_naive_l)

    print 'Denominator check = ', denominator_naive, denominator.data[0]

    
    #neg log likelihood
    L = -torch.log(nominator) + torch.log(denominator)
    print L
    