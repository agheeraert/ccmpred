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
    maskH = torch.eye(L)
    maskH = Variable(maskH)
    
    mask2d = torch.eye(L)
    for r in range(0,L):
        mask3d[r,:,:].copy_(mask2d)
        mask3d[r,r,r] = 0.0
    mask3d = Variable(mask3d)

    #mask to sum over repeting indexes in 3d
    mask_extended = torch.FloatTensor(q,L,L)
    for i in range(0,q):
        mask_extended[i, :, :].copy_(mask2d)
    mask_extended = Variable(mask_extended)

    all_aa = Variable(torch.LongTensor([i for i in range(0, q)]))
    all_aa_si = torch.LongTensor(q,L)
    for i in range(0,q):
        for j, aa in enumerate(msa[b]):
            all_aa_si[i,j] = i*q + aa_to_idx[aa]
    all_aa_si = Variable(all_aa_si)


    sigma_r_sigma_i = torch.LongTensor( L*L )
    idx = 0
    for r, aa in enumerate(msa[b]):
        for i, aa in enumerate(msa[b]):
            sigma_r = aa_to_idx[msa[b][r]]
            sigma_i = aa_to_idx[msa[b][i]]
            sigma_r_sigma_i[idx] = sigma_r*q + sigma_i
            idx+=1

    sigma_r_sigma_i = Variable(sigma_r_sigma_i)
    print sigma_r_sigma_i

    #Computing nominator
    # print s_i
    H_rr = H(s_i).resize(L, L)
    H_r = (H_rr*maskH).sum(dim=1)

    print J(sigma_r_sigma_i).size()
    J_rij = J(sigma_r_sigma_i).resize(L, L, L, L)
    J_rij = J_rij*mask3d
    J_i = J_rij.sum(dim=0).sum(dim=1)
    nominator = torch.exp( H_r + J_i )
    # print nominator

    #Checking if nominator is correct
    Hparam = list(H.parameters())[0]
    Hparam = torch.FloatTensor(q,L).copy_(Hparam.data)
    Jparam = list(J.parameters())[0]
    Jparam = torch.FloatTensor(q,q,L,L).copy_(Jparam.data)

    sigma_r = aa_to_idx[msa[b][r]]
    nominator_naive = Hparam[sigma_r, r]
    print nominator_naive
    for i in range(0,L):
        sigma_i = aa_to_idx[msa[b][i]]
        if i != r:
            nominator_naive += Jparam[sigma_r, sigma_i, r, i]
    nominator_naive = np.exp(nominator_naive)

    # print 'Nominator check = ', nominator_naive, nominator.data[r]

    
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

    # print 'Denominator check = ', denominator_naive, denominator.data[0]

    
    #neg log likelihood
    L = -torch.log(nominator) + torch.log(denominator)
    # print L
    