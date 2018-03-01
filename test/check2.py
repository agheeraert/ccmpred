import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import convert_msa

from Bio.PDB.Polypeptide import aa1

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

    print(aa_to_idx)

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

    #Computing denominator
    J_rili = J(all_aa_si).resize(q,L,L,L)
    J_ili = J_rili[:,:,r,:]
    J_l = (J_ili*mask_extended).sum(dim=1).sum(dim=1)
    denominator = torch.exp(H(all_aa)[:,r] + J_l).sum()
    
    #neg log likelihood
    Lpseudo = -torch.log(nominator) + torch.log(denominator)
    print(Lpseudo)


    s1, s2, s3 = 0, 0, 0
    for i in range(L):
        if i != r:
            s_i = Variable(getQ(msa[b], i))
            s_ir = q*s_r.add(s_i)
            s1 += J(s_ir)[0, r+i*L]
    for l in range(q):
        s3 = 0
        s_l = Variable(torch.LongTensor([l]))
        for i in range(L):
            if i != r:
                s_i = Variable(getQ(msa[b], i))
                s_il = q*l+s_i
                s3 += J(s_il)[0, r+i*L]
        s2 += torch.exp(H(s_l)[0, r]+s3)
                                     
    lpseudo = -torch.log(torch.exp((H(s_r)[0, r] + s1))) + torch.log(s2)
    print(lpseudo)
    print(lpseudo == Lpseudo)