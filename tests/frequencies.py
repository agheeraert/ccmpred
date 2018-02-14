import torch
from torch.autograd import Variable
from torch.nn import Embedding
import numpy as np
import convert_msa

with open("../database/1atzA.aln", "r") as msa_file:
    msa = []
    for line in msa_file.readlines():
        msa.append(list(line))
        
msa_np = np.asarray(convert_msa.to_numbers(msa), np.int32)
msa = torch.from_numpy(msa_np)

B, N = msa.size()[0], msa.size()[1]
embed = Embedding(B*N, 21)

def freqlist(msa, B, N):
    seq_of_tensors = []
    for k in range(21):
        seq_of_tensors.append(torch.sum(torch.eq(msa, k*torch.ones(B, N).int()), 1).float().unsqueeze(1))
    return torch.cat(seq_of_tensors, 1)

def cofreqlist(msa, B, N, i):
    #Generate the pivotation matrix
    #right now it's actually only the cofrequency of all aminos acid combinations at a certain distance
    pivot = torch.cat([torch.cat([torch.zeros(i, N-i), torch.eye(i)], 1), torch.cat([torch.eye(N-i), torch.zeros(N-i, i)], 1)], 0).int()
    msa_bis = torch.mm(msa, pivot)
    seq_of_tensors = []
    for k in range(21):
         for l in range(21):
             seq_of_tensors.append(torch.sum(
                torch.addcmul(torch.zeros(B, N).int(),
                torch.eq(msa, k*torch.ones(B, N).int()).int(),
                torch.eq(msa_bis, l*torch.ones(B, N).int()).int())
                , 1).float().unsqueeze(1))
    return torch.cat(seq_of_tensors, 1).view(B, 21, 21)

def totalcofreqlist(msa, B, N):
    seq_of_tensors = []
    for i in range(1, N):
        print(i)
        seq_of_tensors.append(cofreqlist(msa, B, N, i).unsqueeze(3))
    return torch.cat(seq_of_tensors, 3)

    


print(totalcofreqlist(msa, B, N))            
