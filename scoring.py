#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:02:15 2018

@author: Aria
"""
import torch

def frobenius_norm(J, N):
    """ returns a Frobenius Norm tensor (N x N) of a tensor J of size N x N x 21 x 21 """
    
    #Construction of the average tensors used many times in the calculus of the zero-sum gauge
    avgJk = torch.mean(J, 2)
    avgJl = torch.mean(J, 3)
    avgJkl = torch.mean(avgJk, 2)
    #Construction of the square of the scoring tensor
    FN = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            for k in range(21):
                for l in range(21):
                    FN[i][j] += (J[i][j][k][l] - avgJk[i][j][l] - avgJl[i][j][k] + avgJkl[i][j])**2
    return torch.sqrt(FN)
    
def corrected_norm(J, N):
    """ returns the corrected norm tensor (N x N) of a tensor J of size N x N x 21 x 21 """
    FN = frobenius_norm(J, N)
    #construction of the average tensors used many times in the calculus of the corrected norm
    avgFNi = 1.*torch.mean(FN, 0)
    avgFNj = 1.*torch.mean(FN, 1)
    avgFN = 1.*torch.mean(avgFNi)
    CN = torch.zeros(N, N)
    for i in range(N):
        for j in range(N):
            CN[i][j] = FN[i][j] - 1.*(avgFNi[j] * avgFNj[i])/avgFN
    return CN
            