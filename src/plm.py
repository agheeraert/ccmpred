#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:02:15 2018

@author: Aria
"""
import torch
import numpy as np

def sim(msa, N, a, b):
    """"returns the similarity between two proteins of length N with indices 
    a and b in the MSA msa """
    count = 0
    for i in range(N):
         if msa[a][i] == msa[b][i]:
             count+=1
    return 1.*count/N


def weight_list(msa, B, N, x):
    """ returns the weight regulating the impact of similar proteins of the protein a of length N in the MSA msa containing B proteins. The similarity threshold is set to x """
    countOfSims = np.zeros(B) #list which 
    for i in range(B):
        for j in range(i+1, B): #avoid double counting
            if sim(msa, N, i, j) >= x:
                countOfSims[i]+=1
                countOfSims[j]+=1
        print(1./countOfSims[i])
    return 1./countOfSims
            


def regularized_frequency(msa, B, lambd, Beff, listOfWeights, i, k):
    """ compute the regularized frequency of the amino acid represented by the
    number k at the position i """
    somme = 0
    for b in range(B): #sum over b in the formula
        if msa[b][i] == k:
            somme += listOfWeights[b]
    return 1./(lambd+Beff)*(lambd/21.+somme)

def regularized_frequencies_list(msa, B, N, lambd, Beff, listOfWeights):
    """ returns a tensor with all the frequencies """
    freqTensor = torch.zeros(N, 21)
    for i in range(N):
        for k in range(21):
            freqTensor[i][k] = regularized_frequency(msa, B, lambd, Beff, listOfWeights, i, k)
    
    return freqTensor

def regularized_cofrequency(msa, B, lambd, Beff, listOfWeights, i, j, k, l):
    """ compute the regularized cofrequency of the amino acid represented by the
    number k at the position i and the amino acid represented by the number l at
    position l"""
    somme = 0
    for b in range(B): #sum over b in the formula
        if msa[b][i] == k and msa[b][j] == l:
            somme += listOfWeights[b]     
    return 1./(lambd+Beff)*(lambd/441.+somme)   

def regularized_cofrequencies_list(msa, B, N, lambd, Beff, listOfWeights):
    """ returns a tensor with all the cofrequencies """
    coFreqTensor = torch.zeros(N, N, 21, 21)
    for i in range(N):
        for j in range(i+1, N):
            for k in range(21):
                for l in range(21):
                    if k != l:
                        coFreqTensor[i][j][k][l] = regularized_cofrequency(msa, B, lambd, Beff, listOfWeights, i, j, k, l)
    return coFreqTensor




