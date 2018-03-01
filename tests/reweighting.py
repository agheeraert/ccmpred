#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:02:15 2018

@author: Aria
"""
import torch

def similarity(msa, N, a, b):
    """"returns the similarity between two proteins of length N with indices 
    a and b in the MSA msa """
    
    return 1.*torch.sum(torch.eq(msa[a],msa[b]))/N


def reweight(msa, B, N, x):
    """ returns the weight regulating the impact of similar proteins of the 
    protein a of length N in the MSA msa containing B proteins. 
    The similarity threshold is set to x """  
    print('Starting reweighting')
    countOfSimilarities = torch.ones(B) #counting itself
    for i in range(B):
        for j in range(i+1, B): #avoid double counting and counting itself
            if similarity(msa, N, i, j) > x:
                countOfSimilarities[i] += 1 #sim(a,b)=sim(b,a)
                countOfSimilarities[j] += 1     
    print('Reweighting over')
    return 1./(countOfSimilarities)
            
        
            



