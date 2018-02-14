#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:02:15 2018

@author: Aria
"""

def to_numbers(msa):
    """ Convert a list of lists of letters representing amino
    acids to a list of lists of number"""
    
    conversion = {'-': 0, 'A': 1, 'G': 2, 'S': 3, 'T': 4, 'C': 5, 'V': 6, 'L': 7, 
      'I': 8, 'M': 9, 'P': 10, 'F': 11, 'Y': 12, 'W': 13, 'D': 14, 'E': 15, 'N': 16, 
      'Q': 17, 'H': 18, 'K': 19, 'R': 20, 'X': 0, 'B': 0, 'Z': 0, 'U': 0, 'J': 0}
    for i in range(len(msa)):
        msa[i].pop()
        for j in range(len(msa[0])): #the last column is avoided because full of \n (will be deleted afterwards)           
            msa[i][j] = conversion[msa[i][j]]
    return(msa)


