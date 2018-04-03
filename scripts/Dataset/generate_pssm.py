import os
import sys
import numpy as np
from Bio.PDB.Polypeptide import aa1
from Bio.SubsMat.MatrixInfo import blosum62

DATA_DIR = '/media/lupoglaz/LocalMSA/PDB25_id80'
if __name__=='__main__':
    aa = list(aa1 +'-')
    aa_to_idx = {}
    for i, a in enumerate(aa):
        aa_to_idx[a] = i

    names_list = []
    for filename in os.listdir(DATA_DIR):
        if filename.find('.pdb')!=-1:
            names_list.append(filename[0:-4])

    for filename in names_list:
        with open(os.path.join(DATA_DIR, filename+'.aln')) as fin:
            msa = []
            for line in fin:
                msa.append(list(line.split()[0]))
        M = len(msa) #number of sequences
        L = len(msa[0]) #sequence length
        q = len(aa)
        for m in range(M):
            for r in range(L):
                if msa[m][r] in aa_to_idx:
                    msa[m][r] = aa_to_idx[msa[m][r]]
                else:
                    msa[m][r] = 20
        msa = np.transpose(np.asarray(msa))
        frequencies_list = []
        for r in range(L):
            frequencies_list.append([np.bincount(msa[r])])
        PPM = 1./M*np.transpose(np.concatenate(frequencies_list, axis=0)) #position probability matrix

        PSSM = np.log2(PPM/q)

        #Writing the output
        PSSM = PSSM.astype("str")

        with open("../test/"+filename+".pssm", 'w') as output:
            for k in range(q):
                output.write(";".join(PSSM[k])+"\n")




    






