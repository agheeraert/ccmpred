import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    
    #remove empty msa file
    empty_msa_list = []
    for filename in names_list:
        if os.path.getsize(os.path.join(DATA_DIR, filename+'.aln')) == 0:
            empty_msa_list.append(filename)

    names_list = [filename for filename in names_list if filename not in empty_msa_list]

    for num_file, filename in enumerate(tqdm(names_list)):
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
        occurences_list = []
        for r in range(L):
            occurences_list.append([np.bincount(msa[r], minlength=q)])

        occurences = np.transpose(np.concatenate(occurences_list, axis=0))

        #pseudocounts
        occurences = occurences + np.sqrt(M)/q
        PPM = 1./(M+np.sqrt(M))*occurences #position probability matrix with pseudocounts
        PSSM = np.log2(PPM/q)
        # IC = -PPM*np.log(PPM)
        # print(IC)

        # columns = range(L)
        # rows = aa
        # index = np.arange(len(columns))
        # y_offset = np.zeros(len(columns))
        # plt.title("Logo Plot")
        # for row in range(q):
        #     plt.bar(index, IC[row], 1, bottom=y_offset)
        #     y_offset = y_offset +IC[row]
        # plt.show()



        #Writing the output
        PSSM = PSSM.astype("str")

        with open("../../database/pssm/"+filename+".pssm", 'w') as output:
            for k in range(q):
                output.write(";".join(PSSM[k])+"\n")




    






