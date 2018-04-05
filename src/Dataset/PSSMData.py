import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

from Bio.PDB.Polypeptide import aa1


class PSSMData(Dataset):
    """
    The dataset that loads 
    1. FloatTensor of PSSM
    2. FloatTensor of 1-hot encoding
    3. Mask
    4. target name
    """
    def __init__(self, filename):
        self.filename = filename
        self.q = len(aa1)

        self.aa_to_one_hot = {}
        for i, a in enumerate(aa1):
            one_hot = torch.zeros(self.q)
            one_hot[i] = 1
            self.aa_to_one_hot[a] = one_hot

        with open(self.filename, "r") as data:
            self.pssm_list, self.one_hot_list, self.mask_list, self.name_list = [], [], [], []
            lines = data.readlines()
            for num, line in enumerate(lines):
                if "[ID]" in line:
                    self.name_list.append(lines[num+1][:-1])
                    self.one_hot_list.append(list(lines[num+3][:-1]))
                    pssm = np.zeros([self.q, len(lines[num+5].split("\t")[:-1])])
                    for k in range(self.q):
                        pssm[k]=lines[num+5+k].split("\t")[:-1]
                    self.pssm_list.append(pssm)
                    self.mask_list.append(list(lines[num+31][:-1]))

        for i, pssm in enumerate(self.pssm_list):
            self.pssm_list[i] = torch.from_numpy(pssm).float()

        for i, primary in enumerate(self.one_hot_list):
            for j, aa in enumerate(primary):
                primary[j] = self.aa_to_one_hot[aa].unsqueeze(dim=1)
            self.one_hot_list[i] = torch.cat(primary,dim=1)
        
        self.dataset_size = len(self.pssm_list)

        # for one_hot in self.one_hot_list:
        #     print(torch.sum(one_hot, dim=0))
        # for pssm in self.pssm_list:
        #     print(torch.sum(pssm, dim=0))
        
        def __getitem__(self):
            """
            """
            return self.pssm_list, self.one_hot_list, self.mask_list, self.name_list

        def __len__(self):
            """
            returns the size of the dataset
            """
            return self.dataset_size
        
print(PSSMData("../../scripts/Dataset/alquraishi/testing"))
