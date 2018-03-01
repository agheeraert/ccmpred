

import torch
from torch.autograd import Variable

from Bio.PDB.Polypeptide import aa1

def getQ(seq, i, aa_to_idx):
    return torch.LongTensor([aa_to_idx[seq[i]]])

def loss(msa, H, J):
    
    aa = list(aa1 + '-')
    aa_to_idx = {}
    for i, a in enumerate(aa):
        aa_to_idx[a] = i
    
    B = len(msa)
    L = len(msa[0]) #sequence length
    q = len(aa) #number of aa
    lpseudo = 0
    all_aa = Variable(torch.LongTensor([i for i in range(0, q)]))
    

    for b in range(B):
        all_aa_si = torch.LongTensor(q,L)
        for i in range(0,q):
            for j, aa in enumerate(msa[b]):
                all_aa_si[i,j] = i*q + aa_to_idx[aa]
        all_aa_si = Variable(all_aa_si)
        
        for r in range(L):
            s_r = Variable(getQ(msa[b], r, aa_to_idx))
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

            #Computing nominator
            J_rij = J(s_i).resize(L, L, L)[:,r,:]
            nominator = torch.exp(H(s_r)[0,r] + (J_rij*mask).sum())
            
            del mask
            del s_i
            del J_rij
            del s_r
            #Computing denominator
            J_rili = J(all_aa_si).resize(q,L,L,L)
            J_ili = J_rili[:,:,r,:]
            J_l = (J_ili*mask_extended).sum(dim=1).sum(dim=1)
            denominator = torch.exp(H(all_aa)[:,r] + J_l).sum()
            
            del mask_extended
            del J_rili
            del J_ili
            del J_l
    
            #neg log likelihood
            lpseudo += -torch.log(nominator) + torch.log(denominator)
            del nominator
            del denominator
        del all_aa_si
    return lpseudo
        

    
    