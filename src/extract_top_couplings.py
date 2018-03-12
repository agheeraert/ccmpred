import torch
from torch.autograd import Variable
import argparse
import matplotlib.pylab as plt
import numpy as np
from Bio.PDB import *


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extraction of top couplings')

    parser.add_argument('-file', default='1BDO_A_Kp.out' , help='Name of the file')
    parser.add_argument('-a', default=1.5, type=float , help='Factor between the number of top couplings to extract and sequence size')
    parser.add_argument('-range', default='short', help='Define the range in which perfomance is evaluated')

    args = parser.parse_args()

    Kp = torch.load('../results/'+args.file).cpu()
    
    # Kp = []
    # with open('../database/1BDO_A.mat', 'r') as fin:
    #     for line in fin:
    #         line = line.replace(',', '.')
    #         vec = []
    #         for dig in line.split('\t')[:-1]:
    #             vec.append(float(dig))
    #         Kp.append(vec)
    # Kp = np.array(Kp)
    # Kp = torch.from_numpy(Kp) 
    # Kp = Variable(Kp)   
    
    L = Kp.size()[0]

    if args.range == 'short':
        mask = torch.triu(torch.ones(L, L), diagonal = 6) - torch.triu(torch.ones(L, L), diagonal = 12)
    if args.range == 'medium':
        mask = torch.triu(torch.ones(L, L), diagonal = 12) - torch.triu(torch.ones(L, L), diagonal = 24)
    if args.range == "large":
        mask = torch.triu(torch.ones(L, L), diagonal = 24)
    mask = Variable(mask)
    Kp = Kp*mask


    top_couplings = torch.topk(Kp.view(-1), int(args.a*L))[1]

    top_couplings = ((top_couplings/L), torch.remainder(top_couplings, L))
    k = top_couplings[0].size()[0]
    torch.save(top_couplings,"../results/1BDO_A_top_coupl.out")
    
    # Calculating the distances
    structure = PDBParser().get_structure('1BDO_A', '../database/1BDO_A.pdb')
    model = structure[0]
    L = len(list(structure.get_residues()))
    distances = np.zeros((L, L))

    for chain in model:
         for i in range(L):
            for j in range(L):
                distances[i][j] = chain[i+77]['CA'] - chain[j+77]['CA']

    #Renormalizing to plot the contact map
    cmap = (-1*(distances - np.max(distances)))/np.max(distances)
    f = plt.figure()
    plt.title("Contact map of the real structure")
    plt.imshow(cmap)
    plt.colorbar()
    plt.savefig('../results/real.png')



    image = np.zeros((L, L))
    false_predictions_x = []
    false_predictions_y = []
    for i in range(k):
        x, y = (top_couplings[0][i]).data.cpu().numpy()[0], (top_couplings[1][i]).data.cpu().numpy()[0]
        image[x][y] = 1
        if distances[x][y] > 12:
            false_predictions_x.append(x)
            false_predictions_y.append(y)

    false_predictions_x, false_predictions_y = np.asarray(false_predictions_x), np.asarray(false_predictions_y)

    f = plt.figure()
    plt.title("Top couplings")
    plt.imshow(image, cmap = 'Greys')
    plt.scatter(false_predictions_y, false_predictions_x, c = 'red', marker = 'x')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('../results/false_predictions.png')

    print(1-len(false_predictions_x)/k)

