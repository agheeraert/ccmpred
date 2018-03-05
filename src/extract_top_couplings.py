import torch
import argparse
import matplotlib.pylab as plt
import numpy as np

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Extraction of top couplings')

    parser.add_argument('-file', default='1BDO_A_Kp.out' , help='Name of the file')
    parser.add_argument('-a', default=1.5, type=float , help='Factor between the number of top couplings to extract and sequence size')

    args = parser.parse_args()

    Kp = torch.load('../results/'+args.file)
    L = Kp.size()[0]

    top_couplings = torch.topk(torch.triu(Kp, diagonal=1).view(-1), int(args.a*L))[1]

    top_couplings = ((top_couplings/L), torch.remainder(top_couplings, L))
    image = np.zeros((L, L))

    for i in range(L):
        x, y = (top_couplings[0][i]).data.cpu().numpy()[0], (top_couplings[1][i]).data.cpu().numpy()[0]
        image[x][y] = 1


    plt.title("Top couplings")
    plt.imshow(image, cmap = 'Greys')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

