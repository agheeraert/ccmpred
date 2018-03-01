import torch
from torch.autograd import Variable
from torch.nn import Embedding
import numpy as np
import convert_msa
# x = Variable(torch.FloatTensor(1).fill_(2.0), requires_grad=True)
# y = Variable(torch.FloatTensor(1).fill_(-1.0))

# x_prime = torch.FloatTensor(1)
# x_prime[0] = float(x)

# # z = x**2 - y
# z_prime = Variable(x_prime**2, requires_grad=True) - y

# # z.backward()
# # print(x.grad)

# z_prime.backward()
# print(x.grad)


# x = Variable(torch.FloatTensor(4,4).fill_(0.5), requires_grad=True)

# y = Variable(torch.FloatTensor(4,4).fill_(1.0), requires_grad=False)
# for i in range(0,4): y.data[i,i]=0


# v = Variable(torch.FloatTensor(4).fill_(2.0))
# x1 = x*y
# z = torch.sum(torch.mv(x1,v))
# print (z)
# z.backward()
# print (x.grad)


with open("../database/1atzA.aln", "r") as msa_file:
    msa = []
    for line in msa_file.readlines():
        msa.append(list(line))
        
msa_np = np.asarray(convert_msa.to_numbers(msa), np.int32)
msa = Variable(torch.from_numpy(msa_np).long())

B, N = msa.size()[0], msa.size()[1]
embed = Embedding(B*N, 21)
x = embed(msa)





