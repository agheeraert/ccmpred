import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

Jload = torch.load('../results/1atzA_J.out')
for Jij in Jload.parameters():
    J = Jij
print(J)
J = J.view(21, 21, 80, 80)
J = J - torch.mean(J, dim=0) - torch.mean(J, dim=1) + torch.mean(torch.mean(J, dim=0), dim=0)
S_FN = J*J
S_FN = torch.sqrt(torch.sum(torch.sum(S_FN, dim=0), dim=0))

S_CN = S_FN - torch.addcmul(Variable(torch.zeros(80, 80)), torch.mean(S_FN, dim=0), torch.mean(S_FN, dim=1)) / torch.mean(S_FN)

cmat = S_CN.data.numpy()

plt.title("This algorithm result")
plt.imshow(cmat)
plt.colorbar()
plt.show()