import torch
from e2cnn import gspaces
from e2cnn import nn

import matplotlib.pyplot as plt
import numpy as np

r2_act = gspaces.Rot2dOnR2(N=4) # Cyclic group of order 4

feat_type_in = nn.FieldType(r2_act, [r2_act.irrep(1)])
feat_type_hid = nn.FieldType(r2_act, 8*[r2_act.regular_repr])
feat_type_out = nn.FieldType(r2_act, [r2_act.irrep(1)])

model = nn.SequentialModule(
    nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=3),
    nn.NormNonLinearity(feat_type_hid, 'n_relu'),
    nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=3),
    nn.NormNonLinearity(feat_type_hid, 'n_relu'), # Act upon vector magnitude not direction
    nn.R2Conv(feat_type_hid, feat_type_out, kernel_size=3),
).eval()

S = 11
x = torch.randn(1, 2, S, S)
x = nn.GeometricTensor(x, feat_type_in)

fig, axs = plt.subplots(1, r2_act.fibergroup.order(), sharex=True, sharey=True, figsize=(16, 3))
fig2, axs2 = plt.subplots(1, r2_act.fibergroup.order(), sharex=True, sharey=True, figsize=(16, 3))

X, Y = np.meshgrid(range(S-6), range(S-7, -1, -1))
X2, Y2 = np.meshgrid(range(11), range(11-1, -1, -1))

# for each group element
for i, g in enumerate(r2_act.testing_elements):
    # transform the input
    x_transformed = x.transform(g)
    
    y = model(x_transformed)
    y = y.tensor.detach().numpy().squeeze()
    # plot the output vector field
    axs[i].quiver(X, Y, y[0, ...], y[1, ...], units='xy')
    axs[i].set_title(g*90)
    
for i, g in enumerate(r2_act.testing_elements):
    # transform the input
    x_transformed = x.transform(g)
    
    y = x_transformed.tensor.detach().numpy().squeeze()
    # plot the output vector field
    axs2[i].quiver(X2, Y2, y[0, ...], y[1, ...], units='xy')
    axs2[i].set_title(g*90)

plt.show()
