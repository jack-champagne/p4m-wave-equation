import matplotlib.pyplot as plt
import numpy as np
from sympy import interpolate

import torch

from e2cnn import gspaces
from e2cnn import nn
import numpy as np

r2_act = gspaces.Rot2dOnR2(N=4)

feat_type_in = nn.FieldType(r2_act, [r2_act.trivial_repr])
feat_type_hid = nn.FieldType(r2_act, 8*[r2_act.regular_repr])
feat_type_out = nn.FieldType(r2_act, [r2_act.irrep(1)])

model = nn.SequentialModule(
    nn.R2Conv(feat_type_in, feat_type_hid, kernel_size=3),
    nn.InnerBatchNorm(feat_type_hid),
    nn.ReLU(feat_type_hid),
    nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=3),
    nn.InnerBatchNorm(feat_type_hid),
    nn.ReLU(feat_type_hid),
    nn.R2Conv(feat_type_hid, feat_type_out, kernel_size=3),
).eval()

S = 11
x = torch.randn(1, 1, S, S)
x = nn.GeometricTensor(x, feat_type_in)

fig, axs = plt.subplots(1, r2_act.fibergroup.order(), sharex=True, sharey=True, figsize=(16, 3))
fig2, axs2 = plt.subplots(1, r2_act.fibergroup.order(), sharex=True, sharey=True, figsize=(16, 3))
X, Y = np.meshgrid(range(S-6), range(S-7, -1, -1))

# for each group element
for i, g in enumerate(r2_act.testing_elements):
    # transform the input
    x_transformed = x.transform(g)
    
    y = model(x_transformed)
    y = y.tensor.detach().numpy().squeeze()
    
    # plot the output vector field
    axs[i].quiver(X, Y, y[0, ...], y[1, ...], units='xy')
    axs[i].set_title(g*90)
    axs2[i].imshow(x_transformed.tensor.reshape(S, S), cmap='hot', interpolation='nearest')
plt.show()