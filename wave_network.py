import torch
from e2cnn import gspaces
from e2cnn import nn

import matplotlib.pyplot as plt
import numpy as np

# Act upon vector magnitude not direction: nn.NormNonLinearity...
# What is W? Just a bit of extra notation? Still same conv defined by legend
# GNormBatchNorm...
# Skip connections
# Bilinear interpolation

class WaveNet(nn.EquivariantModule):
    def __init__(self):
        super(WaveNet, self).__init__()
        r2_act = gspaces.Rot2dOnR2(N=4) # Cyclic group of order 4

        feat_type_inA2 = nn.FieldType(r2_act, [r2_act.irrep(1)] + 2*[r2_act.trivial_repr]) # 4 channels
        feat_type_hid8 = nn.FieldType(r2_act, 8*[r2_act.regular_repr])
        feat_type_hid16 = nn.FieldType(r2_act, 16*[r2_act.regular_repr])
        feat_type_hid32 = nn.FieldType(r2_act, 32*[r2_act.regular_repr])
        feat_type_out = nn.FieldType(r2_act, [r2_act.irrep(1) + r2_act.trivial_repr])

        self.input_type = feat_type_inA2

        ## A[2]
        self.a2 = nn.SequentialModule(
            nn.R2Conv(feat_type_inA2, feat_type_hid8, kernel_size=3, padding=1),
            nn.GNormBatchNorm(feat_type_hid8),
            nn.NormNonLinearity(feat_type_hid8, 'n_relu'),
            nn.R2Conv(feat_type_hid8, feat_type_hid8, kernel_size=3, padding=1),
            nn.GNormBatchNorm(feat_type_hid8),
            nn.NormNonLinearity(feat_type_hid8, 'n_relu'), 
        )
        ## SKIP CONNECTION A[2]
        
        ## A[1]
        self.a1 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid8, feat_type_hid8, kernel_size=3, padding=1, stride=2),
            nn.R2Conv(feat_type_hid8, feat_type_hid16, kernel_size=3, padding=1),
            nn.GNormBatchNorm(feat_type_hid16),
            nn.NormNonLinearity(feat_type_hid16, 'n_relu'),
            nn.R2Conv(feat_type_hid16, feat_type_hid16, kernel_size=3, padding=1),
            nn.GNormBatchNorm(feat_type_hid16),
            nn.NormNonLinearity(feat_type_hid16, 'n_relu'),
        )
         ## SKIP CONNECTION A[1]

        ## A[0]
        self.a0 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid16, feat_type_hid16, kernel_size=3, padding=1, stride=2),
            nn.R2Conv(feat_type_hid16, feat_type_hid32, kernel_size=3, padding=1),
            nn.GNormBatchNorm(feat_type_hid32),
            nn.NormNonLinearity(feat_type_hid32, 'n_relu'),
            nn.R2Conv(feat_type_hid32, feat_type_hid32, kernel_size=3, padding=1),
            nn.GNormBatchNorm(feat_type_hid32),
            nn.NormNonLinearity(feat_type_hid32, 'n_relu'), 
        )
        
        self.up16x32 = nn.R2Upsampling(feat_type_hid32, 2)

        ## B[1]
        self.b1 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid32, feat_type_hid16, kernel_size=3, padding=1),
            nn.GNormBatchNorm(feat_type_hid16),
            nn.NormNonLinearity(feat_type_hid16, 'n_relu'),
            nn.R2Conv(feat_type_hid16, feat_type_hid16, kernel_size=3, padding=1),
            nn.GNormBatchNorm(feat_type_hid16),
            nn.NormNonLinearity(feat_type_hid16, 'n_relu'),
        )
        ## SKIP CONNECTION ADD A[1]
        
        self.up32x64 = nn.R2Upsampling(feat_type_hid16, 2)

        ## B[2]
        self.b2 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid16, feat_type_hid8, kernel_size=3, padding=1),
            nn.GNormBatchNorm(feat_type_hid8),
            nn.NormNonLinearity(feat_type_hid8, 'n_relu'),
            nn.R2Conv(feat_type_hid8, feat_type_hid8, kernel_size=3, padding=1),
            nn.GNormBatchNorm(feat_type_hid8),
            nn.NormNonLinearity(feat_type_hid8, 'n_relu'),
        )
        ## SKIP CONNECTION ADD A[2]

        self.up64x128 = nn.R2Upsampling(feat_type_hid8, 2)

        ## B[3]
        self.b3 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid8, feat_type_hid8, kernel_size=3, padding=1),
            nn.GNormBatchNorm(feat_type_hid8),
            nn.NormNonLinearity(feat_type_hid8, 'n_relu'),
            nn.R2Conv(feat_type_hid8, feat_type_hid8, kernel_size=3, padding=1),
            nn.GNormBatchNorm(feat_type_hid8),
            nn.NormNonLinearity(feat_type_hid8, 'n_relu'),
        )
        
        ## W
        self.W = nn.SequentialModule(
            nn.R2Conv(feat_type_hid8, feat_type_out, kernel_size=3, padding=1),
            nn.GNormBatchNorm(feat_type_out),
            nn.NormNonLinearity(feat_type_out, 'n_relu'),
        )

    def forward(self, x: torch.Tensor):
        x = nn.GeometricTensor(x, self.input_type)
        ska2 = self.a2(x) ## SKIP CONNECTION A[2]
        ska1 = self.a1(ska2) ## SKIP CONNECTION A[1]
        x = self.a0(ska1)
        x = self.up16x32(x) ## Bilinear interpolate x from 16x16 to 32x32
        x = self.b1(x)
        x = x + ska1 ## SKIP CONNECTION ADD A[1]
        x = self.up32x64(x) ## Bilinear interpolate x from 32x32 to 64x64
        x = self.b2(x)
        x = x + ska2 ## SKIP CONNECTION ADD A[2]
        x = self.up64x128(x) ## Bilinear interpolate x from 64x64 to 128x128
        x = self.b3(x)
        x = self.W(x)
        return x

    def evaluate_output_shape(self):
        return torch.ones(1, 3, 128, 128).shape

