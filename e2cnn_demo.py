from e2cnn import gspaces
from e2cnn import nn
import torch        
import torchvision.transforms.functional as tf      

r2_act = gspaces.Rot2dOnR2(N=4)

feat_type_in  = nn.FieldType(r2_act,  3*[r2_act.trivial_repr])
feat_type_out = nn.FieldType(r2_act, 10*[r2_act.regular_repr])
conv = nn.R2Conv(feat_type_in, feat_type_out, kernel_size=5)
relu = nn.ReLU(feat_type_out)
x = torch.randn(16, 3, 32, 32)
x1 = nn.GeometricTensor(x, feat_type_in)
x2 = nn.GeometricTensor(x, feat_type_in) # 90 degrees
yc1 = conv(x1)
yc2 = conv(x2)
y1 = relu(yc1)
y2 = relu(yc2)
print(y2)