from archs.ghostnet import ghostnet
import torch

model = ghostnet(num_classes=1000, width=1.0)
model.eval()

x = torch.randn(2, 3, 224, 224)
y = model(x)

print(y.shape)