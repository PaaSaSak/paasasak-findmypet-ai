import torch, timm, torch.nn as nn

# 모델 구조 동일하게 정의
backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
head  = nn.Sequential(nn.Flatten(), nn.Linear(backbone.num_features, 512, bias=False), nn.BatchNorm1d(512))
net   = nn.Sequential(backbone, head)
state = torch.load('outputs/embed_b0_triplet.pt', map_location='cpu')
net.load_state_dict(state, strict=False)
net.eval()

dummy = torch.randn(1,3,224,224)
torch.onnx.export(net, dummy, "serving/embed.onnx", opset_version=17,
                  input_names=["input"], output_names=["emb"])
print("exported to serving/embed.onnx")
