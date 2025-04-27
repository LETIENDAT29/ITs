import torch

# Tải mô hình đã huấn luyện
model = torch.load('runs/train/exp/weights/best.pt')

# In thông tin chi tiết về mô hình
print(model)
