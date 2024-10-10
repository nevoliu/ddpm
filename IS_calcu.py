import numpy as np
import torch
from torchvision import models, transforms
from torch.nn.functional import softmax
from PIL import Image
import os 
from torchvision.models import Inception_V3_Weights


device = torch.device('cuda')

generated_images_folder = './compare/ddim'

files_path = sorted(os.listdir(generated_images_folder))

# 定义图像预处理步骤
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # 调整大小以适应 Inception v3
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载预训练的 Inception 模型
model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device)
model.eval()

def inception_score(images, batch_size=100, splits=10):
    # 存储每个图像的概率分布
    num_images = len(images)
    probs = np.zeros((num_images, 1000))

    # 按批次推理
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            print("i is : ",i)
            batch_images = images[i:i + batch_size]
            batch_tensors = torch.stack([transform(img) for img in batch_images]).to(device) 
            output = model(batch_tensors)
            batch_probs = softmax(output, dim=1).cpu().numpy()
            probs[i:i + batch_size] = batch_probs

    # 计算每个图像的 KL 散度
    split_scores = []
    for k in range(splits):
        part = probs[k * (len(images) // splits):(k + 1) * (len(images) // splits)]
        # 计算每部分的边际分布
        py = np.mean(part, axis=0)  # 平均值
        scores = np.array([np.sum(p * np.log(p / (py + 1e-10) + 1e-10)) for p in part]) 
        split_scores.append(np.exp(np.mean(scores)))  # 取指数

    return np.mean(split_scores), np.std(split_scores)

# 示例图像加载
images = [Image.open(os.path.join(generated_images_folder, path)) for path in files_path]  # 替换为您的图像路径

# 计算 Inception Score

mean_score, std_score = inception_score(images)
print("Inception Score: ", mean_score, "±", std_score)
