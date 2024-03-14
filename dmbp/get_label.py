import os
import torch
from torchvision import models, transforms
from PIL import Image

with open('loader/imagenet_classes.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()] 

model = models.resnet50(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 图像文件夹路径
# image_folder = 'E:/学校材料/保研/科研/实验/FIA/dataset/images'
# image_folder = '/home/qqq/data/lty/FIA/adv/PTNAAPIDI/res-v2'
image_folder = '/home/qqq/data/lty/FIA/dataset/images'

# 分类结果保存文件路径
output_file = 'classification_results.txt'

# 遍历文件夹中的图像文件
with open(output_file, 'w') as f:
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            img = Image.open(image_path)
            img_tensor = preprocess(img)
            img_tensor = torch.unsqueeze(img_tensor, 0)

            # 使用模型进行预测
            with torch.no_grad():
                output = model(img_tensor)
                _, predicted_idx = torch.max(output, 1)
                label = labels[predicted_idx.item()].split(":")[0].strip()

            # 写入分类结果到文件
            f.write(f"{filename}: {label}\n")

print("Save success!")