import torch
from LSTMResNet import LSTMResNet
from Classification import classify_images

import os

# 載入模型
model_path = "Epoch12lstm_resnet_weights.pth"
model = LSTMResNet(num_classes=2, hidden_size=128, num_layers=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# image_path_list 是一個包含 34 張圖片路徑list,可以透過計數器的方式,每蒐集到34張frame 執行此Function
parent_directory = "C:/D/NCTU_CS/113_1/555008IntelligentApplicationsofDeepLearning/Final_Project/ML_Final_Project/NYCU_113_ML_Final_Project_LSTMResNet/images"
image_path_list = [os.path.join(parent_directory, file) for file in os.listdir(parent_directory) if os.path.isfile(os.path.join(parent_directory, file))]
image_path_list.remove(image_path_list[0])
result = classify_images(image_path_list,model,device,32)
print(result)
#刪除parent_directory的資料
