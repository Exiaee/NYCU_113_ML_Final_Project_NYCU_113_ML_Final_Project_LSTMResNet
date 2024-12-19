import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from LSTMResNet import LSTMResNet
import os
import cv2

# 定義測試資料集
class TestDataset(Dataset):
    def __init__(self, image_sequences, transform=None):
        self.image_sequences = image_sequences
        self.transform = transform

    def __len__(self):
        return len(self.image_sequences)

    def __getitem__(self, idx):
        sequence = self.image_sequences[idx]
        processed_sequence = []
        for image_path in sequence:
            image = image_path.convert('RGB')
            #image=cv2.cvtColor(image_path,cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)
            processed_sequence.append(image)
        return torch.stack(processed_sequence)

def classify_images(image_path_list, model,device,batch_size):
    # 資料轉換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # 整理輸入資料
    test_image_sequences = [image_path_list]  # 每張圖片變成一個單獨序列
    print("AU")
    print(image_path_list)
    # 建立資料集與 DataLoader
    test_dataset = TestDataset(test_image_sequences, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    
    class_labels = {0: "Bad", 1: "Good"}
    results = []

    # 開始測試
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            outputs = model(batch)
            for seq_idx, scores in enumerate(outputs):
                logits = scores.cpu().numpy()
                probabilities = torch.nn.functional.softmax(torch.tensor(logits), dim=0)
                probabilities_numpy = probabilities.detach().cpu().numpy()

                result = {
                    "predicted": class_labels[np.argmax(probabilities_numpy)],
                    "probabilities": probabilities_numpy.tolist()
                }

                if probabilities_numpy[0] > probabilities_numpy[1]:
                    if probabilities_numpy[0] > 0.7:
                        result["predicted"] = "Bad"
                    else:
                        result["predicted"] = "Soso"
                else:
                    if probabilities_numpy[1] > 0.7:
                        result["predicted"] = "Good"
                    else:
                        result["predicted"] = "Soso"

                results.append(result)

    return results[0]["predicted"]



    