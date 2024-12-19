# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 23:55:32 2024

@author: USER
"""
import copy
import cv2
import numpy as np
import torch
import itertools
from PIL import Image
from LSTMResNet import LSTMResNet
from Classification import classify_images
model_path = "Epoch12lstm_resnet_weights.pth"
model = LSTMResNet(num_classes=2, hidden_size=128, num_layers=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

cap = cv2.VideoCapture(0)

# 設定擷取影像的尺寸大小
cap_width = 1920
cap_height = 1080
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height )

# 使用 XVID 編碼
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# 建立 VideoWriter 物件，輸出影片至 output.avi
# FPS 值為 20.0，解析度為 640x360

temp=list()
result=['']
while(cap.isOpened()):
  ret, frame = cap.read()

  if ret == True:
    # 寫入影格
   #out.write(frame)
    debug_image = copy.deepcopy(frame)

    debug_image = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
    #temp.append(debug_image.asarray
    debug_image.flags.writeable=True
    
    debug=Image.fromarray(debug_image)
    temp.append(debug)
# 將 NumPy 陣列轉換為 PIL Image
    img=cv2.flip(frame, 1)
    
    cv2.putText(img, "Prediction: "+ result[-1], (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
               cv2.LINE_AA)
  
    cv2.imshow('frame',img)
    #debug_image = debug_image.fromarray(cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)).convert('RGB')
    if len(temp)>40:
        result.append(classify_images(temp,model,device,32))
        print(result)
       
        temp.clear()
    #cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  else:
    break

# 釋放所有資源
cap.release()

cv2.destroyAllWindows()