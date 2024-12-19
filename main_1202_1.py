import csv
import os
import sys
import copy
import itertools
import tkinter as tk
import tkinter.messagebox
import math
import requests
import cv2 as cv
import numpy as np
import mediapipe as mp
from model import KeyPointClassifier
import time
from tkinter import filedialog
import torch
from LSTMResNet import LSTMResNet
from Classification import classify_images

import os


# Timer starts
starttime = time.time()
lapnum = 1

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def line_notify(msg,_token):
    __token = _token  # 填入你的token
    url = 'https://notify-api.line.me/api/notify'
    headers = {
        'Authorization': 'Bearer ' + __token
    }
    data = {
        'message': msg
    }
    requests.post(url, headers=headers, data=data)



def timer(t, s=starttime): #s:上次執行時間
    tempt=round((time.time() - s), 4) 
    print(tempt, " ", s )
    if int(tempt)>=t: #>=t秒
        s=time.time() #重設timer
        return True, s
    else:
        return False, s #繼續計數
    
    
# 根據兩點的座標，計算角度
def vector_2d_angle(v1, v2):
    v1_x = v1[0]
    v1_y = v1[1]
    v2_x = v2[0]
    v2_y = v2[1]
    try:
        angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
    except:
        angle_ = 180
    return angle_

# 根據傳入的 21 個節點座標，得到該手指的角度
def hand_angle(hand_):
    angle_list = []
    # thumb 大拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
        ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
        )
    angle_list.append(angle_)
    # index 食指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
        ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
        )
    angle_list.append(angle_)
    # middle 中指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
        ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
        )
    angle_list.append(angle_)
    # ring 無名指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
        ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
        )
    angle_list.append(angle_)
    # pink 小拇指角度
    angle_ = vector_2d_angle(
        ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
        ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
        )
    angle_list.append(angle_)
    return angle_list

# 根據手指角度的串列內容，返回對應的手勢名稱
def hand_pos(finger_angle):
    f1 = finger_angle[0]   # 大拇指角度
    f2 = finger_angle[1]   # 食指角度
    f3 = finger_angle[2]   # 中指角度
    f4 = finger_angle[3]   # 無名指角度
    f5 = finger_angle[4]   # 小拇指角度
    print(f1,f2,f3,f4,f5)
    # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
    if f1<50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
        return 'Good'
    elif f1>=50 and f2>=50 and f3<85 and f4>=50 and f5>=50:
        ###return 'FUCK!'
        return 'FXXK!'
    elif f1<=50 and f2>=50 and f3<85 and f4>=50 and f5<=50:
        ### return 'FUCK YOU!!!'
        return 'FXXK YOU!!!'
    elif f1<50 and f2<50 and f3>=50 and f4>=50 and f5<50:
        return 'ROCK!'
    elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
        return 'Fist'
    elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5<50:
        return 'pink'
    elif f1>=50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
        return 'Number 1'
    elif f1>=50 and f2<50 and f3<50 and f4>=50 and f5>=50:
        return 'YA!!!'
    elif f1>=50 and f2>=50 and f3<50 and f4<50 and f5<50:
        return 'ok'
    elif f1<50 and f2>=50 and f3<70 and f4<50 and f5<50:
        return 'ok'
    elif f1>=50 and f2<50 and f3<70 and f4<50 and f5>50:
        return '3'
    elif f1>=50 and f2<50 and f3<50 and f4<50 and f5<50:
        return '4'
    elif f1<50 and f2<50 and f3<50 and f4<50 and f5<50:
        return 'NO! NO!'
    elif f1<50 and f2>=50 and f3>=85 and f4>=50 and f5<50:
        return '666!'
    elif f1<50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
        return 'Bang'
    elif f1<50 and f2<50 and f3<50 and f4>=50 and f5>=50:
        return 'Bang Bang!!!'
    elif f1<50 and f2<50 and f3<50 and f4<50 and f5>=50:
        return '9'
    elif f1>50 and f2>50 and f3<50 and f4<50 and f5>=50:
        return 'OK'
    else:
        return ''

def mosaic(img, x_max, x_min, y_max,y_min,w,h):
    if x_max > w: x_max = w      # 如果最大值超過邊界，將最大值等於邊界
    if y_max > h: y_max = h      # 如果最大值超過邊界，將最大值等於邊界
    if x_min < 0: x_min = 10      # 如果最小值超過邊界，將最小值等於邊界
    if y_min < 0: y_min = 10      # 如果最小值超過邊界，將最小值等於邊界
    x_max = int(x_max)
    y_max = int(y_max)
    x_min = int(x_min)        
    y_min = int(y_min)
    mosaic_w = x_max - x_min     # 計算四邊形的寬
    mosaic_h = y_max - y_min     # 計算四邊形的高
    mosaic = img[y_min:y_max,x_min:x_max]
    mosaic = cv.resize(mosaic, (8,8), interpolation=cv.INTER_LINEAR)  # 根據縮小尺寸縮小
    mosaic = cv.resize(mosaic, (mosaic_w,mosaic_h), interpolation=cv.INTER_NEAREST) # 放大到原本的大小
    img[y_min:y_max,x_min:x_max]  = mosaic    # 馬賽克區域
    return img

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_info_text(image, brect, facial_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if facial_text != "":
        info_text = 'Emotion :' + facial_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

root=tk.Tk()

root.geometry("800x240")
root.title('NYCU ML 11thGroup Final Report')
root.configure(bg="#7AFEC6")
#root.geometry('300x300')

temp_name = tk.StringVar()
temp_token = tk.StringVar()


temp_name.set("")
def Info():
    entry_name = En.get()
    entry_token=En2.get()
    entry_period=En3.get()
    _name=str(entry_name)
    _token=str(entry_token)
    _name=_name.strip()
    #print(name)
    #print(len(name))
    if len(_name)==0 or En1.get()!='5487':
        tkinter.messagebox.showinfo("提示", "帳號不能為空或是密碼非5487")
        En.delete(0, 'end')
        En1.delete(0, 'end')
    else :#用get獲得帳密去判斷能不能登入
        global period
        global token
        print(temp_token)
        token=str(_token)
        print(_token)
        global name
        name=str(_name)
        period=int(entry_period)
        if period<=0:
            period=1
        print(name)
        root.destroy()
def Q():
    os._exit(0)
    
def JieShu():
    tkinter.messagebox.showwarning(title='Warning', message='You click the x button.')
    os._exit(0)




'''def radiobutton_event(widget):
    print('radiobutton_event: ' + val.get() + ' checked')
    global Option
    Option=''
    if val.get()=='':
        Option='Send Line Message Periodically'
        #Option = widget['text']
    Option=str(val.get())
    print (Option)
    
def radiobutton_event_2(widget2):
    print('radiobutton_event: ' + val2.get() + ' checked')
    global mos
    mos=''
    print(mos)
    mos = widget2['text']'''
    
Option=["Send Line Message Periodically"]
mos=["Show Your Face"]
videofile_option=["WebCam"]
videofile_path=[]

def radiobutton_event(widget):
    print('radiobutton_event: ' + val.get() + ' checked')
    Option.append(widget['text'])
    print (Option)
    
def radiobutton_event_2(widget2):
    print('radiobutton_event: ' + val2.get() + ' checked')
    mos.append(widget2['text'])
    print(mos)

def radiobutton_event_3(widget3):
    print('radiobutton_event: ' + val_path.get() + ' checked')
    videofile_option.append(widget3['text'])
    print(videofile_option)    

def show():
    file_path = filedialog.askopenfilename()   # 選擇檔案後回傳檔案路徑與名稱
    videofile_path.append((str(file_path)))
    print(videofile_path[-1])
    print(file_path)
    (filepath, tempfilename) = os.path.split(file_path)
    short_filepath=filepath[0:3]+"..."+tempfilename     
    val_path.set(short_filepath)

label=tk.Label(root,text='Account',bg='#DDA0DD',fg="#8B008B",
            font=("Algerian",12,"bold"),anchor='c')
label.grid(row=0,sticky=tk.W)
En=tk.Entry(root,width=50,textvariable=temp_name)
En.grid(row=0,column=1,sticky=tk.W)
label1=tk.Label(root,text='Password',bg='#DDA0DD',fg="#8B008B",
            font=("Algerian",12,"bold"),anchor='c')
label1.grid(row=1,sticky=tk.W)
#En1=tk.Entry(root,show='*')#隱藏密碼
En1=tk.Entry(root,width=50)#隱藏密碼
En1.grid(row=1,column=1,sticky=tk.W)

label=tk.Label(root,text='Line Token',bg='#DDA0DD',fg="#8B008B",
            font=("Algerian",12,"bold"),anchor='c')
label.grid(row=2,sticky=tk.W)
En2=tk.Entry(root,show='*',width=50)
En2.grid(row=2,column=1,sticky=tk.W)

label=tk.Label(root,text='Time Period (s)',bg='#DDA0DD',fg="#8B008B",
            font=("Algerian",12,"bold"),anchor='c')
label.grid(row=3,sticky=tk.W)
En3=tk.Entry(root,width=50)

En3.grid(row=3,column=1,sticky=tk.W)

val = tk.StringVar()
val2 = tk.StringVar()
val_path = tk.StringVar() 
radio_btn1 = tk.Radiobutton(root, text='Send Line Message Periodically',variable=val, value='Send Line Message Periodically',
                            command=lambda: radiobutton_event(radio_btn1))
radio_btn1.grid(row=4,column=0,sticky=tk.W)
radio_btn2 = tk.Radiobutton(root, text='Only Once After Quitting, Except Middle Finger',variable=val, value='Only Once After Quitting, Except Middle Finger',
                            command=lambda: radiobutton_event(radio_btn2))
radio_btn2.grid(row=4,column=1,sticky=tk.E)
#radio_btn1.select()
#radio_btn2.deselect()
val.set('Send Line Message Periodically')

radio_btn3 = tk.Radiobutton(root, text='Show Your Face',variable=val2, value='Show Your Face',
                            command=lambda: radiobutton_event_2(radio_btn3))
radio_btn3.grid(row=5,column=0,sticky=tk.W)
radio_btn4 = tk.Radiobutton(root, text='Mosaic Your Face',variable=val2, value='Mosaic Your Face',
                            command=lambda: radiobutton_event_2(radio_btn4))
radio_btn4.grid(row=5,column=1)
val2.set('Show Your Face')

radio_btn5 = tk.Radiobutton(root, text='Open Video File', value='Open Video File',
                            command=lambda: radiobutton_event_3(radio_btn5))
radio_btn5.grid(row=6,column=0,sticky=tk.W)

radio_btn6 = tk.Radiobutton(root, text='WebCam', value='WebCam',
                            command=lambda: radiobutton_event_3(radio_btn6))
radio_btn6.grid(row=6,column=1)
radio_btn5.deselect()
radio_btn6.select()
val_path.set('                                ')


btn = tk.Button(root,
                text='開啟檔案',
                font=('Arial',12,'bold'),
                command=show
              )
btn.grid(row=6,column=0,sticky=tk.E)
mylabel = tk.Label(root, text='file path:', font=('Arial',10))  # 放入標籤，使用 textvariable=text       
mylabel.grid(row=7,column=0,sticky=tk.W)
mylabel_path = tk.Label(root, textvariable=val_path, font=('Arial',10))  # 放入標籤，使用 textvariable=text       
mylabel_path.grid(row=7,column=0,sticky=tk.E)

b=tk.Button(root,text='Exit',anchor='c',width=6,height=1,command=Q)#quit可以讓pyhon shell結束
b.grid(row=10,column=0,sticky=tk.W)
b1=tk.Button(root,text='Login',anchor='c',width=6,height=1,command=Info)
b1.grid(row=10,column=1)
root.protocol("WM_DELETE_WINDOW", JieShu)
root.mainloop()

cap_device=0
model_path = "Epoch12lstm_resnet_weights.pth"
model = LSTMResNet(num_classes=2, hidden_size=128, num_layers=1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

if(videofile_option[-1]=='WebCam'):
    cap_device=0
else:
    cap_device=videofile_path[-1]

cap_width = 1920
cap_height = 1080
use_brect = True

# Camera preparation
cap = cv.VideoCapture(cap_device)
if not cap.isOpened():
    tkinter.messagebox.showwarning(title='Error', message='Cannot open file or WebCam.')
    os._exit(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
fourcc = cv.VideoWriter_fourcc(*'MP4V')
out = cv.VideoWriter('output01.mp4', fourcc, 20.0, (cap_width,  cap_height))

# Model load
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) 

keypoint_classifier = KeyPointClassifier()

fontFace = cv.FONT_HERSHEY_SIMPLEX  # 印出文字的字型
lineType = cv.LINE_AA               # 印出文字的邊框

# Read labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv',
            encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

mode = 0
Grade = 0
bad_guy = 0
ftime = 0
total_flag=0
pos_flag=0
neg_flag=0
play = 1
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while True:

        # Camera capture

        ret, image = cap.read()
    
        if not ret:
            break
    
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
    
        # Detection implementation
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True

        size = image.shape
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, face_landmarks)
                print(brect)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, face_landmarks)
    
                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
    
                #emotion classification
                facial_emotion_id,percent = keypoint_classifier(pre_processed_landmark_list)
                # Drawing part
                percent=round(percent, 3)
                
                
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(
                        debug_image,
                        brect,
                        keypoint_classifier_labels[facial_emotion_id]+", "+str(percent))
               
                if mos[-1]=="Mosaic Your Face":
                    face_location=list()
                    face_location.clear()
                    for temp in brect:
                        face_location.append(int(temp))
                    w=cap_width
                    h=cap_height
                    x_max = face_location[2]              # 取出 x 座標最大值
                    y_max = face_location[3]              # 取出 y 座標最大值
                    x_min = face_location[0]        # 取出 x 座標最小值
                    y_min = face_location[1]         # 取出 y 座標最小值
                    mosaic(debug_image,x_max,x_min,y_max,y_min,w,h)
                
                if keypoint_classifier_labels[facial_emotion_id]=="Happy":
                    Grade+=1
                    total_flag+=1
                elif keypoint_classifier_labels[facial_emotion_id]=="Sad" or keypoint_classifier_labels[facial_emotion_id]=="Angry":
                    Grade-=1
                    total_flag-=1
                #print(keypoint_classifier_labels[facial_emotion_id])
        result_hand = hands.process(image)
        #lbl = result_hand.multi_handedness[0].classification[0].label
       
        if result_hand.multi_hand_landmarks is not None:
            for hand_landmarks in result_hand.multi_hand_landmarks:
                lbl = result_hand.multi_handedness[0].classification[0].label
                finger_points = []                   # 記錄手指節點座標的串列
                fx = []                              # 記錄所有 x 座標的串列
                fy = [] 
                for i in hand_landmarks.landmark:
                    # 將 21 個節點換算成座標，記錄到 finger_points
                    x = i.x*size[1]
                    y = i.y*size[0]
                    finger_points.append((x,y))
                    fx.append(x)
                    fy.append(y)
                if finger_points:
                    finger_angle = hand_angle(finger_points) # 計算手指角度，回傳長度為 5 的串列
                    #print(finger_angle)                     # 印出角度 ( 有需要就開啟註解 )
                    text = hand_pos(finger_angle)            # 取得手勢所回傳的內容
                    if text=="Good" or text=='YA!!!' or  text=='666!':
                        Grade+=1
                        total_flag+=1
                    elif text=="FXXK!":
                        Grade-=1
                        total_flag-=1
                        if len(result_hand.multi_handedness)==2:
                            Grade-=1
                            total_flag-=1
                        w=size[1]
                        h=size[0]
                        x_max = max(fx)              # 取出 x 座標最大值
                        y_max = max(fy)              # 取出 y 座標最大值
                        x_min = min(fx) - 10         # 取出 x 座標最小值
                        y_min = min(fy) - 10         # 取出 y 座標最小值
                        mosaic(debug_image,x_max,x_min,y_max,y_min,w,h)
                        bad_guy+=1
                    elif text=='FXXK YOU!!!':
                        Grade-=2
                        total_flag-=2
                        if len(result_hand.multi_handedness)==2:
                            Grade-=2
                            total_flag-=2
                        w=size[1]
                        h=size[0]
                        x_max = max(fx)              # 取出 x 座標最大值
                        y_max = max(fy)              # 取出 y 座標最大值
                        x_min = min(fx) - 10         # 取出 x 座標最小值
                        y_min = min(fy) - 10         # 取出 y 座標最小值
                        mosaic(debug_image,x_max,x_min,y_max,y_min,w,h)
                        bad_guy+=2
                    if len(result_hand.multi_handedness)==2:
                        print("both")
                        cv.putText(debug_image, text, (20,130), fontFace, 1.5, (255,255,255), 5, lineType) # 印出文字
                        cv.putText(debug_image, text, (900,130), fontFace, 1.5, (255,255,255), 5, lineType) # 印出文字
                    else:
                        if lbl=="Left":
                            cv.putText(debug_image, text, (30,130), fontFace, 1.5, (255,255,255), 5, lineType) # 印出文字
                        if lbl=="Right":
                            cv.putText(debug_image, text, (900,130), fontFace, 1.5, (255,255,255), 5, lineType) # 印出文字
       
        if  int(time.time()-ftime)<=5:
            cv.putText(debug_image,'You Bastard!!!', (550,40), fontFace, 1, (0,255,255), 3, lineType) # 印出文字 bgr
        
        if total_flag>=200:
            pos_flag+=1
            total_flag-=200
        elif total_flag<=-100:
            neg_flag+=1
            total_flag+=100
        cv.putText(debug_image, "Positive Count: "+str(pos_flag), (20, 30),cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        cv.putText(debug_image, "Negative Count: "+str(neg_flag), (20, 50),cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        cv.putText(debug_image, 'Emotional Score: '+str(Grade), (400,80), fontFace, 1.5, (255,255,0), 3, lineType) # 印出文字
        # Screen reflection
        t1,starttime=timer(period,starttime)
        #print(t1, starttime)
       
        if Grade<=-200 and t1:
            print("nAUU")
            print(Option[-1])
            if Option[-1]=="Send Line Message Periodically":
                line_notify(name+'同事有負面傾向，'+name+'的主管可能要多關心他',token)
            #line_notify('負面傾向',token)
            Grade=0
        elif Grade>=200 and t1:
            if Option[-1]=="Send Line Message Periodically":
                line_notify(name+'同事正面積極，但是這樣是有點反常...',token)
            #line_notify('正面積極',token)
            Grade=0
            #period+=1
        if bad_guy>=50 and t1:
            line_notify(name+'同事亂比不雅手勢，'+name+'的主管會讓他看不到明天上班太陽',token)
            #line_notify('同事亂比不雅手勢',token)
            bad_guy=0
            ftime=time.time()
            #time.sleep(1)
        out.write(debug_image)
        cv.imshow('Facial Emotion and Gesture Recognition', debug_image)
        # Process Key (ESC: end)
        key = cv.waitKey(10)
        if key == 27 or 0xFF == ord('q'):  # ESC
            if pos_flag<2 and neg_flag<2:
                line_notify('今天'+name+'的情緒穩定',token)
            elif pos_flag>2 and neg_flag<2:
                line_notify('今天'+name+'的今天情緒穩定且正面積極',token)
            elif pos_flag<2 and neg_flag>2:
                 line_notify('今天'+name+'的情緒持續負面傾向',token)
            elif pos_flag>2 and neg_flag>2:
                if (pos_flag-neg_flag>5):
                    line_notify('今天'+name+'的情緒不穩定，但是還是正面積極一些',token)
                elif (-5<pos_flag-neg_flag<5):
                    line_notify('今天'+name+'的情緒不穩定',token)
                else:
                    line_notify('今天'+name+'的情緒不穩定，且負面傾向嚴重',token)
            time.sleep(3)
            break
  
    
    cap.release()
    out.release()   
    cv.destroyAllWindows()