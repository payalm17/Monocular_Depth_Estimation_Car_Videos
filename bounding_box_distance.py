from email.mime import image
import time
import cv2
import os
import pandas as pd
import numpy as np

vid_path = 'C:/Users/payal/Downloads/BoostingMonocularDepth-main/BoostingMonocularDepth-main/front4.mp4'
csv_path = 'C:/Users/payal/Downloads/BoostingMonocularDepth-main/BoostingMonocularDepth-main/front4.mp4.csv'

def depthFunc(x1, y1, x2, y2):
    # with open("1foo1.csv") as file_name:
    #     array = np.loadtxt(file_name, delimiter=",")
    #     array_new = cv2.resize(array, (1056, 1920))
    depth_array = np.load('1depth.npy')
    array_new = cv2.resize(depth_array, (1056, 1920))
    print(array_new.shape)
    sum = 0
    numberOfElements = (x2-x1)*(y2-y1)

    for x in range(x1, x2+1):
        for y in range(y1, y2+1):
            sum = sum + array_new[x, y]

    avg_depth = sum/numberOfElements
    return round(avg_depth,2)


vidcap = cv2.VideoCapture(vid_path)
vidcap.set(cv2.CAP_PROP_POS_MSEC, 0000)
df = pd.read_csv(csv_path)
print(df.head())
while vidcap.isOpened():
    ret, frame = vidcap.read()
    fnumber = vidcap.get(cv2.CAP_PROP_POS_FRAMES)
    print(fnumber)
    df_cur = df[df['frame'] == fnumber][['x1', 'y1', 'x2', 'y2', 'cls']].values

    if (ret):
        cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
        for objs in df_cur:
            [x1, y1, x2, y2, clss] = objs
            depthVal = depthFunc(x1, y1, x2, y2)
            frame1 = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
            cv2.putText(frame1, str(depthVal), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12))
            cv2.imwrite('LastFrameDepths.png',frame1)
        cv2.imshow('', frame)
        if(fnumber == 12.0):
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
   
vidcap.release()
cv2.destroyAllWindows()