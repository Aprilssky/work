#傅里叶动画
import cv2
import math
import numpy as np

img = cv2.imread('pikaqiu.jpg')

# 灰度处理
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 高斯模糊
grayimg = cv2.GaussianBlur(grayimg, (3,3),0,0)

# 二值化处理
#ret, grayimg = cv2.threshold(grayimg,128, 255,cv2.THRESH_BINARY)
# 二值化容易出现使提取边缘时边界断开

# 提取边缘算子
cannyImage = cv2.Canny(grayimg, 128,255,5)

# contours为输出轮廓数据集合
contours, heirarchy = cv2.findContours(cannyImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#轮廓图显示
# cv2.imshow("contour",cannyImage)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

path = []
circleCount = 500; #圆的数量
values_nums = len(contours[0])
delta_x = 250
delta_y = 250
f_m = 0.5 #放大系数
for i in range(len(contours[0])):
    path.append([contours[0][i][0][0]*f_m-delta_x,contours[0][i][0][1]*f_m-delta_y])

def exp(theta):
    # 指数变复数
    return [math.cos(theta), math.sin(theta)]

def mul(za,zb):
    # 复数相乘
    return [za[0]*zb[0]-za[1]*zb[1], za[0]*zb[1]+za[1]*zb[0]]

def add(za,zb):
    # 复数相加
    return [za[0]+zb[0],za[1]+zb[1]]

K = []
#k = [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6,...]
for i in range(circleCount):
    if i&1:
        K.append(-(1+i>>1))
    else:
        K.append((1+i>>1))

def getCn(path,K):
    z=[0,0]
    Cn = []
    N = len(path)
    for j in range(len(K)):
        for i in range(N):
            za = [path[i][0], path[i][1]];
            zb = exp(K[j]*i*2*-math.pi/N)
            z = add(z, mul(za,zb))
        z[0] = z[0]/N;
        z[1] = z[1]/N;
        Cn.append([z[0],z[1]]) 
    return Cn

Cn = getCn(path,K)

# 画圆函数
def DrawCircles(path,Cn,img,K,time):
    global center_X,center_Y
    p = [center_X,center_Y]
    a = 2* math.pi*time/len(path)
    for i in range(len(Cn)):
        r = math.hypot(Cn[i][0], Cn[i][1])
        if i>0:
            cv2.circle(img,(int(p[0]),int(p[1])),int(r),(255,128,32))  
            cv2.circle(img,(int(p[0]),int(p[1])),int(r/10),(255,128,32),-1)  
        p = add(p, mul(Cn[i],exp(a*K[i])))

# 绘制连接圆心函数
def DrawLines(path,Cn,img,K,time):
    global center_X,center_Y
    p = [center_X,center_Y]
    a = 2* math.pi*time/len(path)
    for i in range(len(Cn)-1):
        p1 = add(p, mul(Cn[i],exp(a*K[i])))
        if i>0:
            cv2.line(img,(int(p[0]),int(p[1])),(int(p1[0]),int(p1[1])),(0,255,0),1)
        p = p1     

# 绘制路径函数
def DrawPath(path,Cn,img,K,time):
    global center_X,center_Y
    global values_x, values_y
    global values_nums
    p = [center_X,center_Y]
    a = 2* math.pi*time/len(path)
    for i in range(len(Cn)):
        p = add(p, mul(Cn[i],exp(a*K[i])))
    x=int(p[0])
    y=int(p[1])
    cv2.circle(img,(x,y),2,(255,128,32),-1)  
    if len(values_x)<values_nums:
        values_x.append(x)
        values_y.append(y)
    else:
        values_x = values_x[1:]+[x]
        values_y = values_y[1:]+[y]
    for i in range(len(values_x)-1):
        if len(values_x)>1:
            cv2.line(img,(values_x[i],values_y[i]),(values_x[i+1],values_y[i+1]),(255,255,0),2)

time = 0
values_x = []
values_y = []
center_X = 400
center_Y = 400

while True:
    img = np.zeros([800, 800, 3], np.uint8) 
    DrawCircles(path,Cn,img,K,time)
    DrawPath(path,Cn,img,K,time)
    DrawLines(path,Cn,img,K,time)
    time = time +2
    #print(time)
    cv2.imshow('pikaqiu',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
