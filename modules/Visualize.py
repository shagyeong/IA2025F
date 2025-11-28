import numpy as np
import cv2

# 색상 세팅
YEL=(0,255,255) # yellow
PUR=(153,0,102) # pulple
RED=(0,0,255)   # red

# 맵 색상화 함수
# src: list 0, 1, "S", "G"
def colorize(src,size)->np.array:
    dst=np.zeros((size,size,3),np.uint8)
    for i in range(0,size,1):
        for j in range(0,size,1):
            if   src[i][j]==0:dst[i][j]=PUR
            elif src[i][j]==1:dst[i][j]=YEL
            else:             dst[i][j]=RED # 시작, 종료 노드
    return dst

# 맵 업스케일링 함수
def upscale(src,size,scale)->np.array:
    row=size*scale
    col=size*scale
    dst=np.zeros((row,col,3),np.uint8)
    for i in range(0,size,1):
        for j in range(0,size,1):
            for k in range(0,scale,1):
                for l in range(0,scale,1):
                    dst[i*scale+k][j*scale+l]=src[i][j]
    return dst

# 정답열 경로 표시 함수
def drawpath(src,ans,scale,thickness)->None:
    ans=ans*scale # scaled answer coordinate
    ans=ans+((scale+1)//2)
    for i in range(0,ans.shape[0]-1,1):
        ans1=(ans[i][1],  ans[i][0])
        ans2=(ans[i+1][1],ans[i+1][0])
        cv2.line(src,ans1,ans2,RED,thickness)
