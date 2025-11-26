import numpy as np
import cv2
from time import time
from Maps import map
from heapq import heappush, heappop
from math import sqrt
from collections import deque
import heapq

# 맵
map=[
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,"S",0,1,1,0,0,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,0,1,1,0,1,1,1,0,0,0,0,1,0,1,1,0,0,0,1,0,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,1,0,1],
    [1,1,0,0,0,1,0,0,0,1,1,1,1,1,0,0,1,0,1,0,1,1,1,1,1,1,0,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,0,0,1,1,0,1,1,1,0,0,1,1,1,0,0,1],
    [1,1,0,0,1,0,1,0,1,1,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1,1,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,0,1,1,0,1,1,0,1],
    [1,0,0,0,0,1,1,0,0,0,1,1,1,0,1,1,1,1,1,1,0,1,0,1,1,0,0,0,0,0,1,1,1,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,1,1,0,1,1,1,0,1,0,1,1,1],
    [1,1,0,0,0,0,1,1,0,1,1,1,0,1,0,1,1,0,1,1,0,1,1,0,0,1,1,1,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,1,1,1,1,1,0,0,1,0,1,1,0,0,1,1,1,1],
    [1,1,0,0,1,0,1,1,1,1,0,0,1,0,1,1,1,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,1,0,1,0,0,1,0,1,1,0,1,1,1,1,0,0,1,1],
    [1,0,0,0,1,0,1,0,1,0,1,1,1,0,1,0,1,1,1,1,0,1,0,1,0,0,0,1,1,0,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,1,0,0,1,1],
    [1,0,1,0,0,0,0,0,1,1,1,1,1,0,1,1,0,1,0,1,1,0,1,0,0,1,1,1,0,1,1,1,0,1,0,0,0,1,0,0,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,1,1,1,1,0,1,0,0,0,1,1,1,0,0,0,1,0,1,0,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,0,1,0,0,1,1,1,1,0,1,1],
    [1,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,0,0,1,0,0,1,1,1,0,0,1,0,1,0,0,1,0,0,0,1,0,0,1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,1,0,1,1,1,0,1],
    [1,1,0,0,0,0,0,0,0,1,1,0,1,1,0,1,0,0,1,1,0,1,0,1,0,0,0,1,0,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1],
    [1,1,1,0,1,0,0,0,0,0,1,1,1,1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,0,0,1,1,1,1,0,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,1],
    [1,0,0,1,1,0,0,0,0,1,1,0,1,1,1,0,0,0,1,1,1,1,1,1,1,1,0,0,1,0,0,0,1,1,0,1,0,0,1,1,0,1,1,1,1,0,0,1,0,1,0,0,1,1,1,1,0,1,1,1],
    [1,0,0,1,1,1,1,0,0,0,1,1,0,1,0,1,0,0,1,1,1,1,1,0,0,1,1,0,1,0,1,1,0,0,1,1,1,0,1,0,0,1,0,1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1],
    [1,1,0,1,1,0,1,0,0,0,1,0,0,0,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,1,0,0,0,0,0,1,1,0,0,0,1,1,0,0,0,0,1,1,0,0,1,0,0,1,0,1,1,1,1,1],
    [1,1,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,1,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,1],
    [1,0,1,1,0,1,1,1,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,1,0,0,1,1,1,0,1,0,1,1,0,0,1,1,1,0,1,1,1,0,1,1,0,0,1],
    [1,0,1,0,1,1,1,1,1,0,1,1,1,0,0,1,0,1,0,0,0,0,1,0,1,0,0,1,1,0,1,1,1,1,0,1,1,0,1,1,0,1,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,0,1,1],
    [1,0,1,1,1,0,1,0,0,1,1,0,1,1,1,0,1,0,1,1,0,0,1,0,1,1,0,1,1,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,0,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1],
    [1,1,0,0,0,0,0,0,1,1,1,0,1,1,0,1,0,1,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,1,0,1,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1,1,0,1,1],
    [1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,0,1,1,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,1,1],
    [1,1,1,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,0,1,0,0,0,1,1,1,0,0,1,0,0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1],
    [1,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,1,0,1,1,1,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,1,1,1,1],
    [1,0,1,1,1,0,1,1,1,0,1,0,1,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,1,0,1,0,1,1,0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,1,0,0,1,0,1,1,1],
    [1,1,1,0,1,0,1,1,1,1,1,1,0,1,1,0,1,1,1,1,0,0,0,0,0,0,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,0,1,0,1,0,0,0,1,0,1,1,1],
    [1,1,1,1,0,1,0,1,1,0,1,1,0,0,1,1,1,0,0,0,1,0,1,1,0,0,0,0,1,0,0,1,1,1,0,1,0,1,0,1,1,0,1,0,1,0,0,1,0,1,0,0,0,1,0,0,1,1,0,1],
    [1,1,1,1,0,0,0,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,0,1,0,0,0,0,1,0,1,1,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,0,1,1,1,0,1,0,1,1,0,0,1],
    [1,1,0,0,1,0,0,1,0,1,0,0,0,1,0,1,0,1,1,1,1,1,0,1,0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0,0,1,0,1,1,0,1,0,0,1,0,1,0,1,0,1,0,0,1,1],
    [1,1,1,1,1,1,0,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,0,1,0,1,1,1,1,1,0,1,0,1,1,1,1,0,0,1,1,0,1,0,0,0,1,1,1,1,1,1,1,0,1],
    [1,1,0,0,1,0,1,0,1,0,0,1,1,1,1,1,1,1,1,0,1,1,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,0,0,0,1,1,0,0,0,1,0,1,1,0,0,0,1,0,1,0,1,1,1,1],
    [1,1,0,1,0,0,0,0,1,1,1,1,0,1,1,1,0,1,1,0,0,0,0,0,1,1,1,0,0,1,1,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,0,1,1,1,1,1,1,0,0,1,0,1,1],
    [1,1,0,1,0,1,1,0,1,0,1,1,0,1,0,0,1,1,0,1,0,0,1,1,1,0,0,0,1,0,1,0,0,1,1,0,0,0,0,1,1,1,0,1,0,1,1,0,0,1,1,1,1,1,1,1,0,0,1,1],
    [1,0,1,0,0,0,1,1,0,1,1,1,1,1,1,0,0,1,0,1,0,0,1,1,1,1,1,0,0,1,1,0,1,0,1,1,1,0,0,1,1,1,1,0,1,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,0,1,1,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,1],
    [1,0,1,0,1,0,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,0,0,0,1,1,0,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,0,1,0,1,0,0,0,1,0,1,1,1],
    [1,1,1,0,0,0,1,0,1,1,0,0,0,1,0,0,1,1,0,1,0,1,0,0,1,0,0,0,0,1,0,1,0,1,1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,1],
    [1,0,1,0,0,0,1,1,0,1,0,1,0,1,0,1,1,0,0,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,1,1,1,1,0,1,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,0,0,1],
    [1,0,0,0,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,0,0,1,1,1,0,0,1,1,0,1,0,1,0,0,1,1,1,1,0,1,0,1,0,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,1],
    [1,1,1,0,1,0,0,1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1],
    [1,0,0,1,1,0,0,0,1,1,0,1,1,0,0,1,0,1,1,1,1,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,1,1,1,0,0,1,0,1,0,1,0,1,0,1],
    [1,0,1,0,1,0,0,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,0,0,0,1,0,1,1,1,1,0,1,1,0,1,0,0,0,1,0,1,1,1,1,0,1,1,1,0,1,1,0,1,1],
    [1,1,1,0,1,0,0,1,1,0,1,1,1,1,0,1,0,1,0,1,1,0,0,0,0,1,0,1,1,1,1,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,0,1,0,1,1],
    [1,1,1,0,1,1,1,1,1,0,0,1,1,1,0,0,1,0,1,0,1,0,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,0,0,0,1,0,0,0,0,1],
    [1,0,1,0,1,0,0,0,0,1,1,1,1,1,0,0,1,1,1,0,1,0,1,1,0,0,0,1,0,1,1,1,0,0,0,1,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,0,1,0,1,0,1,0,1,1],
    [1,0,1,1,0,1,1,1,0,1,0,0,1,1,1,1,0,1,0,0,0,1,1,1,1,0,1,1,0,1,0,0,1,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,1,0,1,1,1,1,1,0,0,0,1],
    [1,0,0,0,1,1,1,0,1,1,0,0,1,1,1,1,1,0,0,0,1,0,1,1,0,0,0,0,1,1,1,1,1,0,0,0,1,1,0,1,1,0,1,1,1,1,0,1,1,0,0,0,1,0,0,0,1,1,1,1],
    [1,1,0,1,0,0,1,0,0,1,0,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,1,1,1,0,1,1,1,1,0,1,1],
    [1,0,0,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,0,0,1,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,1],
    [1,0,0,0,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,1,0,0,0,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,0,0,1,0,0,1,0,1,0,0,1,1,0,1,1,0,1,1,0,1],
    [1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,1,1,1,0,1,1,1,1,0,0,0,1,0,1,0,0,0,1,0,1,0,0,1,0,1,0,1],
    [1,0,1,0,0,1,1,0,0,0,1,0,0,1,1,1,0,0,1,1,1,1,1,1,1,1,0,1,1,0,0,0,1,1,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1],
    [1,1,0,1,0,0,0,1,0,0,1,0,0,0,1,1,0,1,0,1,0,1,0,1,0,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,0,1,1,0,1,1,1,0,1,0,1,1],
    [1,1,0,1,1,0,1,0,1,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,0,1,1,1,0,1,1,0,0,1,1],
    [1,1,0,1,0,1,1,0,0,1,0,0,0,0,1,0,1,0,0,0,1,1,0,1,0,0,0,1,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,1,1,1,1],
    [1,0,0,1,0,1,1,0,0,0,1,0,1,1,1,0,1,0,1,0,1,0,1,1,0,1,1,0,0,0,0,0,1,1,1,1,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,1,1,1],
    [1,1,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,1,0,0,1,1,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1],
    [1,0,0,0,0,0,1,1,1,1,0,1,1,0,1,1,0,0,0,1,0,0,0,1,1,1,1,0,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,0,1,0,1,1,0,0,1,1,1,0,0,1,1],
    [1,1,0,0,1,0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1,1,1,1,1,1,0,1,0,0,1,0,1,1,1,1,1,0,1,1,0,1,1,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,"G",1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
]



# 시각화 모듈
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




# astar
def Heuristic(a,b): # We are using Menhaten because this map is grid
    (r1,c1),(r2,c2) = a,b #
    return abs(r1-r2) + abs(c2-c1)

def Heuristic2(a, b):
    """유클리드 거리 휴리스틱 (대각선 이동이 허용되거나, 연속 공간 느낌일 때)"""
    (r1, c1), (r2, c2) = a, b
    return sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)



def astar_search(grid):
    
    rows = len(grid)
    cols = len(grid[0]) if rows>0 else 0
    
    ## 1. Start S, Target G
    start = None
    goal = None
    
    for r in range(rows):
        for c in range(cols):
            if(grid[r][c] =="S"):
                start = (r,c)
            elif grid[r][c] == "G":
                goal = (r,c)
    
    if start is None or goal is None:
        raise Exception("Start or Target Error")
    
    
    #print(f"[INIT] start={start}, goal={goal}, rows={rows}, cols={cols}")

    # A start Ready
    
    # open_set: 아직 탐색할 후보 노드들을 저장하는 우선순위 큐(최소 힙)
    # 원소 형태: (f값, g값, (row, col))
    #   - f = g + h
    #   - g = 시작점에서 현재 노드까지의 실제 비용 (여기선 이동 칸 수)
    #   - h = 휴리스틱(맨해튼 거리)
    
    open_set = []
    heappush(open_set, (Heuristic(start,goal),0,start))
    
    
    # came_from: 각 노드에 도달할 때 사용한 '이전 노드'를 기록하는 딕셔너리
    #   - key: (row, col) 현재 노드
    #   - value: (row, col) 현재 노드로 오기 직전의 노드
    came_from = {}  # (r,c) -> before (r,c)
    
    # g_score: 시작점에서 특정 노드까지 도달하는데 드는 '최소 비용'을 저장
    #   - key: (row, col)
    #   - value: g 값 (시작에서 여기까지의 최소 거리)
    g_score = {start:0} # real cost at startPoint, dic
    
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    
    explored_nodes = 0
    
    # main Loop
    while open_set:
        # f: 현재 후보들 중 가장 유망한(예상 총 비용이 최소인) 노드의 f값
        # g: 그 노드까지의 실제 비용
        # current: 현재 노드 위치 (row, col)
        f,g, current = heappop(open_set)
        
        # current 노드에 대해 이미 더 좋은 g값이 기록되있으면 넘어감
        if g>g_score.get(current,float('inf')):
            continue
        
        explored_nodes +=1 #→ Open Set + Closed Set 포함 총 탐색한 노드 수
        
        # reach target
        if current == goal:
            
            path = [current]
            while current in came_from: # Put in value Backtracking
                current = came_from[current]
                path.append(current) # Add path the node
            path.reverse()           # 역추적-> 반대 뒤집힘, -> Reverse
            
            path_length = len(path)
            
            return path,path_length  # path is real return
        
        cr,cc = current # deposition row, col
        
        for dr,dc in directions:
            nr = cr+dr
            nc = cc+dc
            
            if not (0<=nr<rows and 0<=nc<cols):
                continue
            
            cell = grid[nr][nc]
            
            if cell == 1:
                continue
            
            neighbor = (nr,nc) # location of neighbor

            # tentative_g: 현재 노드를 거쳐서 이웃 노드에 도달했을 때의
            #              '후보 g값' (현재 g + 이동 비용 1)            
            tentative_g = g+1 # move One Step, One Step's Weight is 1
            
            # if find more valuable than last value
            if tentative_g < g_score.get(neighbor,float('inf')): # Get(A,B), A: key to find, B: if not find, default value
                came_from[neighbor] = current # record to current
                g_score[neighbor] = tentative_g  # neighbor까지의 최소 g값을 갱신
                f_score = tentative_g + Heuristic(neighbor,goal) # f값 = g + h = 현재까지 실제 비용 + 휴리스틱(맨해튼)
                heappush(open_set,(f_score,tentative_g,neighbor)) # open_set(우선순위 큐)에 이웃 노드를 새로운 후보로 삽입

    return path,0  # if not find.....
    


# UCS
def UCS_search(grid):
    
    rows = len(grid)
    cols = len(grid[0]) if rows>0 else 0
    
    ## 1. Start S, Target G
    start = None
    goal = None
    
    for r in range(rows):
        for c in range(cols):
            if(grid[r][c] =="S"):
                start = (r,c)
            elif grid[r][c] == "G":
                goal = (r,c)
    
    if start is None or goal is None:
        raise Exception("Start or Target Error")
    
    
    #print(f"[INIT] start={start}, goal={goal}, rows={rows}, cols={cols}")

    # A start Ready
    
    # open_set: 아직 탐색할 후보 노드들을 저장하는 우선순위 큐(최소 힙)
    # 원소 형태: (f값, g값, (row, col))
    #   - f = g + h
    #   - g = 시작점에서 현재 노드까지의 실제 비용 (여기선 이동 칸 수)
    #   - h = 휴리스틱(맨해튼 거리)
    
    open_set = []
    heappush(open_set, (0,0,start))
    
    
    # came_from: 각 노드에 도달할 때 사용한 '이전 노드'를 기록하는 딕셔너리
    #   - key: (row, col) 현재 노드
    #   - value: (row, col) 현재 노드로 오기 직전의 노드
    came_from = {}  # (r,c) -> before (r,c)
    
    # g_score: 시작점에서 특정 노드까지 도달하는데 드는 '최소 비용'을 저장
    #   - key: (row, col)
    #   - value: g 값 (시작에서 여기까지의 최소 거리)
    g_score = {start:0} # real cost at startPoint, dic
    
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    
    explored_nodes = 0
    
    # main Loop
    while open_set:
        # f: 현재 후보들 중 가장 유망한(예상 총 비용이 최소인) 노드의 f값
        # g: 그 노드까지의 실제 비용
        # current: 현재 노드 위치 (row, col)
        f,g, current = heappop(open_set)
        
        # current 노드에 대해 이미 더 좋은 g값이 기록되있으면 넘어감
        if g>g_score.get(current,float('inf')):
            continue
        
        explored_nodes +=1 #→ Open Set + Closed Set 포함 총 탐색한 노드 수
        
        # reach target
        if current == goal:
            
            path = [current]
            while current in came_from: # Put in value Backtracking
                current = came_from[current]
                path.append(current) # Add path the node
            path.reverse()           # 역추적-> 반대 뒤집힘, -> Reverse
            
            path_length = len(path)
            
            return path,path_length  # path is real return
        
        cr,cc = current # deposition row, col
        
        for dr,dc in directions:
            nr = cr+dr
            nc = cc+dc
            
            if not (0<=nr<rows and 0<=nc<cols):
                continue
            
            cell = grid[nr][nc]
            
            if cell == 1:
                continue
            
            neighbor = (nr,nc) # location of neighbor

            # tentative_g: 현재 노드를 거쳐서 이웃 노드에 도달했을 때의
            #              '후보 g값' (현재 g + 이동 비용 1)            
            tentative_g = g+1 # move One Step, One Step's Weight is 1
            
            # if find more valuable than last value
            if tentative_g < g_score.get(neighbor,float('inf')): # Get(A,B), A: key to find, B: if not find, default value
                came_from[neighbor] = current # record to current
                g_score[neighbor] = tentative_g  # neighbor까지의 최소 g값을 갱신
                f_score = tentative_g #+ Heuristic2(neighbor,goal) # f값 = g + h = 현재까지 실제 비용 + 휴리스틱(맨해튼)
                heappush(open_set,(f_score,tentative_g,neighbor)) # open_set(우선순위 큐)에 이웃 노드를 새로운 후보로 삽입

    return path,0  # if not find.....



# BFS
def bfs_search(grid):
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # 1. Start S, Target G 위치 찾기
    start = None
    goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    
    if start is None or goal is None:
        raise Exception("Start or Target Error")
    
    # open_set: 탐색할 후보 노드를 저장하는 큐 (FIFO)
    open_set = deque([start])
    
    # came_from: 경로 역추적을 위한 이전 노드 기록 (동시에 방문 기록 역할)
    came_from = {start: None}
    
    # g_score: 시작점에서 현재 노드까지의 실제 비용
    g_score = {start: 0} 
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 상하좌우
    
    explored_nodes = 0
    
    # main Loop
    while open_set:
        current = open_set.popleft() # 큐에서 노드를 꺼냄
        
        explored_nodes += 1
        
        # 목표 도달 시 경로 재구성 및 반환
        if current == goal:
            path = [current]
            while current in came_from and came_from[current] is not None:
                current = came_from[current]
                path.append(current) 
            path.reverse()
            return path, len(path)
        
        cr, cc = current
        
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            
            # 경계 확인
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            
            neighbor = (nr, nc)
            cell = grid[nr][nc]
            
            # 장애물 확인
            if cell == 1:
                continue
            
            # **핵심:** 아직 방문하지 않은 노드만 처리 (came_from에 없는 경우)
            if neighbor not in came_from:
                
                tentative_g = g_score[current] + 1
                g_score[neighbor] = tentative_g
                
                came_from[neighbor] = current # 이전 노드 기록
                open_set.append(neighbor)    # 큐에 삽입
                
    return None, 0 # 경로를 찾지 못한 경우


def dfs_search(grid):
    rows = len(grid)
    cols = len(grid[0])

    # S, G 찾기
    start = None
    goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'G':
                goal = (r, c)

    if start is None or goal is None:
        raise Exception("Start or Goal not found")

    visited = [[False] * cols for _ in range(rows)]
    path = []
    explored_nodes = 0

    # 스택 DFS (재귀 없이 하나의 함수로)
    stack = [start]

    while stack:
        x, y = stack.pop()

        # 이미 방문한 노드면 skip
        if visited[x][y]:
            continue

        visited[x][y] = True
        explored_nodes += 1
        path.append((x, y))

        # 목표 도달
        if (x, y) == goal:
            return path, len(path)

        # 4방향 탐색 (스택이므로 reverse해서 넣으면 자연스러운 DFS 순서)
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if not visited[nx][ny] and grid[nx][ny] != 1:
                    stack.append((nx, ny))

        # 막다른 길이면 되돌아감 (DFS path 유지)
        while path and not stack:
            path.pop()

    # 경로 없음
    return [], 0



    # 실행
    found = dfs(start[0], start[1])

    if found:
        return path, len(path)   # ← A*와 동일한 형식
    else:
        return [], 0



# Greedy BFS
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def greedy_search(grid):
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # 1. Start S, Target G 위치 찾기
    start = None
    goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    
    if start is None or goal is None:
        return None, 0

    # open_set: (h값, (row, col)) 형태의 우선순위 큐 (h = 휴리스틱)
    initial_h = manhattan_distance(start, goal)
    # heapq는 튜플의 첫 번째 요소를 기준으로 정렬합니다.
    open_set = []
    heapq.heappush(open_set, (initial_h, start))
    
    # came_from: 경로 역추적을 위한 이전 노드 기록
    came_from = {start: None}
    
    # visited: 이미 탐색했거나 큐에 추가했던 노드를 다시 처리하지 않기 위한 집합
    visited = {start}
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # 상하좌우
    
    explored_nodes = 0
    
    # main Loop
    while open_set:
        # h: 현재 후보들 중 가장 목표에 가깝다고 예상되는 노드의 h값
        # current: 현재 노드 위치 (row, col)
        h, current = heapq.heappop(open_set)
        
        explored_nodes += 1
        
        # 목표 도달 시 경로 재구성 및 반환
        if current == goal:
            path = [current]
            while current in came_from and came_from[current] is not None:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, len(path)
        
        cr, cc = current
        
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            neighbor = (nr, nc)
            
            # 맵 경계 확인
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            
            cell = grid[nr][nc]
            
            # 장애물(1) 확인
            if cell == 1:
                continue
            
            # **핵심:** 아직 방문하지 않은 노드만 큐에 추가
            if neighbor not in visited:
                
                new_h = manhattan_distance(neighbor, goal) # 새로운 h값 계산
                
                visited.add(neighbor)
                came_from[neighbor] = current # 이전 노드 기록
                
                # 큐에 추가: (h값, 노드) -> h가 낮을수록 우선순위 높음
                heapq.heappush(open_set, (new_h, neighbor))
                
    return None, 0 # 경로를 찾지 못한 경우



# Euclid
def Eclid_search(grid):
    
    rows = len(grid)
    cols = len(grid[0]) if rows>0 else 0
    
    ## 1. Start S, Target G
    start = None
    goal = None
    
    for r in range(rows):
        for c in range(cols):
            if(grid[r][c] =="S"):
                start = (r,c)
            elif grid[r][c] == "G":
                goal = (r,c)
    
    if start is None or goal is None:
        raise Exception("Start or Target Error")
    
    
    #print(f"[INIT] start={start}, goal={goal}, rows={rows}, cols={cols}")

    # A start Ready
    
    # open_set: 아직 탐색할 후보 노드들을 저장하는 우선순위 큐(최소 힙)
    # 원소 형태: (f값, g값, (row, col))
    #   - f = g + h
    #   - g = 시작점에서 현재 노드까지의 실제 비용 (여기선 이동 칸 수)
    #   - h = 휴리스틱(맨해튼 거리)
    
    open_set = []
    heappush(open_set, (Heuristic2(start,goal),0,start))
    
    
    # came_from: 각 노드에 도달할 때 사용한 '이전 노드'를 기록하는 딕셔너리
    #   - key: (row, col) 현재 노드
    #   - value: (row, col) 현재 노드로 오기 직전의 노드
    came_from = {}  # (r,c) -> before (r,c)
    
    # g_score: 시작점에서 특정 노드까지 도달하는데 드는 '최소 비용'을 저장
    #   - key: (row, col)
    #   - value: g 값 (시작에서 여기까지의 최소 거리)
    g_score = {start:0} # real cost at startPoint, dic
    
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    
    explored_nodes = 0
    
    # main Loop
    while open_set:
        # f: 현재 후보들 중 가장 유망한(예상 총 비용이 최소인) 노드의 f값
        # g: 그 노드까지의 실제 비용
        # current: 현재 노드 위치 (row, col)
        f,g, current = heappop(open_set)
        
        # current 노드에 대해 이미 더 좋은 g값이 기록되있으면 넘어감
        if g>g_score.get(current,float('inf')):
            continue
        
        explored_nodes +=1 #→ Open Set + Closed Set 포함 총 탐색한 노드 수
        
        # reach target
        if current == goal:
            
            path = [current]
            while current in came_from: # Put in value Backtracking
                current = came_from[current]
                path.append(current) # Add path the node
            path.reverse()           # 역추적-> 반대 뒤집힘, -> Reverse
            
            path_length = len(path)
            
            return path,path_length  # path is real return
        
        cr,cc = current # deposition row, col
        
        for dr,dc in directions:
            nr = cr+dr
            nc = cc+dc
            
            if not (0<=nr<rows and 0<=nc<cols):
                continue
            
            cell = grid[nr][nc]
            
            if cell == 1:
                continue
            
            neighbor = (nr,nc) # location of neighbor

            # tentative_g: 현재 노드를 거쳐서 이웃 노드에 도달했을 때의
            #              '후보 g값' (현재 g + 이동 비용 1)            
            tentative_g = g+1 # move One Step, One Step's Weight is 1
            
            # if find more valuable than last value
            if tentative_g < g_score.get(neighbor,float('inf')): # Get(A,B), A: key to find, B: if not find, default value
                came_from[neighbor] = current # record to current
                g_score[neighbor] = tentative_g  # neighbor까지의 최소 g값을 갱신
                f_score = tentative_g + Heuristic2(neighbor,goal) # f값 = g + h = 현재까지 실제 비용 + 휴리스틱(맨해튼)
                heappush(open_set,(f_score,tentative_g,neighbor)) # open_set(우선순위 큐)에 이웃 노드를 새로운 후보로 삽입

    return path,0  # if not find.....




# Param = (RunTime, Answer_nodes, path_length)
def RunAlgorithm(func, map_):
    st_time = time()
    answer_nodes, path_length = func(map_)   # <- 알고리즘 함수 호출
    ed_time = time()
    return (ed_time - st_time), answer_nodes, path_length


def run_and_show(name, func, map_):
    """알고리즘 하나 실행 + 결과 출력 + 경로 시각화"""
    print(f"\n=== {name} ===")
    fd_time, fd_node, fd_length = RunAlgorithm(func, map_)

    print(f"fd_time  : {fd_time:.4f} sec")
    print(f"fd_node  : {fd_node}")       # 필요하면 len(fd_node) 등으로 바꿔도 됨
    print(f"fd_length: {fd_length}")

    # 경로 시각화
    img = colorize(map_, 60)
    img = upscale(img, 60, 15)
    drawpath(img, np.array(fd_node, np.uint32), 15, 3)

    cv2.imshow(name, img)   # 알고리즘 이름으로 창 띄우기

def load_map_from_txt(path: str):
    grid = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()  # 양 끝 공백/개행 제거
            if not line:         # 빈 줄은 스킵
                continue
            
            row = []
            for ch in line:
                if ch in ("S", "G"):
                    row.append(ch)        # 시작/목표는 문자열로
                else:
                    row.append(int(ch))   # '0', '1' → 0, 1
            grid.append(row)
    return grid


if __name__ == "__main__":
    map_txt = load_map_from_txt("map.txt")  # map.txt 파일에서 맵 불러오기

    # 1) 실행할 알고리즘들을 (이름, 함수) 튜플로 리스트에 넣기
    algorithms = [
        ("A* Search (Manhattan)", astar_search),
        ("Uniform Cost Search",   UCS_search),
        ("Breadth First Search",  bfs_search),
        ("Depth First Search",    dfs_search),
        ("Greedy Best-First",     greedy_search),
        ("Euclid",                Eclid_search)
#        ("Lefthand Search",       lefthand_search),
    ]

    # 2) for문으로 전부 실행
    for name, func in algorithms:
        run_and_show(name, func, map_txt)

    # 모든 창 띄워놓고 아무 키 누르면 종료
    cv2.waitKey(0)
    cv2.destroyAllWindows()
