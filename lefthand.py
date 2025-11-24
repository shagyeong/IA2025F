# import numpy as np
# import cv2

# # 색상 세팅
# YEL=(0,255,255) # yellow
# PUR=(153,0,102) # pulple
# RED=(0,0,255)   # red

# # 맵 업스케일링 함수
# def upscale(src,size,scale)->np.array:
#     row=size*scale
#     col=size*scale
#     dst=np.zeros((row,col,3),np.uint8)
#     for i in range(0,size,1):
#         for j in range(0,size,1):
#             for k in range(0,scale,1):
#                 for l in range(0,scale,1):
#                     dst[i*scale+k][j*scale+l]=src[i][j]
#     return dst

# # 정답열 경로 표시 함수
# def drawpath(src,ans,scale,thickness)->None:
#     ans=ans*scale # scaled answer coordinate
#     ans=ans+((scale+1)//2)
#     for i in range(0,ans.shape[0]-1,1):
#         ans1=(ans[i][1],  ans[i][0])
#         ans2=(ans[i+1][1],ans[i+1][0])
#         cv2.line(src,ans1,ans2,RED,thickness)

# # 현재 진행 방향 기준 왼쪽에 길이 있으면 왼쪽으로 회전 후 직진합니다.
# # 왼쪽에 벽이 있고 앞에 길이 있으면 직진합니다.
# # 왼쪽과 앞이 모두 막혔고 오른쪽에 길이 있으면 오른쪽으로 회전 후 직진합니다.
# # 세 방향 모두 막혔으면 뒤로 돌아갑니다. 

# def solve_maze_left_hand(maze, start_x, start_y, end_x, end_y):
#     # 미로 크기
#     rows = len(maze)
#     cols = len(maze[0])

#     # 현재 위치 (x, y)와 현재 방향 (dx, dy)
#     # 초기 방향: 오른쪽 (0, 1)
#     x, y = start_x, start_y
#     # 방향: (북, 동, 남, 서) 시계 방향 순서 (회전을 쉽게 하기 위함)
#     dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
#     # 초기 방향 인덱스 (동쪽)
#     current_dir_idx = 1
    
#     path = []
#     visited = set()

#     while (x, y) != (end_x, end_y):
#         if (x, y) in visited:
#             # 순환 방지를 위해 방문한 곳이면 종료 또는 다른 처리 필요
#             # 좌수법은 단일 경로 미로를 전제로 하므로 무한 루프에 빠질 수 있음
#             # 실제 복잡 미로에서는 트레모 알고리즘 등 다른 방법이 더 적합
#             # 여기서는 간단한 좌수법 구현에 초점을 둠
#             pass
#         visited.add((x, y))
#         path.append((x, y))

#         # 현재 방향 기준 왼쪽 방향의 인덱스
#         left_dir_idx = (current_dir_idx - 1 + 4) % 4
#         # 현재 방향 기준 앞 방향의 인덱스 (current_dir_idx와 동일)
#         # 현재 방향 기준 오른쪽 방향의 인덱스
#         right_dir_idx = (current_dir_idx + 1) % 4
#         # 현재 방향 기준 뒤 방향의 인덱스
#         back_dir_idx = (current_dir_idx + 2) % 4

#         # 1. 왼쪽 방향에 길이 있는지 확인 (벽: 1, 길: 0)
#         lx, ly = x + dirs[left_dir_idx][0], y + dirs[left_dir_idx][1]
#         if 0 <= lx < rows and 0 <= ly < cols and maze[lx][ly] == 0:
#             current_dir_idx = left_dir_idx # 왼쪽으로 회전
#             x, y = lx, ly
#             continue
        
#         # 2. 앞에 길이 있는지 확인
#         fx, fy = x + dirs[current_dir_idx][0], y + dirs[current_dir_idx][1]
#         if 0 <= fx < rows and 0 <= fy < cols and maze[fx][fy] == 0:
#             x, y = fx, fy
#             continue

#         # 3. 오른쪽에 길이 있는지 확인
#         rx, ry = x + dirs[right_dir_idx][0], y + dirs[right_dir_idx][1]
#         if 0 <= rx < rows and 0 <= ry < cols and maze[rx][ry] == 0:
#             current_dir_idx = right_dir_idx # 오른쪽으로 회전
#             x, y = rx, ry
#             continue

#         # 4. 뒤로 돌아가기 (모두 막힌 경우)
#         bx, by = x + dirs[back_dir_idx][0], y + dirs[back_dir_idx][1]
#         if 0 <= bx < rows and 0 <= by < cols and maze[bx][by] == 0:
#             current_dir_idx = back_dir_idx # 뒤로 회전
#             x, y = bx, by
#             continue
        
#         # 이동할 수 없는 경우 (막힌 길)
#         break
        
#     if (x, y) == (end_x, end_y):
#         return path + [(x, y)]
#     else:
#         return None # 출구에 도달하지 못함

# # 미로 예시 (0은 길, 1은 벽)
# maze = [
#     [1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 1],
#     [1, 0, 1, 1, 1, 0, 1],
#     [1, 0, 1, 0, 0, 0, 1],
#     [1, 0, 1, 0, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1]
# ]
# start_x, start_y = 1, 1
# end_x, end_y = 5, 5
# path = solve_maze_left_hand(maze, start_x, start_y, end_x, end_y)
# path = np.array(path,np.uint8)

# map=np.zeros((7,7,3),np.uint8)
# for i in range(0,7,1):
#     for j in range(0,7,1):
#         if maze[i][j]==1:map[i][j]=YEL
#         else:            map[i][j]=PUR

# map=upscale(map,7,15)
# drawpath(map,path,15,2)
# cv2.imshow("map",map)
# cv2.waitKey(0)