import numpy as np
import cv2
import collections
import time # 시간 측정을 위한 time 모듈 추가

# --- 1. BFS 길찾기 함수 (수정됨: 노드 수 및 시간 측정 로직 추가) ---
def find_path_bfs(grid):
    """
    주어진 맵(list-of-lists)을 기반으로 BFS를 사용해 'S'에서 'G'까지의 경로를 찾습니다.
    실행 시간과 탐색 노드 수를 반환합니다.
    """
    
    rows = len(grid)
    cols = len(grid[0])
    
    start = None
    goal = None
    
    # 1. 시작점(S)과 도착점(G) 위치 찾기
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'G':
                goal = (r, c)
        if start and goal:
            break
    
    if not start or not goal:
        return None, "시작점 'S' 또는 도착점 'G'를 찾을 수 없습니다.", 0, 0

    # 2. BFS 초기화
    queue = collections.deque([start])
    visited = {start}
    parent_map = {start: None} # 경로 역추적용
    
    # *** 성능 측정 변수 추가 ***
    nodes_explored = 0
    start_time = time.time()

    # 3. BFS 탐색
    path_found = False
    while queue:
        current_r, current_c = queue.popleft()
        
        nodes_explored += 1 # 탐색 노드 수 증가 (큐에서 꺼낼 때 카운트)

        if (current_r, current_c) == goal:
            path_found = True
            break # 목표 도착

        # 4방향 탐색 (상, 하, 좌, 우)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = current_r + dr, current_c + dc

            # 맵 범위 체크, 방문 여부 체크
            if 0 <= r < rows and 0 <= c < cols and (r, c) not in visited:
                
                # *** 중요: 벽(1)이 아닌지 체크 ***
                if grid[r][c] != 1: 
                    visited.add((r, c))
                    queue.append((r, c))
                    parent_map[(r, c)] = (current_r, current_c)

    end_time = time.time()
    execution_time = end_time - start_time

    # 4. 경로 재구성
    if not path_found:
        return None, "경로를 찾을 수 없습니다.", execution_time, nodes_explored

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent_map.get(current) # 안전하게 .get() 사용
    
    path.reverse() # S -> G 순서로 변경
    return path, "경로 찾기 성공", execution_time, nodes_explored
