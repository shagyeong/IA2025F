from collections import deque # BFS를 위해 큐(deque)를 사용
import time

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
