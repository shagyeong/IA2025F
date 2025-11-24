
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
