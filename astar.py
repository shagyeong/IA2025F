from Maps import map
from heapq import heappush, heappop
from math import sqrt
import time

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


#st_time = time.time()
#explored_nodes,path_length,path = astar_search(map)
#ed_etime = time.time()
#print(f"실행 시간: {ed_etime - st_time:.6f} 초")
#print(f"탐색 노드 수: {explored_nodes:,}")   # 2,930 처럼 3자리마다 콤마 표시
#print(f"경로 길이: {path_length}")

# (4) 경로가 있을 경우 격자 위에 표시해 보고 싶다면:


#if path is not None:
#    # S, G는 그대로 두고 0인 길은 '*'로 표시
#    for (r, c) in path:
#        if map[r][c] == 0:      # 지금 map 에서는 0이 int 이니까 '0' 말고 0
#            map[r][c] = '*'     # 경로를 *로 표시

    # 격자를 문자열로 다시 출력
#    for row in map:
#        print("".join(str(x) for x in row))

#else:
#    print("경로가 없습니다.")