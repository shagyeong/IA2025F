import numpy as np
import cv2
from time import time
from Maps import map   # 안 써도 상관없지만 그대로 둠
from heapq import heappush, heappop
from math import sqrt
from collections import deque
import heapq

# ========================= 시각화 모듈 ========================= #

YEL = (0, 255, 255)  # yellow
PUR = (153, 0, 102)  # purple
RED = (0, 0, 255)    # red

# 맵 색상화 함수
# src: list 0, 1, "S", "G"
def colorize(src, size) -> np.array:
    dst = np.zeros((size, size, 3), np.uint8)
    for i in range(size):
        for j in range(size):
            if   src[i][j] == 0: dst[i][j] = PUR
            elif src[i][j] == 1: dst[i][j] = YEL
            else:                dst[i][j] = RED  # 시작, 종료 노드
    return dst

# 맵 업스케일링 함수
def upscale(src, size, scale) -> np.array:
    row = size * scale
    col = size * scale
    dst = np.zeros((row, col, 3), np.uint8)
    for i in range(size):
        for j in range(size):
            for k in range(scale):
                for l in range(scale):
                    dst[i * scale + k][j * scale + l] = src[i][j]
    return dst

# 정답열 경로 표시 함수
def drawpath(src, ans, scale, thickness) -> None:
    ans = ans * scale  # scaled answer coordinate
    ans = ans + ((scale + 1) // 2)
    for i in range(ans.shape[0] - 1):
        ans1 = (ans[i][1],   ans[i][0])
        ans2 = (ans[i+1][1], ans[i+1][0])
        cv2.line(src, ans1, ans2, RED, thickness)

# ========================= 휴리스틱 ========================= #

def Heuristic(a, b):        # Manhattan
    (r1, c1), (r2, c2) = a, b
    return abs(r1 - r2) + abs(c2 - c1)

def Heuristic2(a, b):       # Euclidean
    (r1, c1), (r2, c2) = a, b
    return sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

# ========================= A* (Manhattan) ========================= #

def astar_search(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # S, G 찾기
    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise Exception("Start or Target Error")

    open_set = []
    heappush(open_set, (Heuristic(start, goal), 0, start))

    came_from = {}
    g_score = {start: 0}
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    explored_nodes = 0

    while open_set:
        f, g, current = heappop(open_set)

        if g > g_score.get(current, float('inf')):
            continue

        explored_nodes += 1

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            path_length = len(path)
            return path, path_length, explored_nodes

        cr, cc = current
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell = grid[nr][nc]
            if cell == 1:
                continue

            neighbor = (nr, nc)
            tentative_g = g + 1

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + Heuristic(neighbor, goal)
                heappush(open_set, (f_score, tentative_g, neighbor))

    return [], 0, explored_nodes

# ========================= UCS ========================= #

def UCS_search(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise Exception("Start or Target Error")

    open_set = []
    heappush(open_set, (0, 0, start))

    came_from = {}
    g_score = {start: 0}
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    explored_nodes = 0

    while open_set:
        f, g, current = heappop(open_set)

        if g > g_score.get(current, float('inf')):
            continue

        explored_nodes += 1

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            path_length = len(path)
            return path, path_length, explored_nodes

        cr, cc = current
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell = grid[nr][nc]
            if cell == 1:
                continue

            neighbor = (nr, nc)
            tentative_g = g + 1

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g  # 휴리스틱 없음
                heappush(open_set, (f_score, tentative_g, neighbor))

    return [], 0, explored_nodes

# ========================= BFS ========================= #

def bfs_search(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise Exception("Start or Target Error")

    open_set = deque([start])
    came_from = {start: None}
    g_score = {start: 0}
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    explored_nodes = 0

    while open_set:
        current = open_set.popleft()
        explored_nodes += 1

        if current == goal:
            path = [current]
            while current in came_from and came_from[current] is not None:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, len(path), explored_nodes

        cr, cc = current
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            neighbor = (nr, nc)
            cell = grid[nr][nc]
            if cell == 1:
                continue

            if neighbor not in came_from:
                tentative_g = g_score[current] + 1
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                open_set.append(neighbor)

    return [], 0, explored_nodes

# ========================= DFS ========================= #

def dfs_search(grid):
    rows, cols = len(grid), len(grid[0])

    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'G':
                goal = (r, c)
    if start is None or goal is None:
        raise Exception("Start or Goal not found")

    visited = [[False] * cols for _ in range(rows)]
    parent = {}
    stack = [start]

    explored_nodes = 0

    while stack:
        x, y = stack.pop()

        if visited[x][y]:
            continue
        visited[x][y] = True
        explored_nodes += 1

        if (x, y) == goal:
            path = []
            while (x, y) != start:
                path.append((x, y))
                x, y = parent[(x, y)]
            path.append(start)
            path.reverse()
            return path, len(path), explored_nodes

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if not visited[nx][ny] and grid[nx][ny] != 1:
                    stack.append((nx, ny))
                    parent[(nx, ny)] = (x, y)

    return [], 0, explored_nodes

# ========================= Greedy Best-First ========================= #

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def greedy_search(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    if start is None or goal is None:
        return [], 0, 0

    open_set = []
    initial_h = manhattan_distance(start, goal)
    heapq.heappush(open_set, (initial_h, start))

    came_from = {start: None}
    visited = {start}
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    explored_nodes = 0

    while open_set:
        h, current = heapq.heappop(open_set)
        explored_nodes += 1

        if current == goal:
            path = [current]
            while current in came_from and came_from[current] is not None:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, len(path), explored_nodes

        cr, cc = current
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell = grid[nr][nc]
            if cell == 1:
                continue

            neighbor = (nr, nc)
            if neighbor not in visited:
                new_h = manhattan_distance(neighbor, goal)
                visited.add(neighbor)
                came_from[neighbor] = current
                heapq.heappush(open_set, (new_h, neighbor))

    return [], 0, explored_nodes

# ========================= A* (Euclid) ========================= #

def Eclid_search(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    start = goal = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "S":
                start = (r, c)
            elif grid[r][c] == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise Exception("Start or Target Error")

    open_set = []
    heappush(open_set, (Heuristic2(start, goal), 0, start))

    came_from = {}
    g_score = {start: 0}
    directions = [(-1,0), (1,0), (0,-1), (0,1)]

    explored_nodes = 0

    while open_set:
        f, g, current = heappop(open_set)

        if g > g_score.get(current, float('inf')):
            continue

        explored_nodes += 1

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            path_length = len(path)
            return path, path_length, explored_nodes

        cr, cc = current
        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell = grid[nr][nc]
            if cell == 1:
                continue

            neighbor = (nr, nc)
            tentative_g = g + 1

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + Heuristic2(neighbor, goal)
                heappush(open_set, (f_score, tentative_g, neighbor))

    return [], 0, explored_nodes

# ========================= 공통 유틸 ========================= #

# Param = (RunTime, Answer_nodes, path_length, explored_nodes)
def RunAlgorithm(func, map_):
    st_time = time()
    path, path_length, explored_nodes = func(map_)
    ed_time = time()
    return (ed_time - st_time), path, path_length, explored_nodes

def run_and_show(name, func, map_):
    """알고리즘 실행 + 경로 시각화 이미지 + 시간/길이/노드수 반환"""
    print(f"\n=== {name} ===")
    fd_time, fd_node, fd_length, fd_explored = RunAlgorithm(func, map_)

    print(f"fd_time     : {fd_time:.4f} sec")
    print(f"fd_length   : {fd_length}")
    print(f"fd_explored : {fd_explored}")

    size = len(map_)
    img = colorize(map_, size)
    img = upscale(img, size, 7)
    if len(fd_node) > 0:
        drawpath(img, np.array(fd_node, np.uint32), 7, 2)

    return img, fd_time, fd_length, fd_explored

def make_header(width, height, title, time_sec, path_len, node_cnt):
    """
    알고리즘 이름 + 시간 + 경로 길이 + 탐색 노드 수를 가운데 정렬해서 그린 헤더
    """
    header = np.ones((height, width, 3), dtype=np.uint8) * 255  # 흰 배경

    lines = [
        title,
        f"time: {time_sec:.4f} s",
        f"length: {path_len}",
        f"nodes: {node_cnt}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 12

    total_text_height = line_height * len(lines)
    y0 = (height - total_text_height) // 2 + line_height  # 첫 줄 baseline

    for i, text in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (width - tw) // 2
        y = y0 + i * line_height
        cv2.putText(header, text, (x, y), font, font_scale,
                    (0, 0, 0), thickness, cv2.LINE_AA)

    return header

def make_labeled_grid(results, rows, cols):
    """
    results: [(name, img, time, length, explored), ...]
    rows, cols: 그래프 기준 (예: 2행 3열)
    """
    if not results:
        return None

    sample_img = results[0][1]
    gh, gw, gc = sample_img.shape

    header_h = 60

    canvas_h = rows * (header_h + gh)
    canvas_w = cols * gw
    canvas = np.zeros((canvas_h, canvas_w, gc), dtype=np.uint8)

    for idx, (name, img, t, length, explored) in enumerate(results):
        if idx >= rows * cols:
            break

        r = idx // cols
        c = idx % cols

        x0 = c * gw
        header_y0 = r * (header_h + gh)
        graph_y0 = header_y0 + header_h

        header_img = make_header(gw, header_h, name, t, length, explored)

        canvas[header_y0:header_y0 + header_h, x0:x0 + gw] = header_img
        canvas[graph_y0:graph_y0 + gh, x0:x0 + gw] = img

    return canvas

def load_map_from_txt(path: str):
    grid = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = []
            for ch in line:
                if ch in ("S", "G"):
                    row.append(ch)
                else:
                    row.append(int(ch))
            grid.append(row)
    return grid

# ========================= main ========================= #

if __name__ == "__main__":
    map_txt = load_map_from_txt("map.txt")

    algorithms = [
        ("A* Search (Manhattan)", astar_search),
        ("Uniform Cost Search",   UCS_search),
        ("Breadth First Search",  bfs_search),
        ("Depth First Search",    dfs_search),
        ("Greedy Best-First",     greedy_search),
        ("Euclid",                Eclid_search),
    ]

    results = []  # (name, img, time, length, explored)
    for name, func in algorithms:
        img, t, length, explored = run_and_show(name, func, map_txt)
        results.append((name, img, t, length, explored))

    grid_img = make_labeled_grid(results, rows=2, cols=3)

    cv2.imwrite("result_path.png", grid_img)
    cv2.imshow("All Algorithms (2x3 Grid + Header)", grid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
