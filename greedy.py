def greedy_search(grid):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # 1. Start S, Goal G 찾기
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

    # 스택 기반 (LIFO) → DFS 형태의 "순수 그리디"
    stack = [start]
    visited = {start}
    came_from = {start: None}

    # 이웃 탐색 순서 (고정된 로컬 규칙만 사용, 휴리스틱 X)
    # 예: 오른쪽, 아래, 왼쪽, 위 순으로 탐색
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    explored_nodes = 0

    while stack:
        current = stack.pop()  # 스택에서 하나 꺼내기 (가장 최근에 넣은 것)
        explored_nodes += 1

        if current == goal:
            # 경로 재구성
            path = [current]
            while current in came_from and came_from[current] is not None:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path, explored_nodes  # 또는 len(path)로 바꿔도 됨

        cr, cc = current

        for dr, dc in directions:
            nr, nc = cr + dr, cc + dc
            # 맵 경계 체크
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            cell = grid[nr][nc]
            # 장애물(1)은 통과 불가
            if cell == 1:
                continue

            neighbor = (nr, nc)
            if neighbor in visited:
                continue

            visited.add(neighbor)
            came_from[neighbor] = current
            stack.append(neighbor)  # DFS: 가능한 이웃을 스택에 쌓으면서 진행

    # 경로를 찾지 못한 경우
    return None, explored_nodes
