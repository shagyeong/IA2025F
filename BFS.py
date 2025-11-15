adj_list_orig = [
    [3, 2, 1], #0
    [5, 4, 2, 0], #1
    [5, 3, 1, 0], #2
    [6, 5, 2, 0], #3
    [7, 5, 1], #4
    [7, 6, 4, 3, 2, 1], #5
    [7, 5, 3], #6
    [6, 5, 4] #7
]

adj_list = [sorted(sublist, reverse=True) for sublist in adj_list_orig]

n = len(adj_list)
visited = [False for _ in range(n)]

def bfs(v):
    queue = []
    visited[v] = True
    queue.append(v)

    while len(queue) != 0:
        u = queue.pop(0)
        print(u, ' ', end='')

        for w in adj_list[u]:
            if not visited[w]:
                visited[w] = True
                queue.append(w)

print('BFS 방문 순서:')
for i in range(n):
    if not visited[i]:
        bfs(i)