import numpy as np
import cv2
from time import time

from modules.Maps import map
from modules.astar import astar_search
from modules.UCS import UCS_search
from modules.bfs import bfs_search
from modules.dfs import dfs_search
from modules.greedy import greedy_search
#from lefthand import lefthand_search
from modules.Visualize import drawpath, upscale, colorize

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


if __name__ == "__main__":

    # 1) 실행할 알고리즘들을 (이름, 함수) 튜플로 리스트에 넣기
    algorithms = [
        ("A* Search (Manhattan)", astar_search),
        ("Uniform Cost Search",   UCS_search),
        ("Breadth First Search",  bfs_search),
        ("Depth First Search",    dfs_search),
        ("Greedy Best-First",     greedy_search)#,
#        ("Lefthand Search",       lefthand_search),
    ]

    # 2) for문으로 전부 실행
    for name, func in algorithms:
        run_and_show(name, func, map)

    # 모든 창 띄워놓고 아무 키 누르면 종료
    cv2.waitKey(0)
    cv2.destroyAllWindows()
