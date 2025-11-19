from Maps import map
from astar import astar_search
from bfs import bfs_search
from dfs import dfs_search
from time import time
from Visualize import drawpath,upscale

def RunAlgorithm(func, map):
    st_time = time()
    explored_nodes,path_length,path = func(map)
    ed_time = time()
    return (ed_time-st_time),explored_nodes,path_length,path


if __name__ == "__main__":
    fd_time, fd_node,fd_length,fd_path = RunAlgorithm(astar_search, map)
    