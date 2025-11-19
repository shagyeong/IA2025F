from Maps import map
from astar import astar_search
#from bfs import bfs_search
#from dfs import dfs_search
from time import time
#from Visualize import drawpath,upscale

# Param = (RunTime,Answer_nodes,path_length)
def RunAlgorithm(func, map):
    st_time = time()
    Answer_nodes,path_length = func(map)
    ed_time = time()
    return (ed_time-st_time),Answer_nodes,path_length

if __name__ == "__main__":
    fd_time, fd_node,fd_length = RunAlgorithm(astar_search, map)
    print("fd_time: ",fd_time,"fd_node: ", fd_node,"fd_length: ",fd_length)
