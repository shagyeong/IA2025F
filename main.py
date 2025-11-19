import cv2
import numpy as np

from Maps import map
from astar import astar_search
from UCS import UCS_search
from Euclid import Eclid_search
#from bfs import bfs_search
#from dfs import dfs_search
from time import time
from Visualize import drawpath,upscale,colorize
from Visualize import YEL,PUR,RED

# Param = (RunTime,Answer_nodes,path_length)
def RunAlgorithm(func, map):
    st_time = time()
    Answer_nodes,path_length = func(map)
    ed_time = time()
    return (ed_time-st_time),Answer_nodes,path_length

if __name__ == "__main__":
    print("Astar Search with Manhattan Heuristic")
    fd_time,fd_node,fd_length = RunAlgorithm(astar_search, map)
    print("fd_time: ",fd_time,"fd_node: ",''' fd_node, ''' "fd_length: ",fd_length)

    map_astar=colorize(map,60)
    map_astar=upscale(map_astar,60,15)
    drawpath(map_astar,np.array(fd_node,np.uint32),15,3)
    cv2.imshow("path_atar",map_astar)
    cv2.waitKey(0)

    # print("\nAstar Search with Euclidean Heuristic")
    # fd_time, fd_node,fd_length = RunAlgorithm(Eclid_search, map)
    # print("fd_time: ",fd_time,''' "fd_node: ", fd_node,''' "fd_length: ",fd_length)

    # print("\nUniform Cost Search")
    # fd_time, fd_node,fd_length = RunAlgorithm(UCS_search, map)
    # print("fd_time: ",fd_time,"fd_node: ", fd_node,"fd_length: ",fd_length)
