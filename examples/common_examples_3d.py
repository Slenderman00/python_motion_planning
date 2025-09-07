"""
@file: common_examples_3d.py
@breif: Examples of Python Motion Planning library
@author: Wu Maojia, Joar Heimonen
@update: 2025.4.11
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from python_motion_planning import *
from python_motion_planning.utils import Env3D, Grid3D, Map3D
from python_motion_planning.global_planner.graph_search import AStar3D

if __name__ == '__main__':
    # Create environment with custom obstacles
    grid_env = Grid3D(51, 31, 31)
    obstacles = grid_env.obstacles
    for i in range(10, 21):
        obstacles.add((i, 15, 10))
    for i in range(15):
        obstacles.add((20, i, 20))
    for i in range(15, 30):
        obstacles.add((30, i, 30))
    for i in range(16):
        obstacles.add((40, i, 15))
    grid_env.update(obstacles)

    map_env = Map3D(51, 31, 31)
    obs_rect = [
        [14, 12, 8, 2],
        [18, 22, 8, 3],
        [26, 7, 2, 12],
        [32, 14, 10, 2]
    ]
    obs_circ = [
        [7, 12, 3],
        [46, 20, 2],
        [15, 5, 2],
        [37, 7, 3],
        [37, 23, 3]
    ]
    map_env.update(obs_rect=obs_rect, obs_circ=obs_circ)

    # plot3d = Plot3D(start=(5, 5, 5), goal=(25, 31, 22), env=grid_env)
    # plot3d.plotEnv("test")
    # -------------global planners-------------
    plt = AStar3D(start=(5, 5, 2), goal=(25, 29, 22), env=grid_env)
    # plt = DStar(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = DStarLite(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = Dijkstra(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = GBFS(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = JPS(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = ThetaStar(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = LazyThetaStar(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = SThetaStar(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = LPAStar(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = VoronoiPlanner(start=(5, 5), goal=(45, 25), env=grid_env)

    # plt = RRT(start=(18, 8), goal=(37, 18), env=map_env)
    # plt = RRTConnect(start=(18, 8), goal=(37, 18), env=map_env)
    # plt = RRTStar(start=(18, 8), goal=(37, 18), env=map_env)
    # plt = InformedRRT(start=(18, 8), goal=(37, 18), env=map_env)

    # plt = ACO(start=(5, 5), goal=(45, 25), env=grid_env)
    # plt = PSO(start=(5, 5), goal=(45, 25), env=grid_env)

    plt.run()
    # -------------local planners-------------
    # plt = PID(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = DWA(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = APF(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = LQR(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = RPP(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt = MPC(start=(5, 5, 0), goal=(45, 25, 0), env=grid_env)
    # plt.run()

    # -------------curve generators-------------
    # points = [(0, 0, 0), (10, 10, -90), (20, 5, 60), (30, 10, 120),
    #           (35, -5, 30), (25, -10, -120), (15, -15, 100), (0, -10, -90)]

    # plt = Dubins(step=0.1, max_curv=0.25)
    # plt = Bezier(step=0.1, offset=3.0)
    # plt = Polynomial(step=2, max_acc=1.0, max_jerk=0.5)
    # plt = ReedsShepp(step=0.1, max_curv=0.25)
    # plt = CubicSpline(step=0.1)
    # plt = BSpline(step=0.01, k=3)

    # plt.run(points)
