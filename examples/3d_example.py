"""
@file: common_examples_3d.py
@brief: Minimal, easy-to-debug 3D maps for pathfinding sanity checks
@author: Joar Heimonen
@update: 2025.09.09
"""
import argparse
from scenarios import scenarios, carve_safety_bubble
from algorithms import algorithms
from python_motion_planning.utils import Grid3D
from python_motion_planning.global_planner.graph_search import AStar3D, DStar3D, Dijkstra3D, GBFS3D, JPS3D, LazyThetaStar3D, ThetaStar3D, VoronoiPlanner3D, LPAStar3D
from python_motion_planning.global_planner.evolutionary_search import ACO3D, PSO3D

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple 3D maps for pathfinding tests")
    parser.add_argument(
        "--scenario",
        choices=scenarios.keys(),
        default="door",
        help="Which map to generate"
    )
    parser.add_argument("--width", type=int, default=21, help="Grid size X")
    parser.add_argument("--height", type=int, default=15, help="Grid size Y")
    parser.add_argument("--depth", type=int, default=11, help="Grid size Z")
    parser.add_argument(
        "--planner",
        "-p",
        choices=algorithms.keys(),
        default="astar",
        help="Which planner to run"
    )
    args = parser.parse_args()

    grid_env = Grid3D(args.width, args.height, args.depth)
    XR, YR, ZR = grid_env.x_range, grid_env.y_range, grid_env.z_range

    # Start in one corner, goal in opposite corner (inside the walls)
    start = (1, 1, 1)
    goal = (XR - 2, YR - 2, ZR - 2)

    scenario = scenarios[args.scenario]
    obstacles = scenario(grid_env)

    # Ensure start/goal are always free
    carve_safety_bubble(obstacles, start, radius=1)
    carve_safety_bubble(obstacles, goal, radius=1)

    grid_env.update(obstacles)

    algorithm = algorithms[args.planner]
    planner = algorithm(start=start, goal=goal, env=grid_env)

    planner.run()

    # Keep viewer alive if your vis relies on a running process
    while True:
        pass
