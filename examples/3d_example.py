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
    parser.add_argument("--width", type=int, default=26, help="Grid size X")
    parser.add_argument("--height", type=int, default=20, help="Grid size Y")
    parser.add_argument("--depth", type=int, default=16, help="Grid size Z")
    parser.add_argument(
        "--planner",
        "-p",
        choices=algorithms.keys(),
        default="astar",
        help="Which planner to run"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Enable random start and goal positions"
    )
    args = parser.parse_args()

    import random

    grid_env = Grid3D(args.width, args.height, args.depth)
    XR, YR, ZR = grid_env.x_range, grid_env.y_range, grid_env.z_range

    if args.random:
        # Pick random free positions inside the grid (not on the walls)
        def random_pos():
            return (
                random.randint(1, XR - 2),
                random.randint(1, YR - 2),
                random.randint(1, ZR - 2)
            )
        start = random_pos()
        goal = random_pos()
        # Ensure start and goal are not the same
        while goal == start:
            goal = random_pos()
    else:
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