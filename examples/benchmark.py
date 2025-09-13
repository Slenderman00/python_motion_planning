"""
@file: common_examples_3d.py
@brief: Minimal, easy-to-debug 3D maps for pathfinding sanity checks
@author: Joar Heimonen
@update: 2025.09.09
"""

from python_motion_planning.utils import Grid3D
from python_motion_planning.global_planner.graph_search import AStar3D, DStar3D, Dijkstra3D, GBFS3D, JPS3D, LazyThetaStar3D, ThetaStar3D, VoronoiPlanner3D, LPAStar3D
from python_motion_planning.global_planner.evolutionary_search import ACO3D, PSO3D
from scenarios import scenarios, carve_safety_bubble
from algorithms import algorithms
import time
import csv
from tqdm import tqdm

if __name__ == '__main__':
    width = 21
    height = 15
    depth = 11

    algorithm_names = [
        "astar",
        "dstar",
        "dijkstra",
        "gbfs",
        # "jps",
        "lazy_theta_star",
        "theta_star",
        # "voronoi",
        "lpastar",
    ]

    iterations = 100

    import random
    with open('3d_pathfinding_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Scenario', 'Algorithm', 'Runtime (s)', 'Distance', 'Visited Nodes', "Start", "Goal", "Seed"])

        for scenario, scenario_func in tqdm(scenarios.items()):
            for algorithm_name in tqdm(algorithm_names):
                algorithm = algorithms[algorithm_name]

                for i in range(iterations):
                    grid_env = Grid3D(width, height, depth)
                    XR, YR, ZR = grid_env.x_range, grid_env.y_range, grid_env.z_range

                    # Seed random for reproducibility
                    random.seed(i)
                    # Random start and goal (not on walls)
                    start = (
                        random.randint(1, XR - 2),
                        random.randint(1, YR - 2),
                        random.randint(1, ZR - 2)
                    )
                    goal = (
                        random.randint(1, XR - 2),
                        random.randint(1, YR - 2),
                        random.randint(1, ZR - 2)
                    )
                    # Ensure start and goal are not the same
                    while goal == start:
                        goal = (
                            random.randint(1, XR - 2),
                            random.randint(1, YR - 2),
                            random.randint(1, ZR - 2)
                        )

                    obstacles = scenario_func(grid_env)

                    # Ensure start/goal are always free
                    carve_safety_bubble(obstacles, start, radius=2)
                    carve_safety_bubble(obstacles, goal, radius=2)

                    grid_env.update(obstacles)

                    planner = algorithm(start=start, goal=goal, env=grid_env)

                    start_time = time.time()
                    cost, path, expand = planner.plan()
                    end_time = time.time()

                    visited_nodes = len(expand)
                    runtime = end_time - start_time

                    # Write results to CSV
                    writer.writerow([scenario, algorithm_name, runtime, cost, visited_nodes, start, goal, i])

        print("Results saved to 3d_pathfinding_results.csv")