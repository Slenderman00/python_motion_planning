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

# Import trajectory planning components
from python_motion_planning.trajectory import (
    PolynomialTrajectory3D,
    SplineTrajectory3D,
    TimeOptimalTrajectory3D,
    TrajectoryConstraints
)
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple 3D maps for pathfinding tests")
    parser.add_argument(
        "--scenario",
        "-s",
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
        "-r",
        action="store_true",
        help="Enable random start and goal positions"
    )
    parser.add_argument(
        "--trajectory",
        "-t",
        choices=["none", "polynomial", "spline", "time_optimal"],
        default="none",
        help="Trajectory generation method (none = path only)"
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

    # Run path planning
    if args.trajectory != "none":
        # Run planning without visualization (we'll handle it with trajectory)
        cost, path, expand = planner.plan()

        if path:
            print(f"Path found: {len(path)} waypoints, cost: {cost:.2f}")

            # Generate trajectory
            print(f"Generating {args.trajectory} trajectory...")

            # Define trajectory constraints
            constraints = TrajectoryConstraints(
                max_velocity=np.array([2.0, 2.0, 1.5]),      # m/s
                max_acceleration=np.array([1.5, 1.5, 1.0]),   # m/s^2
                max_jerk=np.array([1.0, 1.0, 0.8]),          # m/s^3
                min_time_step=0.05
            )

            trajectory = None
            if args.trajectory == "polynomial":
                trajectory = PolynomialTrajectory3D(
                    path=path,
                    constraints=constraints
                )
                trajectory.optimize_time_allocation(max_iterations=5)
            elif args.trajectory == "spline":
                trajectory = SplineTrajectory3D(
                    path=path,
                    constraints=constraints,
                    spline_degree=3
                )
            elif args.trajectory == "time_optimal":
                trajectory = TimeOptimalTrajectory3D(
                    path=path,
                    constraints=constraints,
                    path_resolution=0.05
                )

            if trajectory:
                trajectory_points = trajectory.generate()
                print(f"Trajectory generated: {len(trajectory_points)} points, duration: {trajectory.total_time:.2f}s")

                # Check constraints
                constraint_status = trajectory.check_constraints()
                print(f"Constraints satisfied: {constraint_status}")

                # Visualize
                planner.plot.animation(
                    path=path,
                    name=f"{args.planner.title()} + {args.trajectory.title()} Trajectory",
                    cost=cost,
                    expand=expand,
                    trajectory=trajectory
                )
        else:
            print("No path found!")
    else:
        # Original visualization without trajectory
        planner.run()
