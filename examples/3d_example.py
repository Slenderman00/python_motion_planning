"""
@file: common_examples_3d.py
@brief: Minimal, easy-to-debug 3D maps for pathfinding sanity checks
@author: Joar Heimonen
@update: 2025.09.09
"""
import sys, os, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python_motion_planning.utils import Grid3D
from python_motion_planning.global_planner.graph_search import AStar3D, DStar3D, Dijkstra3D, GBFS3D, JPS3D, LazyThetaStar3D, ThetaStar3D, VoronoiPlanner3D, LPAStar3D
from python_motion_planning.global_planner.evolutionary_search import ACO3D, PSO3D


def shell_walls(grid):
    """Thin boundary walls so agents stay in-bounds."""
    XR, YR, ZR = grid.x_range, grid.y_range, grid.z_range
    obs = set()
    for x in [0, XR - 1]:
        for y in range(YR):
            for z in range(ZR):
                obs.add((x, y, z))
    for y in [0, YR - 1]:
        for x in range(XR):
            for z in range(ZR):
                obs.add((x, y, z))
    for z in [0, ZR - 1]:
        for x in range(XR):
            for y in range(YR):
                obs.add((x, y, z))
    return obs


def scenario_empty_box(grid):
    """Only the bounding box. Great for baseline timing & correctness."""
    return shell_walls(grid)


def scenario_two_rooms_with_door(grid, door_size=2):
    """
    One interior wall splitting the box into two rooms, with a small door
    centered to ensure a single obvious choke point.
    """
    XR, YR, ZR = grid.x_range, grid.y_range, grid.z_range
    obs = shell_walls(grid)

    x0 = XR // 2  # interior wall at mid X
    for y in range(1, YR - 1):
        for z in range(1, ZR - 1):
            obs.add((x0, y, z))

    # carve a small "door" in the wall
    y_mid, z_mid = YR // 2, ZR // 2
    for dy in range(-(door_size // 2), door_size - (door_size // 2)):
        for dz in range(-(door_size // 2), door_size - (door_size // 2)):
            obs.discard((x0, y_mid + dy, z_mid + dz))

    return obs


def scenario_stacked_floors(grid, floors=3, hole_size=2):
    """
    A few horizontal floors (full planes) with small offset holes,
    so the algorithm must 'climb' in Z through different openings.
    """
    XR, YR, ZR = grid.x_range, grid.y_range, grid.z_range
    obs = shell_walls(grid)

    # choose evenly spaced Z-levels for floors
    zs = [ZR * (i + 1) // (floors + 1) for i in range(floors)]

    # place floors with small holes that shift position each layer
    for i, z0 in enumerate(zs):
        for x in range(1, XR - 1):
            for y in range(1, YR - 1):
                obs.add((x, y, z0))

        # carve a hole on each floor, shifting in X/Y per layer
        hx = 2 + (i * 3) % (XR - 4)
        hy = 2 + (i * 2) % (YR - 4)
        for dx in range(hole_size):
            for dy in range(hole_size):
                obs.discard((min(XR - 2, hx + dx), min(YR - 2, hy + dy), z0))

    return obs


def carve_safety_bubble(obs, center, radius=1):
    """Guarantee free space around start/goal."""
    cx, cy, cz = center
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                obs.discard((cx + dx, cy + dy, cz + dz))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple 3D maps for pathfinding tests")
    parser.add_argument(
        "--scenario",
        choices=["empty", "door", "floors"],
        default="door",
        help="Which map to generate"
    )
    parser.add_argument("--width", type=int, default=21, help="Grid size X")
    parser.add_argument("--height", type=int, default=15, help="Grid size Y")
    parser.add_argument("--depth", type=int, default=11, help="Grid size Z")
    parser.add_argument(
        "--planner",
        "-p",
        choices=["astar", "dstar", "dijkstra", "gbfs", "jps", "lazy_theta_star", "theta_star", "voronoi", "lpastar", "aco", 'pso'],
        default="astar",
        help="Which planner to run"
    )
    args = parser.parse_args()

    grid_env = Grid3D(args.width, args.height, args.depth)
    XR, YR, ZR = grid_env.x_range, grid_env.y_range, grid_env.z_range

    # Start in one corner, goal in opposite corner (inside the walls)
    start = (1, 1, 1)
    goal = (XR - 2, YR - 2, ZR - 2)

    if args.scenario == "empty":
        obstacles = scenario_empty_box(grid_env)
    elif args.scenario == "door":
        obstacles = scenario_two_rooms_with_door(grid_env, door_size=2)
    else:
        obstacles = scenario_stacked_floors(grid_env, floors=3, hole_size=2)

    # Ensure start/goal are always free
    carve_safety_bubble(obstacles, start, radius=1)
    carve_safety_bubble(obstacles, goal, radius=1)

    grid_env.update(obstacles)

    if args.planner == "astar":
        planner = AStar3D(start=start, goal=goal, env=grid_env, heuristic_type="euclidean")
 
    if args.planner == "dstar":
        planner = DStar3D(start=start, goal=goal, env=grid_env)

    if args.planner == "dijkstra":
        planner = Dijkstra3D(start=start, goal=goal, env=grid_env)

    if args.planner == "gbfs":
        planner = GBFS3D(start=start, goal=goal, env=grid_env)

    if args.planner == "jps":
        planner = JPS3D(start=start, goal=goal, env=grid_env)
 
    if args.planner == "lazy_theta_star":
        planner = LazyThetaStar3D(start=start, goal=goal, env=grid_env)
    
    if args.planner == "theta_star":
        planner = ThetaStar3D(start=start, goal=goal, env=grid_env)

    if args.planner == "voronoi":
        planner = VoronoiPlanner3D(start=start, goal=goal, env=grid_env)
    
    if args.planner == "lpastar":
        planner = LPAStar3D(start=start, goal=goal, env=grid_env)
    
    if args.planner == "aco":
        planner = ACO3D(start=start, goal=goal, env=grid_env)

    if args.planner == "pso":
        planner = PSO3D(start=start, goal=goal, env=grid_env)
 
    planner.run()

    # Keep viewer alive if your vis relies on a running process
    while True:
        pass
