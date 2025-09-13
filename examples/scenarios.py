import random


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

def scenario_3d_maze(grid, seed=0, vertical_connector_prob=0.12):
    """
    3D perfect maze (DFS/backtracker) on an odd-grid lattice.
    Cells at odd coordinates are passages; even coordinates are walls.
    Returns a set of obstacle voxels suitable for Grid3D.update().

    Args:
        grid: Grid3D
        seed: RNG seed for reproducibility (set None for random)
        vertical_connector_prob: chance to add extra Z tunnels between already
                                 carved neighbors to increase 3D-ness (0..1)
    """
    XR, YR, ZR = grid.x_range, grid.y_range, grid.z_range
    rng = random.Random(seed)

    # Helper: is an odd coordinate inside the inner box (excluding shell walls)?
    def is_odd_interior(x, y, z):
        return (1 <= x < XR - 1 and 1 <= y < YR - 1 and 1 <= z < ZR - 1
                and x % 2 == 1 and y % 2 == 1 and z % 2 == 1)

    # Lattice of "cells" only on odd coords
    start = (1, 1, 1)
    # If dimensions are too small to host odd lattice, just return shell.
    if not is_odd_interior(*start):
        return shell_walls(grid)

    stack = [start]
    visited = {start}
    passages = {start}  # voxel positions that will be *free*

    # 6-neighborhood steps in lattice (jump by 2 to land on next cell)
    dirs = [(2, 0, 0), (-2, 0, 0), (0, 2, 0), (0, -2, 0), (0, 0, 2), (0, 0, -2)]

    while stack:
        cx, cy, cz = stack[-1]
        rng.shuffle(dirs)
        advanced = False
        for dx, dy, dz in dirs:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            if not is_odd_interior(nx, ny, nz) or (nx, ny, nz) in visited:
                continue
            # carve passage to neighbor: add midpoint and neighbor as passages
            wx, wy, wz = cx + dx // 2, cy + dy // 2, cz + dz // 2
            passages.add((wx, wy, wz))
            passages.add((nx, ny, nz))
            visited.add((nx, ny, nz))
            stack.append((nx,ny,nz))
            advanced = True
            break
        if not advanced:
            stack.pop()

    # Add a few extra vertical connectors between already-adjacent passages
    # to avoid the maze becoming too layer-like when Z is small.
    if ZR >= 5 and vertical_connector_prob > 0:
        for x in range(1, XR - 1, 2):
            for y in range(1, YR - 1, 2):
                for z in range(3, ZR - 2, 2):  # check neighbors above/below
                    if ((x, y, z) in passages and (x, y, z - 2) in passages
                        and rng.random() < vertical_connector_prob):
                        passages.add((x, y, z - 1))
                    if ((x, y, z) in passages and (x, y, z + 2) in passages
                        and rng.random() < vertical_connector_prob):
                        passages.add((x, y, z + 1))

    # Build obstacles: start with full interior filled, subtract passages, then add shell.
    obs = shell_walls(grid)
    for x in range(1, XR - 1):
        for y in range(1, YR - 1):
            for z in range(1, ZR - 1):
                if (x, y, z) not in passages:
                    obs.add((x, y, z))
    return obs


def scenario_city(grid, skyscraper_density=1, seed=2):
    """
    City scenario: dense vertical skyscrapers (columns) distributed throughout the grid.
    Skyscrapers are tough obstacles for pathfinding, forcing navigation around them.
    """
    XR, YR, ZR = grid.x_range, grid.y_range, grid.z_range
    rng = random.Random(seed)
    obs = shell_walls(grid)

    # Place skyscrapers as vertical columns, avoiding shell walls and leaving some corridors
    num_skyscrapers = int((XR - 2) * (YR - 2) * skyscraper_density)
    possible_positions = [
        (x, y)
        for x in range(2, XR - 2)
        for y in range(2, YR - 2)
        if x % 3 != 0 and y % 3 != 0  # leave some corridors for navigation
    ]
    rng.shuffle(possible_positions)
    skyscraper_positions = possible_positions[:num_skyscrapers]

    for x, y in skyscraper_positions:
        # Each skyscraper gets a random height (at least 3, up to ZR-2)
        height = rng.randint(3, ZR - 2) if ZR > 3 else 1
        base_z = 1
        for z in range(base_z, base_z + height):
            if z < ZR - 1:
                obs.add((x, y, z))

    return obs


def carve_safety_bubble(obs, center, radius=1):
    """Guarantee free space around start/goal."""
    cx, cy, cz = center
    # Determine grid bounds from existing obstacles (assumes shell_walls present)
    xs = [x for x, _, _ in obs]
    ys = [y for _, y, _ in obs]
    zs = [z for _, _, z in obs]
    if xs and ys and zs:
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
    else:
        # Fallback: allow all
        min_x = min_y = min_z = 0
        max_x = max_y = max_z = 999999
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            for dz in range(-radius, radius + 1):
                x, y, z = cx + dx, cy + dy, cz + dz
                # Only carve if not on boundary
                if (
                    x > min_x and x < max_x and
                    y > min_y and y < max_y and
                    z > min_z and z < max_z
                ):
                    obs.discard((x, y, z))


scenarios = {
    "empty": lambda grid_env: scenario_empty_box(grid_env),
    "door": lambda grid_env: scenario_two_rooms_with_door(grid_env, door_size=2),
    "floors": lambda grid_env: scenario_stacked_floors(grid_env, floors=3, hole_size=2),
    "maze": lambda grid_env: scenario_3d_maze(grid_env, seed=0),
    "city": lambda grid_env: scenario_city(grid_env, skyscraper_density=0.18, seed=0),
}