"""
@file: lazy_theta_star.py
@breif: Lazy Theta* motion planning
@author: Yang Haodong, Wu Maojia, Joar Heimonen
@update: 2025-09-10
"""
import heapq
import itertools
import logging
from typing import Dict, List, Optional, Tuple

from .graph_search_3d import GraphSearcher3D
from python_motion_planning.utils import Env3D, Grid3D, Node3D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

Coord = Tuple[int, int, int]


class LazyThetaStar3D(GraphSearcher3D):
    """
    Lazy Theta* for 3D grids (any-angle / any-voxel pathfinding).
    - Uses deferred line-of-sight validation at expansion time.
    - Neighbor generation uses your 3D motion set (6/18/26-connected).
    """

    def __init__(self, start: Coord, goal: Coord, env: Grid3D, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)
        self.start.g = 0.0
        self.start.h = self.h(self.start, self.goal)

    def __str__(self) -> str:
        return "Lazy Theta* 3D"

    # ---------- public API ----------

    def plan(self) -> tuple:
        """
        Returns:
            cost (float): total path cost
            path (list[Coord]): start->goal path
            expand (list[Node3D]): expanded nodes for visualization
        """
        OPEN: List[tuple] = []
        CLOSED: Dict[Coord, Node3D] = {}
        counter = itertools.count()

        heapq.heappush(OPEN, (self.start.g + self.start.h, self.start.h, next(counter), self.start))

        expansions = 0
        log_interval = 1000

        while OPEN:
            _, _, _, node = heapq.heappop(OPEN)

            # Lazy validation: if parent exists in CLOSED but no LOS, fix parent/g using best CLOSED neighbor
            node_p = CLOSED.get(node.parent)
            if node_p and not self.lineOfSight(node_p, node):
                # invalidate and relax via CLOSED neighbors (classic Lazy Theta* "set-vertex")
                node.g = float("inf")
                for nbh in self.getNeighbor(node):
                    if nbh.current in CLOSED:
                        cand = CLOSED[nbh.current]
                        new_g = cand.g + self.dist(cand, node)
                        if new_g < node.g:
                            node.g = new_g
                            node.parent = cand.current

            # If we already have a better CLOSED entry, skip
            best_closed = CLOSED.get(node.current)
            if best_closed is not None and node.g >= best_closed.g:
                continue

            CLOSED[node.current] = node
            expansions += 1

            # goal check
            if node == self.goal:
                logging.info(f"Goal reached after expanding {expansions} nodes")
                cost, path = self.extractPath(CLOSED)
                return cost, path, list(CLOSED.values())

            if expansions % log_interval == 0:
                logging.info(f"Expanded {expansions} nodes, OPEN size={len(OPEN)}")

            # expand neighbors
            for nbr in self.getNeighbor(node):
                # endpoint obstacle guard + CLOSED dominance
                if nbr.current in self.obstacles:
                    continue
                closed_n = CLOSED.get(nbr.current)
                tentative_g_path1 = node.g + self.dist(node, nbr)
                if closed_n is not None and tentative_g_path1 >= closed_n.g:
                    continue

                # path 1: parent = node (optimistic; LOS will be verified lazily when popped)
                q = Node3D(nbr.current, node.current, tentative_g_path1, 0.0)
                q.h = self.h(q, self.goal)

                # path 2: optionally use node's parent if it seems cheaper (still lazy—it'll be verified at pop)
                node_p = CLOSED.get(node.parent)
                if node_p:
                    self.updateVertex(node_p, q)

                heapq.heappush(OPEN, (q.g + q.h, q.h, next(counter), q))

        logging.warning("No path found")
        return float("inf"), [], list(CLOSED.values())

    def run(self):
        cost, path, expand = self.plan()
        self.plot.animation(path, str(self), cost, expand)

    # ---------- core helpers ----------

    def updateVertex(self, node_p: Node3D, node_c: Node3D) -> None:
        """
        "Path 2" update: try connecting node_c to node_p (node's parent).
        In Lazy Theta*, this does NOT check LOS here; LOS is deferred until pop.
        """
        alt_g = node_p.g + self.dist(node_c, node_p)
        if alt_g < node_c.g:
            node_c.g = alt_g
            node_c.parent = node_p.current

    def getNeighbor(self, node: Node3D) -> List[Node3D]:
        """
        3D neighbors using the motion set from the base class.
        Guards:
          - in-bounds (via env.grid_map if available)
          - endpoint not an obstacle
          - edge free (self.isCollision)
        """
        neighbors: List[Node3D] = []
        cx, cy, cz = node.current
        grid_map = getattr(self.env, "grid_map", None)

        for motion in self.motions:
            cand = node + motion  # Node3D with cand.current = (cx+dx, cy+dy, cz+dz)

            if grid_map is not None and cand.current not in grid_map:
                continue
            if cand.current in self.obstacles:
                continue
            if self.isCollision(node, cand):
                continue

            neighbors.append(cand)

        return neighbors

    # ---------- 3D line-of-sight ----------

    def lineOfSight(self, a: Node3D, b: Node3D) -> bool:
        """
        Integer 3D Bresenham LOS: returns True if all voxels on the line a->b are free.
        Includes both endpoints; call only for nodes inside the grid.
        """
        (x0, y0, z0) = a.current
        (x1, y1, z1) = b.current

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        sx = 1 if x1 >= x0 else -1
        sy = 1 if y1 >= y0 else -1
        sz = 1 if z1 >= z0 else -1

        # early exit if endpoints are blocked
        if a.current in self.obstacles or b.current in self.obstacles:
            return False

        x, y, z = x0, y0, z0
        gm = getattr(self.env, "grid_map", None)

        def in_bounds(p: Coord) -> bool:
            return gm is None or p in gm

        # Visit each voxel along the line (excluding the start to avoid self-block)
        if dx >= dy and dx >= dz:
            err_y = dx // 2
            err_z = dx // 2
            while x != x1:
                x += sx
                err_y -= dy
                err_z -= dz
                if err_y < 0:
                    y += sy
                    err_y += dx
                if err_z < 0:
                    z += sz
                    err_z += dx
                if not in_bounds((x, y, z)) or (x, y, z) in self.obstacles:
                    return False
            return True

        if dy >= dx and dy >= dz:
            err_x = dy // 2
            err_z = dy // 2
            while y != y1:
                y += sy
                err_x -= dx
                err_z -= dz
                if err_x < 0:
                    x += sx
                    err_x += dy
                if err_z < 0:
                    z += sz
                    err_z += dy
                if not in_bounds((x, y, z)) or (x, y, z) in self.obstacles:
                    return False
            return True

        # dz is dominant
        err_x = dz // 2
        err_y = dz // 2
        while z != z1:
            z += sz
            err_x -= dx
            err_y -= dy
            if err_x < 0:
                x += sx
                err_x += dz
            if err_y < 0:
                y += sy
                err_y += dz
            if not in_bounds((x, y, z)) or (x, y, z) in self.obstacles:
                return False
        return True

    # ---------- path reconstruction ----------

    def extractPath(self, closed_list: Dict[Coord, Node3D]) -> tuple:
        """
        Reconstruct start->goal path using parents stored in CLOSED.
        """
        cost = 0.0
        node = closed_list[self.goal.current]
        path = [node.current]

        while node != self.start:
            parent = closed_list[node.parent]
            cost += self.dist(node, parent)
            node = parent
            path.append(node.current)

        path.reverse()
        return cost, path
