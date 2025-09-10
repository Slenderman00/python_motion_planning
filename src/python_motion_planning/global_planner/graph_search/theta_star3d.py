"""
@file: theta_star.py
@breif: Theta* motion planning
@author: Yang Haodong, Wu Maojia, Joar Heimonen
@update: 2025-09-10
"""
import heapq
import itertools
import logging
from typing import Dict, List, Tuple

from .graph_search_3d import GraphSearcher3D
from python_motion_planning.utils import Env3D, Grid3D, Node3D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

Coord = Tuple[int, int, int]


class ThetaStar3D(GraphSearcher3D):
    """
    Theta* for 3D voxel grids (any-angle / any-voxel pathfinding).
    Uses immediate line-of-sight checks to shortcut via the parent's parent.
    """

    def __init__(self, start: Coord, goal: Coord, env: Grid3D, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)
        self.start.g = 0.0
        self.start.h = self.h(self.start, self.goal)

    def __str__(self) -> str:
        return "Theta* 3D"

    def plan(self) -> tuple:
        """
        Returns:
            cost (float): path cost
            path (list[Coord]): start->goal path
            expand (list[Node3D]): nodes expanded (for visualization)
        """
        OPEN: List[tuple] = []
        CLOSED: Dict[Coord, Node3D] = {}
        counter = itertools.count()

        heapq.heappush(OPEN, (self.start.g + self.start.h, self.start.h, next(counter), self.start))

        expansions = 0
        log_interval = 1000

        while OPEN:
            _, _, _, node = heapq.heappop(OPEN)

            best_closed = CLOSED.get(node.current)
            if best_closed is not None and node.g >= best_closed.g:
                continue

            CLOSED[node.current] = node
            expansions += 1

            if node == self.goal:
                logging.info(f"Goal reached after expanding {expansions} nodes")
                cost, path = self.extractPath(CLOSED)
                return cost, path, list(CLOSED.values())

            if expansions % log_interval == 0:
                logging.info(f"Expanded {expansions} nodes, OPEN size={len(OPEN)}")

            for nbr in self.getNeighbor(node):
                # endpoint obstacle guard (often redundant with getNeighbor but cheap)
                if nbr.current in self.obstacles:
                    continue
                closed_n = CLOSED.get(nbr.current)

                # Path 1: connect via `node`
                g1 = node.g + self.dist(node, nbr)
                if closed_n is not None and g1 >= closed_n.g:
                    continue

                q = Node3D(nbr.current, node.current, g1, 0.0)
                q.h = self.h(q, self.goal)

                # Path 2: if LOS from parent(node) to neighbor, try that shortcut immediately
                node_p = CLOSED.get(node.parent)
                if node_p:
                    self.updateVertex(node_p, q)

                heapq.heappush(OPEN, (q.g + q.h, q.h, next(counter), q))

        logging.warning("No path found")
        return float("inf"), [], list(CLOSED.values())

    def run(self):
        cost, path, expand = self.plan()
        self.plot.animation(path, str(self), cost, expand)

    # ---------- Theta* core ----------

    def updateVertex(self, node_p: Node3D, node_c: Node3D) -> None:
        """
        Attempt 'path 2' through node_p if there is line-of-sight(node_p, node_c).
        """
        if self.lineOfSight(node_c, node_p):
            alt_g = node_p.g + self.dist(node_c, node_p)
            if alt_g < node_c.g:
                node_c.g = alt_g
                node_c.parent = node_p.current

    def getNeighbor(self, node: Node3D) -> List[Node3D]:
        """
        3D neighbor generation using base motion set.
        Guards:
          - in-bounds (via env.grid_map if present)
          - endpoint free
          - edge free (self.isCollision)
        """
        neighbors: List[Node3D] = []
        grid_map = getattr(self.env, "grid_map", None)

        for motion in self.motions:
            cand = node + motion  # Node3D with updated .current tuple

            if grid_map is not None and cand.current not in grid_map:
                continue
            if cand.current in self.obstacles:
                continue
            if self.isCollision(node, cand):
                continue

            neighbors.append(cand)

        return neighbors

    # ---------- 3D Line of Sight (integer Bresenham) ----------

    def lineOfSight(self, a: Node3D, b: Node3D) -> bool:
        """
        Returns True if the straight segment from a -> b passes only through free voxels.
        Includes endpoints.
        """
        (x0, y0, z0) = a.current
        (x1, y1, z1) = b.current

        # early out if endpoints blocked
        if a.current in self.obstacles or b.current in self.obstacles:
            return False

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)
        sx = 1 if x1 >= x0 else -1
        sy = 1 if y1 >= y0 else -1
        sz = 1 if z1 >= z0 else -1

        x, y, z = x0, y0, z0
        gm = getattr(self.env, "grid_map", None)

        def in_bounds(p: Coord) -> bool:
            return gm is None or p in gm

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

        # dz dominant
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
        Build start->goal path using parents stored in CLOSED.
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
