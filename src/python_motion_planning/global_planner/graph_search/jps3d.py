"""
@file: jps3d.py
@breif: Jump Point Search motion planning
@author: Yang Haodong, Wu Maojia, Joar Heimonen
@update: 2025-09-10
"""
import heapq
import itertools
import logging
from typing import Optional, Tuple, Iterable

from .graph_search_3d import GraphSearcher3D
from python_motion_planning.utils import Env3D, Grid3D, Node3D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


Coord = Tuple[int, int, int]


class JPS3D(GraphSearcher3D):
    """
    Jump Point Search for 3D grids.
    - Uses env.motions as direction set (typically 26-connected).
    - Tuple-based heap so Node3D doesn't need __lt__.
    """

    def __init__(self, start: Coord, goal: Coord, env: Grid3D, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)
        self.start.g = 0.0
        self.start.h = self.h(self.start, self.goal)

    def __str__(self) -> str:
        return "Jump Point Search (JPS) 3D"

    # ---------- public API ----------

    def plan(self) -> tuple:
        """
        Returns:
            cost (float), path (list[tuple]), expand (list[Node3D])
        """
        OPEN = []
        CLOSED = {}
        counter = itertools.count()

        heapq.heappush(OPEN, (self.start.g + self.start.h, self.start.h, next(counter), self.start))

        expansions = 0
        log_interval = 1000

        while OPEN:
            _, _, _, node = heapq.heappop(OPEN)

            if node.current in CLOSED and node.g >= CLOSED[node.current].g:
                continue

            CLOSED[node.current] = node
            expansions += 1

            if node == self.goal:
                logging.info(f"Goal reached after expanding {expansions} nodes")
                cost, path = self.extractPath(CLOSED)
                return cost, path, list(CLOSED.values())

            if expansions % log_interval == 0:
                logging.info(f"Expanded {expansions} nodes, OPEN size={len(OPEN)}")

            # JPS neighbor generation via jumping along motion directions
            jps_candidates = []
            for motion in self.motions:
                jp = self.jump(node, motion)
                if jp is None:
                    continue
                if jp.current in CLOSED and jp.g >= CLOSED[jp.current].g:
                    continue
                jps_candidates.append(jp)

            for jp in jps_candidates:
                f = jp.g + jp.h
                heapq.heappush(OPEN, (f, jp.h, next(counter), jp))

        logging.warning("No path found")
        return float("inf"), [], list(CLOSED.values())

    def run(self):
        cost, path, expand = self.plan()
        self.plot.animation(path, str(self), cost, expand)

    # ---------- core JPS ----------

    def jump(self, node: Node3D, motion: Node3D) -> Optional[Node3D]:
        """
        Recursively move from `node` in `motion` direction until:
          - we hit an obstacle (return None),
          - we reach the goal (return that node),
          - we encounter a forced neighbor (return that node),
          - otherwise continue.
        """
        dx, dy, dz = motion.current
        if dx == dy == dz == 0:
            return None

        cur = node
        while True:
            nxt = cur + motion  # type: ignore[operator]
            # out of known grid?
            if getattr(self.env, "grid_map", None) is not None and nxt.current not in self.env.grid_map:
                return None
            # blocked endpoint or segment collision
            if nxt.current in self.obstacles or self.isCollision(cur, nxt):
                return None

            # set scores incrementally (distance from `node`, not from start)
            # we still use CLOSED[node.current].g for final g when pushing;
            # but cache distance traveled inside the Node3D for convenience.
            # NOTE: we’ll recompute absolute g just before pushing to OPEN.
            # Here we set parent and heuristic.
            nxt.parent = cur.current
            nxt.h = self.h(nxt, self.goal)

            # goal?
            if nxt == self.goal:
                # absolute g will be computed in plan() caller
                return self._mk_qnode(node, nxt)

            # forced neighbor?
            if self._has_forced_neighbor(nxt.current, (dx, dy, dz)):
                return self._mk_qnode(node, nxt)

            # diagonal checks: like 2D JPS, but extended
            nz = (dx != 0) + (dy != 0) + (dz != 0)
            if nz >= 2:
                # axes projections
                if dx != 0 and self.jump(nxt, Node3D((dx, 0, 0), None, 1.0, 0.0)):
                    return self._mk_qnode(node, nxt)
                if dy != 0 and self.jump(nxt, Node3D((0, dy, 0), None, 1.0, 0.0)):
                    return self._mk_qnode(node, nxt)
                if dz != 0 and self.jump(nxt, Node3D((0, 0, dz), None, 1.0, 0.0)):
                    return self._mk_qnode(node, nxt)

                # plane-diagonal projections (helpful in 3D)
                if dx != 0 and dy != 0 and self.jump(nxt, Node3D((dx, dy, 0), None, 1.0, 0.0)):
                    return self._mk_qnode(node, nxt)
                if dx != 0 and dz != 0 and self.jump(nxt, Node3D((dx, 0, dz), None, 1.0, 0.0)):
                    return self._mk_qnode(node, nxt)
                if dy != 0 and dz != 0 and self.jump(nxt, Node3D((0, dy, dz), None, 1.0, 0.0)):
                    return self._mk_qnode(node, nxt)

            # continue stepping
            cur = nxt

    # helper to create queue node with absolute g based on original `from_node`
    def _mk_qnode(self, from_node: Node3D, at_node: Node3D) -> Node3D:
        g_new = from_node.g + self.dist(from_node, at_node)
        return Node3D(at_node.current, from_node.current, g_new, self.h(at_node, self.goal))

    # ---------- forced neighbor detection (3D) ----------

    def _has_forced_neighbor(self, c: Coord, d: Coord) -> bool:
        """
        3D forced-neighbor detection.
        Treats out-of-bounds as free (assumes walls as obstacles if desired).
        """
        x, y, z = c
        dx, dy, dz = d
        obs = self.obstacles

        def blocked(p: Coord) -> bool:
            return p in obs

        def free(p: Coord) -> bool:
            # if using explicit grid bounds, also require membership
            gm = getattr(self.env, "grid_map", None)
            if gm is not None and p not in gm:
                return False
            return p not in obs

        nz = (dx != 0) + (dy != 0) + (dz != 0)

        # Axis move: check 4 face-adjacent blockers that open after the step
        if nz == 1:
            if dx != 0:
                for oy in (-1, 1):
                    if blocked((x, y + oy, z)) and free((x + dx, y + oy, z)):
                        return True
                for oz in (-1, 1):
                    if blocked((x, y, z + oz)) and free((x + dx, y, z + oz)):
                        return True
            elif dy != 0:
                for ox in (-1, 1):
                    if blocked((x + ox, y, z)) and free((x + ox, y + dy, z)):
                        return True
                for oz in (-1, 1):
                    if blocked((x, y, z + oz)) and free((x, y + dy, z + oz)):
                        return True
            else:  # dz != 0
                for ox in (-1, 1):
                    if blocked((x + ox, y, z)) and free((x + ox, y, z + dz)):
                        return True
                for oy in (-1, 1):
                    if blocked((x, y + oy, z)) and free((x, y + oy, z + dz)):
                        return True
            return False

        # Plane-diagonal move: use 2D JPS rules in the plane of motion
        if nz == 2:
            if dx != 0 and dy != 0:
                if blocked((x - dx, y, z)) and free((x - dx, y + dy, z)):
                    return True
                if blocked((x, y - dy, z)) and free((x + dx, y - dy, z)):
                    return True
                return False
            if dx != 0 and dz != 0:
                if blocked((x - dx, y, z)) and free((x - dx, y, z + dz)):
                    return True
                if blocked((x, y, z - dz)) and free((x + dx, y, z - dz)):
                    return True
                return False
            if dy != 0 and dz != 0:
                if blocked((x, y - dy, z)) and free((x, y - dy, z + dz)):
                    return True
                if blocked((x, y, z - dz)) and free((x, y + dy, z - dz)):
                    return True
                return False

        # Space diagonal: combine plane checks
        if nz == 3:
            # (x-dx, y, z) -> (x-dx, y+dy, z)
            if blocked((x - dx, y, z)) and free((x - dx, y + dy, z)):
                return True
            # (x, y-dy, z) -> (x+dx, y-dy, z)
            if blocked((x, y - dy, z)) and free((x + dx, y - dy, z)):
                return True
            # (x, y, z-dz) -> (x+dx, y, z-dz)
            if blocked((x, y, z - dz)) and free((x + dx, y, z - dz)):
                return True
            return False

        return False

    # ---------- path reconstruction ----------

    def extractPath(self, closed_list: dict) -> tuple:
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
