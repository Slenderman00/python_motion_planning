"""
@file: lpa_star3d.py
@breif: Lifelong Planning A* motion planning
@author: Yang Haodong, Wu Maojia, Joar Heimonen
@update: 2025-09-10 
"""
import heapq
from typing import Dict, List, Optional, Tuple

from .graph_search_3d import GraphSearcher3D
from python_motion_planning.utils import Env3D, Node3D, Grid3D


class LNode3D(Node3D):
    """
    LPA* node for 3D grids. Coordinates live in .current (x,y,z).
    Fields:
        g   : current best cost-from-start (estimate)
        rhs : one-step lookahead cost
        key : 2-tuple priority [k1, k2] (list for compatibility)
    """
    __slots__ = ("current", "g", "rhs", "key", "parent")

    def __init__(self, current: tuple, g: float, rhs: float, key: Optional[list]) -> None:
        # Ensure base bookkeeping exists; parent/h are unused by LPA* here
        Node3D.__init__(self, current, None, g if g != float("inf") else float("inf"), 0.0)
        self.current = current
        self.g = g
        self.rhs = rhs
        self.key = key
        self.parent = None  # only used when extracting/visualizing a path

    def __add__(self, motion: Node3D) -> Node3D:
        cx, cy, cz = self.current
        dx, dy, dz = motion.current
        return Node3D((cx + dx, cy + dy, cz + dz), None, 0.0, 0.0)

    def __lt__(self, other) -> bool:
        # For heap ordering when keys tie-break
        return self.key < other.key

    def __repr__(self) -> str:
        return f"LNode3D(current={self.current}, g={self.g}, rhs={self.rhs}, key={self.key})"


class LPAStar3D(GraphSearcher3D):
    """
    Lifelong Planning A* on 3D voxel grids.

    Example:
        >>> planner = LPAStar3D((1,1,1), (18,13,9), grid_env)
        >>> cost, path, _ = planner.plan()
        >>> planner.run()
    """

    def __init__(self, start: tuple, goal: tuple, env: Grid3D, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)

        # Start and goal as LPA* nodes
        self.start = LNode3D(start, float("inf"), 0.0, None)
        self.goal = LNode3D(goal, float("inf"), float("inf"), None)

        # OPEN set (priority queue U) and expansion trace
        self.U: List[LNode3D] = []
        self.EXPAND: List[LNode3D] = []

        # Per-voxel bookkeeping map
        self.map: Dict[tuple, LNode3D] = {
            s: LNode3D(s, float("inf"), float("inf"), None) for s in self.env.grid_map
        }
        self.map[self.start.current] = self.start
        self.map[self.goal.current] = self.goal

        # Initialize OPEN with start
        self.start.key = self.calculateKey(self.start)
        heapq.heappush(self.U, self.start)

    def __str__(self) -> str:
        return "Lifelong Planning A* 3D"

    # -------- public API --------

    def plan(self) -> tuple:
        """Compute/repair shortest path and return (cost, path, None)."""
        self.computeShortestPath()
        cost, path = self.extractPath()
        return cost, path, None

    def run(self) -> None:
        cost, path, _ = self.plan()
        # guard empty paths to avoid viewer exceptions
        if not path:
            import logging
            logging.warning("LPAStar3D: no path found — skipping animation.")
            return
        self.plot.animation(path, str(self), cost, self.EXPAND)

    def apply_change(self, coord: tuple, blocked: Optional[bool] = None) -> tuple:
        """
        Toggle or set occupancy at `coord` and incrementally replan.

        blocked=None -> toggle
        blocked=True -> add obstacle
        blocked=False -> remove obstacle
        """
        self.EXPAND.clear()
        if blocked is None:
            if coord in self.obstacles:
                self.obstacles.remove(coord)
                # when freeing, immediately relax this node
                self.updateVertex(self.map[coord])
            else:
                self.obstacles.add(coord)
        elif blocked:
            self.obstacles.add(coord)
        else:
            if coord in self.obstacles:
                self.obstacles.remove(coord)
                self.updateVertex(self.map[coord])

        # update environment
        self.env.update(self.obstacles)

        # neighbors of changed node might need updates
        if coord in self.map:
            changed = self.map[coord]
            for n in self.getNeighbor(changed):
                self.updateVertex(n)

        # replan
        return self.plan()

    # -------- core LPA* --------

    def computeShortestPath(self) -> None:
        """Incrementally compute/repair the shortest path."""
        while self.U:
            node = min(self.U, key=lambda n: n.key)
            # stop when start's key >= goal's key and goal is consistent
            if node.key >= self.calculateKey(self.goal) and self.goal.rhs == self.goal.g:
                break

            # pop 'node' from OPEN
            self.U.remove(node)
            self.EXPAND.append(node)

            # over-consistent -> make consistent
            if node.g > node.rhs:
                node.g = node.rhs
            # under-consistent -> make over-consistent and propagate
            else:
                node.g = float("inf")
                self.updateVertex(node)

            for node_n in self.getNeighbor(node):
                self.updateVertex(node_n)

    def updateVertex(self, node: LNode3D) -> None:
        """Update rhs and membership of node in the OPEN set."""
        if node != self.start:
            neighs = self.getNeighbor(node)
            if neighs:
                node.rhs = min(n.g + self.cost(n, node) for n in neighs)
            else:
                node.rhs = float("inf")

        if node in self.U:
            self.U.remove(node)

        if node.g != node.rhs:
            node.key = self.calculateKey(node)
            heapq.heappush(self.U, node)

    def calculateKey(self, node: LNode3D) -> list:
        """Compute the 2-tuple priority key for LPA*."""
        m = min(node.g, node.rhs)
        return [m + self.h(node, self.goal), m]

    def getNeighbor(self, node: LNode3D) -> List[LNode3D]:
        """
        3D neighbors using the motion set. Filters:
          - in-bounds (via self.map)
          - endpoint not blocked
        (Edge collisions are accounted for when extracting the path.)
        """
        neighbors: List[LNode3D] = []
        cx, cy, cz = node.current
        for motion in self.motions:
            dx, dy, dz = motion.current
            nxt = (cx + dx, cy + dy, cz + dz)
            if nxt not in self.map:
                continue
            n = self.map[nxt]
            if n.current in self.obstacles:
                continue
            neighbors.append(n)
        return neighbors

    def extractPath(self) -> Tuple[float, List[tuple]]:
        """
        Greedy path extraction from goal back to start using lowest-g neighbor,
        skipping edges that collide. Returns (cost, path). If no path, returns (cost, []).
        """
        node = self.goal
        path = [node.current]
        cost = 0.0
        safety = 0

        while node != self.start:
            # pick best neighbor by g that is also collision-free along the edge
            candidates = [n for n in self.getNeighbor(node) if not self.isCollision(node, n)]
            if not candidates:
                return cost, []  # disconnected

            next_node = min(candidates, key=lambda n: n.g)
            cost += self.cost(node, next_node)
            node = next_node
            path.append(node.current)

            safety += 1
            if safety >= 100000:  # very conservative loop guard for huge grids
                return cost, []

        path.reverse()
        return cost, path
