"""
@file: dijkstra3d.py
@brief: Dijkstra motion planning (3D-safe)
"""
import heapq
import itertools
import logging

from .graph_search_3d import GraphSearcher3D
from python_motion_planning.utils import Env3D, Grid3D, Node3D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

class Dijkstra3D(GraphSearcher3D):
    """
    Dijkstra for 3D grids/maps (i.e., A* with heuristic = 0).

    Example:
        >>> grid = Grid3D(51, 31, 21)
        >>> planner = Dijkstra3D((5,5,5), (45,25,15), grid)
        >>> cost, path, expand = planner.plan()
        >>> planner.plot.animation(path, str(planner), cost, expand)
    """

    def __init__(self, start: tuple, goal: tuple, env: Grid3D) -> None:
        # No heuristic needed for Dijkstra, pass None
        super().__init__(start, goal, env, None)
        # Initialize start scores
        self.start.g = 0.0
        self.start.h = 0.0

    def __str__(self) -> str:
        return "Dijkstra (3D)"

    def plan(self) -> tuple:
        """
        Returns:
            cost (float): path cost
            path (list[tuple]): start->goal path
            expand (list[Node3D]): nodes expanded (for visualization)
        """
        OPEN = []
        CLOSED = {}
        counter = itertools.count()

        # push start with priority = g (since h = 0)
        heapq.heappush(OPEN, (self.start.g, 0.0, next(counter), self.start))

        expansions = 0
        log_interval = 1000

        while OPEN:
            _, _, _, node = heapq.heappop(OPEN)

            best_closed = CLOSED.get(node.current)
            if best_closed is not None and node.g >= best_closed.g:
                continue

            CLOSED[node.current] = node
            expansions += 1

            if expansions % log_interval == 0:
                logging.info(f"Expanded {expansions} nodes, OPEN size={len(OPEN)}")

            # goal check
            if node == self.goal:
                logging.info(f"Goal reached after expanding {expansions} nodes")
                cost, path = self.extractPath(CLOSED)
                return cost, path, list(CLOSED.values())

            # expand neighbors (26-connected if your env provides it)
            for node_n in self.getNeighbor(node):
                tentative_g = node.g + self.dist(node, node_n)
                closed_n = CLOSED.get(node_n.current)
                if closed_n is not None and tentative_g >= closed_n.g:
                    continue

                # Create/update a light Node3D for the queue
                node_q = Node3D(node_n.current, node.current, tentative_g, 0.0)  # h=0 for Dijkstra
                heapq.heappush(OPEN, (tentative_g, 0.0, next(counter), node_q))

        logging.warning("No path found")
        return float("inf"), [], list(CLOSED.values())

    def getNeighbor(self, node: Node3D) -> list:
        """
        Robust 3D neighbors:
        - in-bounds
        - endpoint not an obstacle
        - no diagonal corner-cutting through blocked faces
        - edge not colliding (ray/segment check)
        """
        neighbors = []
        cx, cy, cz = node.current
        grid_map = getattr(self.env, "grid_map", None)

        for motion in self.motions:
            cand = node + motion
            nx, ny, nz = cand.current

            # 1) bounds
            if grid_map is not None and (nx, ny, nz) not in grid_map:
                continue

            # 2) endpoint blocked
            if (nx, ny, nz) in self.obstacles:
                continue

            # 3) prevent corner-cutting through faces
            dx, dy, dz = nx - cx, ny - cy, nz - cz
            if (dx != 0 and (cx + dx, cy, cz) in self.obstacles) \
            or (dy != 0 and (cx, cy + dy, cz) in self.obstacles) \
            or (dz != 0 and (cx, cy, cz + dz) in self.obstacles):
                continue

            # 4) segment collision (keeps your env's finer checks)
            if self.isCollision(node, cand):
                continue

            neighbors.append(cand)

        return neighbors

    def extractPath(self, closed_list: dict) -> tuple:
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

    def run(self):
        cost, path, expand = self.plan()
        self.plot.animation(path, str(self), cost, expand)
