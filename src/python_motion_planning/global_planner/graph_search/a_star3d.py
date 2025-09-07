"""
@file: a_star.py
@brief: A* motion planning (3D-safe)
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

class AStar3D(GraphSearcher3D):
    """
    A* for 3D grids/maps.
    Uses a tuple heap (f, h, tie, node) to avoid relying on Node3D.__lt__.
    """

    def __init__(self, start: tuple, goal: tuple, env: Grid3D, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)
        # Ensure start scores are initialized
        self.start.g = 0.0
        self.start.h = self.h(self.start, self.goal)

    def __str__(self) -> str:
        return "A*"

    def plan(self) -> tuple:
        OPEN = []
        CLOSED = {}
        counter = itertools.count()

        self.start.g = 0.0
        self.start.h = self.h(self.start, self.goal)
        heapq.heappush(OPEN, (self.start.g + self.start.h, self.start.h, next(counter), self.start))

        expansions = 0
        log_interval = 1000  # how often to log

        while OPEN:
            _, _, _, node = heapq.heappop(OPEN)

            best_closed = CLOSED.get(node.current)
            if best_closed is not None and node.g >= best_closed.g:
                continue

            CLOSED[node.current] = node
            expansions += 1

            # progress log
            if expansions % log_interval == 0:
                logging.info(f"Expanded {expansions} nodes, OPEN size={len(OPEN)}")

            # goal check
            if node == self.goal:
                logging.info(f"Goal reached after expanding {expansions} nodes")
                cost, path = self.extractPath(CLOSED)
                return cost, path, list(CLOSED.values())

            # expand neighbors
            for node_n in self.getNeighbor(node):
                tentative_g = node.g + self.dist(node, node_n)
                closed_n = CLOSED.get(node_n.current)
                if closed_n is not None and tentative_g >= closed_n.g:
                    continue

                node_n = Node3D(node_n.current, node_n.current, tentative_g, 0.0)
                node_n.parent = node.current
                node_n.h = self.h(node_n, self.goal)
                heapq.heappush(OPEN, (tentative_g + node_n.h, node_n.h, next(counter), node_n))

        logging.warning("No path found")
        return float("inf"), [], list(CLOSED.values())

    def getNeighbor(self, node: Node3D) -> list:
        """
        Generate valid 3D neighbors using the motion set from base class.
        Assumes self.motions contains 3D moves (6/18/26-connectivity).
        """
        return [
            node + motion
            for motion in self.motions
            if not self.isCollision(node, node + motion)
        ]

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

        path.reverse()  # start -> goal
        return cost, path

    def run(self):
        cost, path, expand = self.plan()
        self.plot.animation(path, str(self), cost, expand)

