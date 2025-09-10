"""
@file: gbfs3d.py
@breif: Greedy Best First Search motion planning
@author: Yang Haodong, Wu Maojia, Joar Heimonen
@update: 2024.9.10
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

class GBFS3D(GraphSearcher3D):
    """
    Greedy Best First Search for 3D grids (prioritizes heuristic h only).
    """

    def __init__(self, start: tuple, goal: tuple, env: Grid3D, heuristic_type: str = "euclidean") -> None:
        super().__init__(start, goal, env, heuristic_type)
        # start node scores
        self.start.g = 0.0
        self.start.h = self.h(self.start, self.goal)

    def __str__(self) -> str:
        return "Greedy Best First Search (GBFS) 3D"

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

        # Priority = heuristic only
        heapq.heappush(OPEN, (self.start.h, next(counter), self.start))

        expansions = 0
        log_every = 1000

        while OPEN:
            _, _, node = heapq.heappop(OPEN)

            # already expanded
            if node.current in CLOSED:
                continue

            CLOSED[node.current] = node
            expansions += 1
            if expansions % log_every == 0:
                logging.info(f"Expanded {expansions} nodes, OPEN size={len(OPEN)}")

            # goal
            if node == self.goal:
                logging.info(f"Goal reached after expanding {expansions} nodes")
                cost, path = self.extractPath(CLOSED)
                return cost, path, list(CLOSED.values())

            # expand neighbors
            for nbr in self.getNeighbor(node):
                # optional explicit endpoint obstacle guard (keeps parity with your 2D version)
                if nbr.current in self.obstacles:
                    continue
                if nbr.current in CLOSED:
                    continue

                h_val = self.h(nbr, self.goal)
                qnode = Node3D(nbr.current, node.current, 0.0, h_val)  # g unused in GBFS
                heapq.heappush(OPEN, (h_val, next(counter), qnode))

        logging.warning("No path found")
        return float("inf"), [], list(CLOSED.values())

    def getNeighbor(self, node: Node3D) -> list:
        """
        Generate valid 3D neighbors using the motion set from the base class.
        Assumes self.motions contains 3D moves (6/18/26-connectivity).
        """
        return [
            node + motion
            for motion in self.motions
            if not self.isCollision(node, node + motion)
        ]

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

    def run(self):
        cost, path, expand = self.plan()
        self.plot.animation(path, str(self), cost, expand)
