"""
@file: graph_search_3d.py
@breif: Base class for planner based on graph searching
@author: Winter, Joar Heimonen
@update: 2025.9.7
"""
import math
from python_motion_planning.utils import Env3D, Node3D, Planner3D, Grid3D


class GraphSearcher3D(Planner3D):
    """
    Base class for planner based on graph searching.

    Parameters:
        start (tuple): start point coordinate
        goal (tuple): goal point coordinate
        env (Grid): environment
        heuristic_type (str): heuristic function type
    """
    def __init__(self, start: tuple, goal: tuple, env: Grid, heuristic_type: str="euclidean") -> None:
        super().__init__(start, goal, env)
        # heuristic type
        self.heuristic_type = heuristic_type
        # allowed motions
        self.motions = self.env.motions
        # obstacles
        self.obstacles = self.env.obstacles

    def h(self, node: Node3D, goal: Node3D) -> float:
        """
        Calculate heuristic.

        Parameters:
            node (Node): current node
            goal (Node): goal node

        Returns:
            h (float): heuristic function value of node
        """

        dx = abs(goal.x - node.x)
        dy = abs(goal.y - node.y)
        dz = abs(goal.z - node.z)

        if self.heuristic_type == "manhattan":
            return dx + dy + dz
        elif self.heuristic_type == "euclidean":
            return math.sqrt(dx**2 + dy**2 + dz**2)

    def cost(self, node1: Node3D, node2: Node3D) -> float:
        """
        Calculate cost for this motion.

        Parameters:
            node1 (Node): node 1
            node2 (Node): node 2

        Returns:
            cost (float): cost of this motion
        """
        if self.isCollision(node1, node2):
            return float("inf")
        return self.dist(node1, node2)

    def isCollision(self, node1: Node3D, node2: Node3D) -> bool:
        """
        Judge collision when moving from node1 to node2.

        Parameters:
            node1 (Node): node 1
            node2 (Node): node 2

        Returns:
            collision (bool): True if collision exists else False
        """

        p1 = (node1.x, node1.y, node1.z)
        p2 = (node2.x, node2.y, node2.z)

        if p1 in self.obstacles or p2 in self.obstacles:
            return True

        x1, y1, z1 = p1
        x2, y2, z2 = p2
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1

        if max(abs(dx), abs(dy), abs(dz)) > 1:
            return False

        changes = (dx != 0) + (dy != 0) + (dz != 0)

        if changes <= 1:
            return False

        checks = []
        if changes == 2:
            if dx != 0 and dy != 0:
                checks += [(x1 + dx, y1, z1), (x1, y1 + dy, z1)]
            if dx != 0 and dz != 0:
                checks += [(x1 + dx, y1, z1), (x1, y1, z1 + dz)]
            if dy != 0 and dz != 0:
                checks += [(x1, y1 + dy, z1), (x1, y1, z1 + dz)]
        else:
            checks += [(x1 + dx, y1, z1), (x1, y1 + dy, z1), (x1, y1, z1 + dz)]

        return any(c in self.obstacles for c in checks)
