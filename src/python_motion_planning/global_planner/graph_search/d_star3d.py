"""
@file: d_star_3d.py
@brief: Dynamic A* (D*) motion planning in 3D grids
@author: Yang Haodong, Wu Maojia, Joar Heimonen
@update: 2025.09.09
"""
from __future__ import annotations
import logging
from typing import Dict, Iterable, List, Optional, Tuple

from .graph_search_3d import GraphSearcher3D
from python_motion_planning.utils import Env3D, Node3D, Grid3D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

Coord3D = Tuple[int, int, int]


class DNode3D(Node3D):
    """
    D* node in 3D.

    Args:
        current: (x, y, z)
        parent: parent coordinate (or None)
        t: state label in {'NEW','OPEN','CLOSED'}
        h: current best cost-to-go (from goal to this node)
        k: minimum historical h during OPEN processing (D*'s priority key)
    """
    __slots__ = ("current", "parent", "t", "h", "k", "x", "y", "z")

    def __init__(self, current: Coord3D, parent: Optional[Coord3D], t: str, h: float, k: float) -> None:
        # Initialize Node3D so x,y,z props exist
        super().__init__(*current)
        self.current: Coord3D = current
        self.parent: Optional[Coord3D] = parent
        self.t: str = t
        self.h: float = h
        self.k: float = k

    def __add__(self, other: "DNode3D") -> "DNode3D":
        """Vector addition used for motion to neighbor coordinates."""
        nx, ny, nz = self.x + other.x, self.y + other.y, self.z + other.z
        # h/k/t/parent fields on this temp node are irrelevant; only .current is used.
        return DNode3D((nx, ny, nz), self.current, self.t, self.h, self.k)

    def __repr__(self) -> str:
        return (f"DNode3D(current={self.current}, parent={self.parent}, "
                f"t={self.t}, h={self.h:.3f}, k={self.k:.3f})")

    def __str__(self) -> str:
        return ("----------\n"
                f"current:{self.current}\nparent:{self.parent}\n"
                f"t:{self.t}\nh:{self.h}\nk:{self.k}\n"
                "----------")


class DStar3D(GraphSearcher3D):
    """
    Dynamic A* (D*) in 3D voxel grids.

    Args:
        start: (x, y, z)
        goal: (x, y, z)
        env: Grid3D

    Example:
        >>> import python_motion_planning as pmp
        >>> grid = pmp.Grid3D(51, 31, 21)  # x,y,z sizes
        >>> planner = pmp.DStar3D((5,5,5), (45,25,15), grid)
        >>> cost, path, _ = planner.plan()
    """

    def __init__(self, start: Coord3D, goal: Coord3D, env: Grid3D) -> None:
        super().__init__(start, goal, env, None)  # GraphSearcher3D sets env/obstacles/metrics
        self.start = DNode3D(start, None, "NEW", float("inf"), float("inf"))
        self.goal = DNode3D(goal, None, "NEW", 0.0, float("inf"))

        # Motions: convert base env motions (Node3D-like) into DNode3D with (dx,dy,dz) and cost in .h holder
        # We only use motion.current for coordinate deltas and motion.g for step cost via self.cost()
        self.motions: List[DNode3D] = [
            DNode3D(m.current, None, "NEW", getattr(m, "g", 0.0), 0.0)  # h carries step cost if needed
            for m in self.env.motions
        ]

        # OPEN (priority by .k) and an expansion trace
        self.OPEN: List[DNode3D] = []
        self.EXPAND: List[DNode3D] = []

        # Map of all voxels to D* bookkeeping nodes
        self.map: Dict[Coord3D, DNode3D] = {
            s: DNode3D(s, None, "NEW", float("inf"), float("inf")) for s in self.env.grid_map
        }
        self.map[self.goal.current] = self.goal
        self.map[self.start.current] = self.start

        # Seed OPEN with goal
        self.insert(self.goal, 0.0)

    def __str__(self) -> str:
        return "Dynamic A* (D*) 3D"

    # ---------- Public API ----------

    def plan(self) -> Tuple[float, List[Coord3D], None]:
        """Run static planning once from goal to start (classic D* init)."""
        while True:
            kmin = self.processState()
            if kmin < 0:  # OPEN empty
                break
            if self.start.t == "CLOSED":
                break
        cost, path = self.extractPath(self.map)
        return cost, path, None

    def apply_dynamic_obstacles(self, newly_blocked: Iterable[Coord3D]) -> Tuple[float, List[Coord3D]]:
        """
        Update environment with new blocked voxels and locally replan from the
        current start back-pointer chain toward the goal (D*'s modify loop).

        Returns updated (cost, path). Call this whenever the world changes.
        """
        # Update obstacle set/env
        for v in newly_blocked:
            self.obstacles.add(v)
        self.env.update(self.obstacles)

        # Walk the backpointers from start to goal, repairing as needed
        node = self.start
        self.EXPAND.clear()
        path: List[Coord3D] = []
        cost: float = 0.0

        while node != self.goal:
            if node.parent is None:
                # No known parent yet; force a repair by pushing this node
                self.modify(node, self.goal)
                break

            node_parent = self.map[node.parent]
            if self.isCollision(node, node_parent):
                self.modify(node, node_parent)
                continue

            path.append(node.current)
            cost += self.cost(node, node_parent)
            node = node_parent

        if node == self.goal:
            path.append(self.goal.current)

        return cost, path

    # ---------- Core D* methods ----------

    def extractPath(self, closed_list: Dict[Coord3D, DNode3D]) -> Tuple[float, List[Coord3D]]:
        """Follow backpointers from start to goal to produce path and cost."""
        cost = 0.0
        node = self.start
        path = [node.current]
        while node != self.goal:
            if node.parent is None:
                # No connection found yet, force the algorithm to continue
                break
            node_parent = closed_list[node.parent]
            cost += self.cost(node, node_parent)
            node = node_parent
            path.append(node.current)
        return cost, path

    def processState(self) -> float:
        """
        Pop the state with min k from OPEN and perform RAISE/LOWER processing.

        Returns:
            The new minimum k in OPEN after processing, or -1 if OPEN is empty.
        """
        node = self.min_state
        if node is None:
            return -1.0

        self.EXPAND.append(node)

        k_old = node.k
        self.delete(node)  # move from OPEN to CLOSED

        # RAISE: k_min < h[x] -> try to reduce h[x] via neighbors
        if k_old < node.h:
            for node_n in self.getNeighbor(node):
                if node_n.h <= k_old and node.h > node_n.h + self.cost(node, node_n):
                    node.parent = node_n.current
                    node.h = node_n.h + self.cost(node, node_n)

        # LOWER: k_min == h[x]
        if k_old == node.h:
            for node_n in self.getNeighbor(node):
                if (node_n.t == "NEW"
                    or (node_n.parent == node.current and node_n.h != node.h + self.cost(node, node_n))
                    or (node_n.parent != node.current and node_n.h > node.h + self.cost(node, node_n))):
                    node_n.parent = node.current
                    self.insert(node_n, node.h + self.cost(node, node_n))
        else:
            for node_n in self.getNeighbor(node):
                if (node_n.t == "NEW"
                    or (node_n.parent == node.current and node_n.h != node.h + self.cost(node, node_n))):
                    node_n.parent = node.current
                    self.insert(node_n, node.h + self.cost(node, node_n))
                else:
                    if (node_n.parent != node.current
                        and node_n.h > node.h + self.cost(node, node_n)):
                        # LOWER occurred in OPEN list; reinsert x
                        self.insert(node, node.h)
                    elif (node_n.parent != node.current
                          and node.h > node_n.h + self.cost(node, node_n)
                          and node_n.t == "CLOSED"
                          and node_n.h > k_old):
                        # LOWER occurred in CLOSED neighbor; reinsert neighbor
                        self.insert(node_n, node_n.h)

        return self.min_k if self.min_state is not None else -1.0

    @property
    def min_state(self) -> Optional[DNode3D]:
        """Node in OPEN with minimum k, or None if OPEN empty."""
        if not self.OPEN:
            return None
        return min(self.OPEN, key=lambda n: n.k)

    @property
    def min_k(self) -> float:
        """Minimum k in OPEN (assumes non-empty)."""
        return self.min_state.k  # type: ignore[union-attr]

    def insert(self, node: DNode3D, h_new: float) -> None:
        """
        Insert/update a node in OPEN with new h/k per D* rules.
        """
        if node.t == "NEW":
            node.k = h_new
        elif node.t == "OPEN":
            node.k = min(node.k, h_new)
        elif node.t == "CLOSED":
            node.k = min(node.h, h_new)
        node.h = h_new
        node.t = "OPEN"
        if node not in self.OPEN:
            self.OPEN.append(node)

    def delete(self, node: DNode3D) -> None:
        """Move node from OPEN to CLOSED."""
        if node.t == "OPEN":
            node.t = "CLOSED"
        if node in self.OPEN:
            self.OPEN.remove(node)

    def modify(self, node: DNode3D, node_parent: DNode3D) -> None:
        """
        Start processing from a node where a discrepancy was found (dynamic change).
        """
        if node.t == "CLOSED":
            self.insert(node, node_parent.h + self.cost(node, node_parent))
        while True:
            k_min = self.processState()
            if k_min < 0 or k_min >= node.h:
                break

    def getNeighbor(self, node: DNode3D) -> List[DNode3D]:
        """
        Collect valid, non-colliding 26-connected neighbors (or whatever `env.motions` defines).
        """
        neighbors: List[DNode3D] = []
        for motion in self.motions:
            nxt = (node.x + motion.x, node.y + motion.y, node.z + motion.z)
            # Bounds/known-voxel check
            if nxt not in self.map:
                continue
            n = self.map[nxt]
            # Collision check on the edge (node -> n) in 3D
            if not self.isCollision(node, n):
                neighbors.append(n)
        return neighbors
