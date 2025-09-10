"""
@file: aco3d.py
@breif: Ant Colony Optimization (ACO) motion planning in 3D (bounded & robust pheromones)
@author: Yang Haodong, Wu Maojia, Joar Heimonen
@update: 2025.9.10
"""
import random
from bisect import bisect_left

from .evolutionary_search3d import EvolutionarySearcher3D
from python_motion_planning.utils import Env3D, Node3D, Grid3D


class ACO3D(EvolutionarySearcher3D):
    """
    Ant Colony Optimization (ACO) motion planning in 3D.
    """

    def __init__(
        self,
        start: tuple,
        goal: tuple,
        env: Grid3D,
        heuristic_type: str = "euclidean",
        n_ants: int = 50,
        alpha: float = 1.0,
        beta: float = 5.0,
        rho: float = 0.1,
        Q: float = 1.0,
        max_iter: int = 100,
    ) -> None:
        super().__init__(start, goal, env, heuristic_type)
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.max_iter = max_iter

    def __str__(self) -> str:
        return "Ant Colony Optimization 3D (ACO3D)"

    class Ant:
        def __init__(self) -> None:
            self.reset()

        def reset(self) -> None:
            self.found_goal = False
            self.current_node = None
            self.path = []
            self.path_set = set()
            self.steps = 0

    # ---- helpers ----
    def _in_bounds(self, coord: tuple) -> bool:
        x, y, z = coord
        return (
            0 <= x < self.env.x_range
            and 0 <= y < self.env.y_range
            and 0 <= z < self.env.z_range
        )

    def plan(self) -> tuple:
        """
        Ant Colony Optimization (ACO) motion plan function for 3D.

        Returns:
            cost (float): path cost
            path (list[tuple]): planning path as a list of (x, y, z)
            cost_list (list[float]): best path length over iterations
        """
        best_length_list, best_path = [], None

        # --- Pheromone initialization on all feasible local edges (bounded) ---
        pheromone_edges = {}
        for x in range(self.env.x_range):
            for y in range(self.env.y_range):
                for z in range(self.env.z_range):
                    if (x, y, z) in self.obstacles:
                        continue
                    cur_node = Node3D((x, y, z), (x, y, z), 0, 0)
                    for node_n in self.getNeighbor(cur_node):
                        pheromone_edges[(cur_node, node_n)] = 1.0

        # --- Heuristic max steps (scaled for 3D volume) ---
        vol = self.env.x_range * self.env.y_range * self.env.z_range
        max_dim = max(self.env.x_range, self.env.y_range, self.env.z_range)
        max_steps = vol / 3 + max_dim

        # --- Main loop ---
        cost_list = []
        for _ in range(self.max_iter):
            ants_list = []

            for _ in range(self.n_ants):
                ant = self.Ant()
                ant.current_node = self.start

                while ant.current_node is not self.goal and ant.steps < max_steps:
                    ant.path.append(ant.current_node)
                    ant.path_set.add(ant.current_node.current)

                    # Build candidate moves
                    prob_sum = 0.0
                    next_positions, next_probabilities = [], []

                    for node_n in self.getNeighbor(ant.current_node):
                        # avoid revisits
                        if node_n.current in ant.path_set:
                            continue

                        node_n.parent = ant.current_node.current

                        # goal found
                        if node_n == self.goal:
                            ant.path.append(node_n)
                            ant.path_set.add(node_n.current)
                            ant.found_goal = True
                            break

                        next_positions.append(node_n)
                        # pheromone * heuristic (inverse distance), robust lookup
                        tau = pheromone_edges.get((ant.current_node, node_n), 1.0)
                        prob_new = tau ** self.alpha * (1.0 / self.h(node_n, self.goal)) ** self.beta
                        next_probabilities.append(prob_new)
                        prob_sum += prob_new

                    if prob_sum == 0 or ant.found_goal:
                        break

                    # Roulette wheel selection
                    next_probabilities = [p / prob_sum for p in next_probabilities]
                    cumulative, cp = 0.0, []
                    for p in next_probabilities:
                        cumulative += p
                        cp.append(cumulative)
                    ant.current_node = next_positions[bisect_left(cp, random.random())]
                    ant.steps += 1

                ants_list.append(ant)

            # --- Pheromone evaporation ---
            for key in list(pheromone_edges.keys()):
                pheromone_edges[key] *= (1 - self.rho)

            # --- Pheromone deposition by successful ants ---
            bpl, bp = float("inf"), None
            for ant in ants_list:
                if ant.found_goal:
                    if len(ant.path) < bpl:
                        bpl, bp = len(ant.path), ant.path
                    delta = self.Q / len(ant.path)
                    for i in range(len(ant.path) - 1):
                        key = (ant.path[i], ant.path[i + 1])
                        pheromone_edges[key] = pheromone_edges.get(key, 1.0) + delta

            if bpl < float("inf"):
                best_length_list.append(bpl)

            if best_length_list:
                cost_list.append(min(best_length_list))
                if bpl <= min(best_length_list):
                    best_path = bp

        if best_path:
            cost = 0.0
            path = [self.start.current]
            for i in range(len(best_path) - 1):
                cost += self.dist(best_path[i], best_path[i + 1])
                path.append(best_path[i + 1].current)
            return cost, path, cost_list

        return [], [], []

    def getNeighbor(self, node: Node3D) -> list:
        """
        Find neighbors of a 3D node, clamped to grid bounds and collision-checked.
        """
        nbrs = []
        for motion in self.motions:
            cand = node + motion
            # Bounds check first to prevent stepping outside the grid
            if not self._in_bounds(cand.current):
                continue
            # Collision / corner-cutting check (from EvolutionarySearcher3D/GraphSearcher3D)
            if not self.isCollision(node, cand):
                nbrs.append(cand)
        return nbrs

    def run(self) -> None:
        """
        Run both planning and animation.
        """
        cost, path, cost_list = self.plan()
        self.plot.animation(path, str(self), cost, cost_curve=cost_list)
