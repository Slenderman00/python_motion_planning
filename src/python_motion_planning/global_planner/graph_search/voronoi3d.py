"""
@file: voronoi3d.py
@breif: Voronoi-based motion planning
@author: Yang Haodong, Wu Maojia, Joar Heimonen
@update: 2025-09-10 
"""
import heapq, math
import itertools
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.spatial import cKDTree, Voronoi

from .graph_search_3d import GraphSearcher3D
from python_motion_planning.utils import Env3D, Node3D, Grid3D


Coord3D = Tuple[int, int, int]


class VoronoiPlanner3D(GraphSearcher3D):
    """
    Voronoi-based planner for 3D voxel maps.

    Parameters:
        start (tuple): start coordinate (x, y, z)
        goal (tuple): goal coordinate (x, y, z)
        env (Grid3D): 3D environment
        heuristic_type (str): heuristic function type (e.g., "euclidean")
        n_knn (int): #neighbors to try from each sampled point
        max_edge_len (float): maximum edge length allowed for the roadmap
        inflation_r (float): minimum clearance from obstacles for edges

    Notes:
        - Uses scipy.spatial.Voronoi in 3D on obstacle points to get vertices (skeleton).
        - Adds start/goal to the sample set.
        - Builds a KNN roadmap with collision/clearance checks in 3D, then runs Dijkstra/A*.
    """

    def __init__(
        self,
        start: tuple,
        goal: tuple,
        env: Grid3D,
        heuristic_type: str = "euclidean",
        n_knn: int = 10,
        max_edge_len: float = 10.0,
        inflation_r: float = 1.0,
    ) -> None:
        super().__init__(start, goal, env, heuristic_type)
        self.n_knn = int(n_knn)
        self.max_edge_len = float(max_edge_len)
        self.inflation_r = float(inflation_r)

    def __str__(self) -> str:
        return "Voronoi-based Planner 3D"

    # ---------- Public API ----------

    def plan(self):
        """
        Build Voronoi roadmap in 3D, then search shortest path.

        Returns:
            cost (float): path cost
            path (list[tuple]): planning path
            expand (list[Node3D]): Voronoi-sampled nodes
        """
        # --- 1) Sample Voronoi vertices from obstacle set (3D) ---
        obs = list(self.env.obstacles)
        expand_points: List[Tuple[float, float, float]] = []

        if len(obs) >= 5:
            try:
                vor = Voronoi(np.asarray(obs, dtype=float))
                # Filter vertices to stay in-bounds if grid dimensions are available
                gm = getattr(self.env, "grid_map", None)
                xr = getattr(self.env, "x_range", None)
                yr = getattr(self.env, "y_range", None)
                zr = getattr(self.env, "z_range", None)

                def in_bounds(p: Tuple[float, float, float]) -> bool:
                    if xr is not None and (p[0] < 0 or p[0] >= xr): return False
                    if yr is not None and (p[1] < 0 or p[1] >= yr): return False
                    if zr is not None and (p[2] < 0 or p[2] >= zr): return False
                    if gm is not None and (int(round(p[0])), int(round(p[1])), int(round(p[2]))) not in gm:
                        # If grid_map enumerates discrete cells, require membership
                        return False
                    return True

                for v in vor.vertices:
                    p = (float(v[0]), float(v[1]), float(v[2]))
                    if in_bounds(p):
                        expand_points.append(p)
            except Exception:
                # Voronoi may fail with degenerate inputs; fall back to empty set
                expand_points = []

        # Always include start/goal
        expand_points.append(tuple(map(float, self.start.current)))
        expand_points.append(tuple(map(float, self.goal.current)))

        # Deduplicate
        expand_points = list(dict.fromkeys(expand_points))

        # Create Node3D list for visualization ("expand")
        expand = [Node3D(p, None, 0.0, 0.0) for p in expand_points]

        # If we barely have samples, try direct segment start->goal
        if len(expand_points) < 3:
            road_map = self._connect_direct_only(expand_points)
            cost, path = self.getShortestPath(road_map, dijkstra=True)
            return cost, path, expand

        # --- 2) Build KNN roadmap with collision & clearance checks ---
        pts = np.asarray(expand_points, dtype=float)
        tree = cKDTree(pts)

        # adjacency by coordinate tuple
        road_map: Dict[Tuple[float, float, float], List[Tuple[float, float, float]]] = {
            tuple(p): [] for p in expand_points
        }

        k = min(self.n_knn + 1, len(expand_points))  # +1 for the point itself
        for i, p in enumerate(expand_points):
            dists, idxs = tree.query(p, k=k)
            # Ensure iterable behavior when k == 1
            if np.isscalar(dists):
                dists = [dists]
                idxs = [idxs]

            for j, q_idx in enumerate(idxs):
                if j == 0:
                    continue  # skip self
                q = expand_points[q_idx]
                # Edge length and collision checks
                if not self._edge_feasible(p, q):
                    continue
                road_map[tuple(p)].append(tuple(q))
                road_map[tuple(q)].append(tuple(p))  # undirected

        # --- 3) Shortest path on the roadmap ---
        cost, path = self.getShortestPath(road_map, dijkstra=True)
        return cost, path, expand

    def run(self):
        cost, path, expand = self.plan()
        self.plot.animation(path, str(self), cost, expand)

    # ---------- Roadmap building helpers ----------

    def _edge_len(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
        ax, ay, az = a
        bx, by, bz = b
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2)

    def _edge_feasible(self, a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
        """
        3D collision/clearance test along segment a->b with max length constraint.
        Uses env.obstacles_tree (KDTree) if present; otherwise falls back to base isCollision.
        """
        # length
        L = self._edge_len(a, b)
        if L == 0.0 or L > self.max_edge_len:
            return False

        a_int = (int(round(a[0])), int(round(a[1])), int(round(a[2])))
        b_int = (int(round(b[0])), int(round(b[1])), int(round(b[2])))

        # endpoints blocked?
        if a_int in self.obstacles or b_int in self.obstacles:
            return False

        # if no KDTree available or inflation <= 0, defer to base collision
        obstacles_tree = getattr(self.env, "obstacles_tree", None)
        if obstacles_tree is None or self.inflation_r <= 0.0:
            n1, n2 = Node3D(a, None, 0.0, 0.0), Node3D(b, None, 0.0, 0.0)
            return not self.isCollision(n1, n2)

        # Clearance sampling along the edge
        step = max(self.inflation_r, 1e-6)
        n_steps = max(1, int(math.ceil(L / step)))
        ux = (b[0] - a[0]) / L
        uy = (b[1] - a[1]) / L
        uz = (b[2] - a[2]) / L

        x, y, z = a
        for _ in range(n_steps):
            dist_to_obs, _ = obstacles_tree.query([x, y, z])
            if dist_to_obs <= self.inflation_r:
                return False
            x += step * ux
            y += step * uy
            z += step * uz

        # Check goal endpoint explicitly
        dist_to_obs, _ = obstacles_tree.query([b[0], b[1], b[2]])
        if dist_to_obs <= self.inflation_r:
            return False

        return True

    def _connect_direct_only(self, pts: List[Tuple[float, float, float]]):
        """Roadmap that tries only the direct edge start<->goal when Voronoi is unavailable."""
        road_map: Dict[Tuple[float, float, float], List[Tuple[float, float, float]]] = {
            tuple(p): [] for p in pts
        }
        if len(pts) >= 2:
            a, b = pts[-2], pts[-1]  # start, goal appended last in plan()
            if self._edge_feasible(a, b):
                road_map[tuple(a)].append(tuple(b))
                road_map[tuple(b)].append(tuple(a))
        return road_map

    # ---------- Shortest path on the roadmap ----------

    def getShortestPath(self, road_map: dict, dijkstra: bool = True) -> list:
        """
        Graph search over the roadmap.
        road_map: dict[Coord(float,float,float)] -> list[Coord(...)]
        dijkstra: True => Dijkstra, False => A*

        Returns:
            cost (float), path (list[Coord3D])
        """
        if not road_map:
            return float("inf"), []

        start = tuple(map(float, self.start.current))
        goal = tuple(map(float, self.goal.current))

        # OPEN is a heap of (priority, h, tie, coord, g)
        OPEN: List[Tuple[float, float, int, Tuple[float, float, float], float]] = []
        CLOSED: Dict[Tuple[float, float, float], float] = {}
        PARENT: Dict[Tuple[float, float, float], Optional[Tuple[float, float, float]]] = {}
        counter = itertools.count()

        # heuristic helper
        def h_of(p: Tuple[float, float, float]) -> float:
            if dijkstra:
                return 0.0
            # Use base heuristic with a temporary Node3D
            return self.h(Node3D(p, None, 0.0, 0.0), self.goal)

        h0 = h_of(start)
        heapq.heappush(OPEN, (h0, h0, next(counter), start, 0.0))
        PARENT[start] = None

        while OPEN:
            _, _, _, u, g_u = heapq.heappop(OPEN)

            # dominance check
            best_g = CLOSED.get(u)
            if best_g is not None and g_u >= best_g:
                continue

            CLOSED[u] = g_u

            if u == goal:
                # reconstruct path (round coords to int grid cells for path)
                path = []
                cur = u
                while cur is not None:
                    path.append(tuple(int(round(c)) for c in cur))
                    cur = PARENT[cur]
                path.reverse()
                return g_u, path

            for v in road_map.get(u, []):
                w = self._edge_len(u, v)
                g_v = g_u + w
                hv = h_of(v)
                f_v = g_v + hv
                best_v = CLOSED.get(v)
                if best_v is not None and g_v >= best_v:
                    continue
                PARENT[v] = u
                heapq.heappush(OPEN, (f_v, hv, next(counter), v, g_v))

        return float("inf"), []
