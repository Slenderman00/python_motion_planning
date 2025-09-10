from python_motion_planning.global_planner.graph_search import AStar3D, DStar3D, Dijkstra3D, GBFS3D, JPS3D, LazyThetaStar3D, ThetaStar3D, VoronoiPlanner3D, LPAStar3D
from python_motion_planning.global_planner.evolutionary_search import ACO3D, PSO3D

algorithms = {
  "astar": AStar3D,
  "dstar": DStar3D,
  "dijkstra": Dijkstra3D,
  "gbfs": GBFS3D,
  "jps": JPS3D,
  "lazy_theta_star": LazyThetaStar3D,
  "theta_star": ThetaStar3D,
  "voronoi": VoronoiPlanner3D,
  "lpastar": LPAStar3D,
  "aco": ACO3D,
  "pso": PSO3D
}