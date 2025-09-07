"""
Plot tools 3D
@author: huiming zhou (3D refactor)
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import time

from ..environment.env3d import Env3D, Grid3D, Map3D, Node3D
from ..environment.env import Env, Grid, Map, Node


class Plot3D:
    def __init__(self, start, goal, env: Env3D):
        # Accept (x,y,z) tuples or Node3D for start/goal
        sx, sy, sz = self._xyz(start)
        gx, gy, gz = self._xyz(goal)
        self.start = Node3D((sx, sy, sz), (sx, sy, sz), 0, 0)
        self.goal = Node3D((gx, gy, gz), (gx, gy, gz), 0, 0)

        self.env = env
        self.fig = plt.figure("planning")
        # Keep a 2D axis for cost curves only (no scene drawing here)
        self.ax2d = self.fig.add_subplot(121) if plt.rcParams.get('figure.subplot.left', None) else None
        # Main 3D axis
        self.ax3d = self.fig.add_subplot(111, projection="3d")

        # ESC to quit, once
        self.fig.canvas.mpl_connect(
            'key_release_event',
            lambda event: (exit(0) if event.key == 'escape' else None)
        )

    # =========================
    # Public API
    # =========================
    def animation(
        self,
        path: list,
        name: str,
        cost: float = None,
        expand: list = None,
        history_pose: list = None,
        predict_path: list = None,
        lookahead_pts: list = None,
        cost_curve: list = None,
        ellipse: np.ndarray = None
    ) -> None:

        name = f"{name}\ncost: {cost}" if cost is not None else name
        self.plotEnv(name)

        if expand is not None:
            self.plotExpand(expand)

        if history_pose is not None:
            self.plotHistoryPose(history_pose, predict_path, lookahead_pts)

        if path is not None:
            self.plotPath(path)

        if cost_curve:
            plt.figure("cost curve")
            self.plotCostCurve(cost_curve, name)

        if ellipse is not None:
            self.plotEllipse(ellipse)

        plt.show()

    def plotEnv(self, name: str, draw_edge: bool = False) -> None:
        """
        Plot environment with static obstacles.
        """
        ax = self.ax3d
        ax.cla()

        # Start/Goal
        ax.scatter(self.start.x, self.start.y, self.start.z, marker="s", color="#ff0000", label="Start")
        ax.scatter(self.goal.x, self.goal.y, self.goal.z, marker="s", color="#1155cc", label="Goal")

        # Obstacles
        if isinstance(self.env, Grid3D):
            if draw_edge:
                obs_x = [x[0] for x in self.env.obstacles]
                obs_y = [x[1] for x in self.env.obstacles]
                obs_z = [x[2] for x in self.env.obstacles]
            else:
                obs_x, obs_y, obs_z = [], [], []
                for x, y, z in self.env.obstacles:
                    if (
                        x not in (0, self.env.x_range - 1) and
                        y not in (0, self.env.y_range - 1) and
                        z not in (0, self.env.z_range - 1)
                    ):
                        obs_x.append(x); obs_y.append(y); obs_z.append(z)
            if obs_x:
                ax.scatter(obs_x, obs_y, obs_z, c="k", marker="s", label="Obstacles")

            # Auto-bounds for better aspect
            ax.set_xlim(0, self.env.x_range)
            ax.set_ylim(0, self.env.y_range)
            ax.set_zlim(0, self.env.z_range)

        elif isinstance(self.env, Map3D):
            # Draw boundary and obstacles at z=0 as 3D polygons (wireframe fill)
            polys = []
            # boundary rectangles (ox, oy, w, h)
            for (ox, oy, w, h) in getattr(self.env, 'boundary', []):
                polys.append(self._rect3d(ox, oy, w, h, z=0))
            for (ox, oy, w, h) in getattr(self.env, 'obs_rect', []):
                polys.append(self._rect3d(ox, oy, w, h, z=0))
            for (ox, oy, r) in getattr(self.env, 'obs_circ', []):
                polys.append(self._circle3d(ox, oy, r, z=0, n=40))

            if polys:
                # Fill lightly + wireframe
                collection = Poly3DCollection(polys, facecolors="lightgray", edgecolors="black", alpha=0.5)
                ax.add_collection3d(collection)

            # Bounds heuristic
            xs = [v[0] for poly in polys for v in poly] if polys else [self.start.x, self.goal.x]
            ys = [v[1] for poly in polys for v in poly] if polys else [self.start.y, self.goal.y]
            ax.set_xlim(min(xs) - 1, max(xs) + 1)
            ax.set_ylim(min(ys) - 1, max(ys) + 1)
            # If env has a z range, use it; else infer from start/goal
            zmin = min(getattr(self.env, 'z_min', 0), self.start.z, self.goal.z)
            zmax = max(getattr(self.env, 'z_max', 0), self.start.z, self.goal.z) or 1
            ax.set_zlim(zmin, zmax)

        ax.set_title(name)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc="best")

        self._refresh()

    def plotExpand(self, expand: list) -> None:
        """
        Plot expanded nodes/cells as they are explored (animated).
        Accepts Node3D or (x,y,z).
        """
        ax = self.ax3d
        expands = [x for x in expand]

        # Remove start/goal if they appear in Node3D form or tuple form
        expands = [e for e in expands if self._xyz(e) != (self.start.x, self.start.y, self.start.z)]
        expands = [e for e in expands if self._xyz(e) != (self.goal.x, self.goal.y, self.goal.z)]

        total = len(expands)
        count = 0

        if isinstance(self.env, Grid3D):
            for e in expands:
                count += 1
                x, y, z = self._xyz(e)
                ax.plot([x], [y], [z], linestyle='None', marker='s', color="#dddddd")
                # Adaptive throttling
                if count < total / 3:
                    step = 20
                elif count < 2 * total / 3:
                    step = 30
                else:
                    step = 40
                if count % step == 0:
                    self._refresh(0.001)

        elif isinstance(self.env, Map3D):
            # Expect e.parent to be (px, py, pz)
            for e in expands:
                count += 1
                if getattr(e, "parent", None) is not None:
                    px, py, pz = e.parent
                    x, y, z = self._xyz(e)
                    ax.plot([px, x], [py, y], [pz, z], color="#dddddd", linestyle='-')
                    if count % 10 == 0:
                        self._refresh(0.001)

        self._refresh()

    def plotPath(self, path: list, path_color: str = '#13ae00', path_style: str = "-") -> None:
        """
        Plot a 3D path. Accepts list of Node3D or (x,y,z).
        """
        if not path:
            return
        xs, ys, zs = zip(*[self._xyz(p) for p in path])
        self.ax3d.plot(xs, ys, zs, path_style, linewidth=2, color=path_color)
        # Re-draw start/goal on top
        self.ax3d.scatter(self.start.x, self.start.y, self.start.z, marker="s", color="#ff0000")
        self.ax3d.scatter(self.goal.x, self.goal.y, self.goal.z, marker="s", color="#1155cc")
        self._refresh()

    def plotAgent(self, pose: tuple, radius: float = 1) -> None:
        """
        Plot agent in 3D at pose (x, y, z, theta), where theta is yaw in XY plane.
        If pose is length-3 (x,y,theta) it assumes z=0.
        """
        if len(pose) == 3:
            x, y, theta = pose
            z = 0.0
        else:
            x, y, z, theta = pose

        # Clean previous agent artists (arrow + sphere) if any
        # We'll tag them with a custom attribute
        for art in list(self.ax3d.lines) + list(self.ax3d.artists):
            if hasattr(art, "_is_agent"):
                try:
                    art.remove()
                except Exception:
                    pass
        for coll in list(self.ax3d.collections):
            if hasattr(coll, "_is_agent"):
                try:
                    coll.remove()
                except Exception:
                    pass

        # Agent center
        center = self.ax3d.scatter([x], [y], [z])
        center._is_agent = True  # tag

        # Heading arrow (unit in theta direction)
        dx, dy, dz = np.cos(theta), np.sin(theta), 0.0
        q = self.ax3d.quiver([x], [y], [z], [dx], [dy], [dz], length=radius, normalize=True)
        q._is_agent = True

        self._refresh()

    def plotHistoryPose(self, history_pose, predict_path=None, lookahead_pts=None) -> None:
        """
        history_pose: list of (x,y,z,theta) or (x,y,theta) where z defaults to 0
        predict_path: optional list of arrays with shape (N,3) or (N,2)->z=0
        lookahead_pts: optional list of (x,y,z) or (x,y)->z=0
        """
        lookahead_handle = None

        for i, pose in enumerate(history_pose):
            # draw segment to next pose
            if i < len(history_pose) - 1:
                x1, y1, z1 = self._xyz_pose(history_pose[i])
                x2, y2, z2 = self._xyz_pose(history_pose[i + 1])
                self.ax3d.plot([x1, x2], [y1, y2], [z1, z2], c="#13ae00")

                if predict_path is not None and i < len(predict_path):
                    arr = np.asarray(predict_path[i])
                    if arr.shape[1] == 2:
                        zs = np.zeros(arr.shape[0])
                        self.ax3d.plot(arr[:, 0], arr[:, 1], zs, c="#dddddd")
                    else:
                        self.ax3d.plot(arr[:, 0], arr[:, 1], arr[:, 2], c="#dddddd")

            # agent
            self.plotAgent(pose)

            # lookahead
            if lookahead_handle is not None:
                try:
                    lookahead_handle.remove()
                except Exception:
                    pass
            if lookahead_pts is not None:
                lp = lookahead_pts[i] if i < len(lookahead_pts) else lookahead_pts[-1]
                lx, ly, lz = self._xyz(lp)
                lookahead_handle = self.ax3d.scatter([lx], [ly], [lz], c="b")

            if i % 5 == 0:
                self._refresh(0.03)

        self._refresh()

    def plotCostCurve(self, cost_list: list, name: str) -> None:
        plt.clf()
        plt.title(name + " — cost")
        plt.plot(cost_list)
        plt.xlabel("epochs")
        plt.ylabel("cost value")
        plt.grid(True)

    def plotEllipse(self, ellipse: np.ndarray, color: str = 'darkorange', linestyle: str = '--', linewidth: float = 2, z: float = None):
        """
        ellipse: shape (2,N) or (3,N). If 2×N, draws at z (defaults to start.z).
        """
        if ellipse.shape[0] == 2:
            z = self.start.z if z is None else z
            xs, ys = ellipse[0, :], ellipse[1, :]
            zs = np.full_like(xs, z, dtype=float)
        else:
            xs, ys, zs = ellipse[0, :], ellipse[1, :], ellipse[2, :]

        self.ax3d.plot(xs, ys, zs, linestyle=linestyle, color=color, linewidth=linewidth)
        self._refresh()

    def connect(self, name: str, func) -> None:
        self.fig.canvas.mpl_connect(name, func)

    def clean(self):
        self.ax3d.cla()
        self._refresh()

    def update(self):
        self._refresh()

    # =========================
    # Helpers
    # =========================
    @staticmethod
    def _xyz(obj):
        """Return (x,y,z) from Node3D/Node/tuple/list."""
        if hasattr(obj, 'x') and hasattr(obj, 'y'):
            z = getattr(obj, 'z', 0.0)
            return float(obj.x), float(obj.y), float(z)
        # tuple/list
        if len(obj) == 2:
            return float(obj[0]), float(obj[1]), 0.0
        return float(obj[0]), float(obj[1]), float(obj[2])

    @staticmethod
    def _xyz_pose(pose):
        """Pose may be (x,y,theta) or (x,y,z,theta)."""
        if len(pose) == 3:
            x, y, th = pose
            return float(x), float(y), 0.0
        x, y, z, _ = pose
        return float(x), float(y), float(z)

    @staticmethod
    def _rect3d(ox, oy, w, h, z=0.0):
        return [(ox, oy, z),
                (ox + w, oy, z),
                (ox + w, oy + h, z),
                (ox, oy + h, z)]

    @staticmethod
    def _circle3d(cx, cy, r, z=0.0, n=36):
        t = np.linspace(0, 2*np.pi, n, endpoint=True)
        return [(cx + r*np.cos(tt), cy + r*np.sin(tt), z) for tt in t]

    def _refresh(self, sleep_s: float = 0.0):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        if sleep_s > 0:
            time.sleep(sleep_s)

