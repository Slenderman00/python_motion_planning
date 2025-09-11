import time
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

from ..environment.env3d import Env3D, Grid3D, Map3D, Node3D
from ..environment.env import Env, Grid, Map, Node

pv.global_theme.allow_empty_mesh = True

class Plot3D:
    def __init__(self, start, goal, env: Env3D):
        sx, sy, sz = self._xyz(start)
        gx, gy, gz = self._xyz(goal)
        self.start = Node3D((sx, sy, sz), (sx, sy, sz), 0, 0)
        self.goal  = Node3D((gx, gy, gz), (gx, gy, gz), 0, 0)
        self.env   = env

        self.pl = pv.Plotter(window_size=(1200, 850))
        self.pl.enable_trackball_style()
        self.pl.add_key_event('Escape', lambda: self.pl.close())
        self.pl.add_key_event('r', self._replay_hotkey)
        self.pl.add_key_event('R', self._replay_hotkey)
        self._shown = False

        self._obs_actor = None
        self._obs_mesh  = None

        self._expand_pts_actor = None
        self._expand_pts_mesh  = None
        self._expand_pts_list  = []

        self._expand_lines_actor = None
        self._expand_lines_poly  = None

        self._path_actor = None
        self._agent_sphere_actor = None
        self._agent_arrow_actor  = None

        self.hide_outer_walls = True
        self.outer_shell_thickness = 1
        self.clip_obstacles_to_path = True
        self.clip_margin = 2
        self.sense_color   = "#ff8800"
        self.sense_size    = 5
        self.sense_opacity = 0.9

        self.batch_size = 10  # draw 10 dots/segments per cycle

        self._last_anim = None

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
        ellipse: np.ndarray = None,
        block_at_end: bool = True,
        total_time_sec: float = None,  # ignored
    ) -> None:
        title = f"{name}\ncost: {cost}" if cost is not None else name

        self._last_anim = dict(
            path=path, title=title, expand=expand,
            history_pose=history_pose, predict_path=predict_path, lookahead_pts=lookahead_pts,
            cost_curve=cost_curve, ellipse=ellipse
        )

        self.plotEnv(title)

        if expand is not None:
            self._animate_expand_batched(expand, delay_sec=0.5)

        if history_pose is not None:
            self._draw_history_instant(history_pose, predict_path, lookahead_pts)

        if path is not None:
            self.plotPath(path)
            if self.clip_obstacles_to_path and isinstance(self.env, Grid3D):
                self._clip_and_rerender_obstacles_around_path(path, margin=self.clip_margin)

        if cost_curve:
            plt.figure("cost curve")
            self.plotCostCurve(cost_curve, title)

        if ellipse is not None:
            self.plotEllipse(ellipse)

        self.pl.render()

    def plotEnv(self, name: str, draw_edge: bool = False) -> None:
        self._ensure_window()

        self.pl.clear()
        self._obs_actor = None
        self._expand_pts_actor = None
        self._expand_pts_mesh  = None
        self._expand_pts_list  = []
        self._expand_lines_actor = None
        self._expand_lines_poly  = None
        self._path_actor = None
        self._agent_sphere_actor = None
        self._agent_arrow_actor  = None

        self.pl.add_text(name, font_size=12)
        self.pl.add_mesh(pv.Sphere(radius=0.6, center=(self.start.x, self.start.y, self.start.z)), color="red", name="start")
        self.pl.add_mesh(pv.Sphere(radius=0.6, center=(self.goal.x,  self.goal.y,  self.goal.z)),  color="#1155cc", name="goal")

        if isinstance(self.env, Grid3D):
            self._build_and_render_grid_obstacles()
            self.pl.view_isometric()
            self.pl.show_bounds(grid=True, location='outer', all_edges=True)
            self.pl.reset_camera()
        elif isinstance(self.env, Map3D):
            polys = []
            for (ox, oy, w, h) in getattr(self.env, 'boundary', []):
                polys.append(self._rect3d(ox, oy, w, h, z=0))
            for (ox, oy, w, h) in getattr(self.env, 'obs_rect', []):
                polys.append(self._rect3d(ox, oy, w, h, z=0))
            for (ox, oy, r) in getattr(self.env, 'obs_circ', []):
                polys.append(self._circle3d(ox, oy, r, z=0, n=48))
            for poly in polys:
                surf = pv.PolyData(np.array(poly)).delaunay_2d()
                self.pl.add_mesh(surf, color="lightgray", opacity=0.4, backface_culling=True)
            self.pl.view_isometric()
            self.pl.show_bounds(grid=True, location='outer', all_edges=True)
            self.pl.reset_camera()

        self.pl.render()
        self._poll()

    def plotExpand(self, expand: list, sleep_per_flush: float = 0.0) -> None:
        pass

    def plotPath(self, path: list, path_color: str = '#13ae00', path_style: str = "-") -> None:
        if not path:
            return
        pts = np.array([self._xyz(p) for p in path], dtype=float)
        poly = self._polyline(pts, close=False)
        if self._path_actor is not None:
            try: self.pl.remove_actor(self._path_actor)
            except Exception: pass
        self._path_actor = self.pl.add_mesh(poly, color=path_color, line_width=4)
        self.pl.render()

    def plotAgent(self, pose: tuple, radius: float = 1) -> None:
        if len(pose) == 3:
            x, y, theta = pose
            z = 0.0
        else:
            x, y, z, theta = pose

        for actor in (self._agent_sphere_actor, self._agent_arrow_actor):
            if actor is not None:
                try: self.pl.remove_actor(actor)
                except Exception: pass

        self._agent_sphere_actor = self.pl.add_mesh(pv.Sphere(radius=radius*0.6, center=(x, y, z)), color="red")
        dx, dy, dz = np.cos(theta), np.sin(theta), 0.0
        self._agent_arrow_actor = self.pl.add_mesh(
            pv.Arrow(start=(x, y, z), direction=(dx,dy,dz),
                     tip_length=0.3, tip_radius=0.1*radius, shaft_radius=0.05*radius),
            color="red"
        )
        self.pl.render()

    def plotHistoryPose(self, history_pose, predict_path=None, lookahead_pts=None, sleep_per_flush: float = 0.0) -> None:
        pass

    def plotCostCurve(self, cost_list: list, name: str) -> None:
        plt.clf()
        plt.title(name + " — cost")
        plt.plot(cost_list)
        plt.xlabel("epochs")
        plt.ylabel("cost value")
        plt.grid(True)

    def plotEllipse(self, ellipse: np.ndarray, color: str = 'darkorange', linestyle: str = '--', linewidth: float = 2, z: float = None):
        if ellipse.shape[0] == 2:
            z = self.start.z if z is None else z
            xs, ys = ellipse[0, :], ellipse[1, :]
            zs = np.full_like(xs, z, dtype=float)
        else:
            xs, ys, zs = ellipse[0, :], ellipse[1, :], ellipse[2, :]
        pts = np.vstack([xs, ys, zs]).T
        poly = self._polyline(pts, close=True)
        self.pl.add_mesh(poly, color=color, line_width=2)
        self.pl.render()

    def connect(self, key_name: str, func) -> None:
        self.pl.add_key_event(key_name, func)

    def clean(self):
        self.pl.clear()
        self.pl.render()

    def update(self):
        self.pl.render()

    # -------- batched, per-cycle expand animation (10 items per cycle) --------
    def _sleep_with_poll(self, seconds: float):
        end = time.perf_counter() + float(seconds)
        while time.perf_counter() < end and self._window_alive():
            self._poll()
            time.sleep(0.01)

    def _animate_expand_batched(self, expand: list, delay_sec: float = 0.5) -> None:
        expands = [e for e in expand
                   if self._xyz(e) != (self.start.x, self.start.y, self.start.z)
                   and self._xyz(e) != (self.goal.x,  self.goal.y,  self.goal.z)]

        if isinstance(self.env, Grid3D):
            self._expand_pts_list = []
            for i in range(0, len(expands), self.batch_size):
                batch = expands[i:i+self.batch_size]
                for e in batch:
                    self._expand_pts_list.append(self._xyz(e))

                if self._expand_pts_actor is not None:
                    try:
                        self.pl.remove_actor(self._expand_pts_actor)
                    except Exception:
                        pass
                    self._expand_pts_actor = None
                    self._expand_pts_mesh = None

                arr = np.array(self._expand_pts_list, dtype=float)
                self._expand_pts_mesh = pv.PolyData(arr)
                self._expand_pts_actor = self.pl.add_points(
                    self._expand_pts_mesh,
                    color=self.sense_color,
                    point_size=self.sense_size,
                    render_points_as_spheres=False,
                    opacity=self.sense_opacity,
                )
                self.pl.render()
                self._sleep_with_poll(delay_sec)

        elif isinstance(self.env, Map3D):
            pts_accum = []
            conn_accum = []
            base = 0
            segs = []
            for e in expands:
                if getattr(e, "parent", None) is None:
                    continue
                px, py, pz = e.parent
                x, y, z = self._xyz(e)
                segs.append(((px, py, pz), (x, y, z)))

            for i in range(0, len(segs), self.batch_size):
                batch = segs[i:i+self.batch_size]
                for (a, b) in batch:
                    pts_accum.extend([a, b])
                    conn_accum.extend([2, base, base + 1])
                    base += 2

                if self._expand_lines_actor is not None:
                    try:
                        self.pl.remove_actor(self._expand_lines_actor)
                    except Exception:
                        pass
                    self._expand_lines_actor = None
                    self._expand_lines_poly = None

                pts_np = np.array(pts_accum, dtype=float)
                conn_np = np.array(conn_accum, dtype=np.int64)
                poly = pv.PolyData(pts_np)
                poly.lines = conn_np
                self._expand_lines_poly = poly
                self._expand_lines_actor = self.pl.add_mesh(
                    poly, color=self.sense_color, line_width=1, render_lines_as_tubes=False
                )
                self.pl.render()
                self._sleep_with_poll(delay_sec)

    def _draw_history_instant(self, history_pose, predict_path=None, lookahead_pts=None):
        last_pt = None
        for i, pose in enumerate(history_pose):
            if len(pose) == 3:
                x, y, th = pose; z = 0.0
            else:
                x, y, z, th = pose
            cur = np.array([x, y, z], float)
            if last_pt is not None:
                seg = self._polyline(np.vstack([last_pt, cur]))
                self.pl.add_mesh(seg, color="#13ae00", line_width=2)
            for actor in (self._agent_sphere_actor, self._agent_arrow_actor):
                if actor is not None:
                    try: self.pl.remove_actor(actor)
                    except Exception: pass
            self.plotAgent(pose)
            if lookahead_pts is not None:
                lp = lookahead_pts[i] if i < len(lookahead_pts) else lookahead_pts[-1]
                lx, ly, lz = self._xyz(lp)
                self.pl.add_mesh(pv.Sphere(radius=0.4, center=(lx, ly, lz)), color="blue")
            last_pt = cur
        self.pl.render()

    # ------------------------ Obstacles (Grid3D) ------------------------
    def _build_and_render_grid_obstacles(self, bounds=None):
        xr, yr, zr = self.env.x_range, self.env.y_range, self.env.z_range
        vol = np.zeros((xr, yr, zr), dtype=np.uint8)

        t = max(0, int(self.outer_shell_thickness))
        def is_outer(x,y,z):
            return (x < t) or (y < t) or (z < t) or (x >= xr - t) or (y >= yr - t) or (z >= zr - t)

        if bounds is None:
            for (x, y, z) in self.env.obstacles:
                if self.hide_outer_walls and is_outer(x,y,z):
                    continue
                if 0 <= x < xr and 0 <= y < yr and 0 <= z < zr:
                    vol[x, y, z] = 1
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = bounds
            for (x, y, z) in self.env.obstacles:
                if self.hide_outer_walls and is_outer(x,y,z):
                    continue
                if xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax:
                    vol[x, y, z] = 1

        arr = vol
        grid = pv.ImageData() if hasattr(pv, "ImageData") else pv.UniformGrid()
        grid.dimensions = np.array(arr.shape) + 1
        grid.spacing = (1, 1, 1)
        grid.origin  = (0, 0, 0)
        grid.cell_data["occ"] = arr.flatten(order="F")

        surf = grid.threshold(0.5, scalars="occ").extract_surface().clean()
        self._obs_mesh = surf

        if self._obs_actor is not None:
            try: self.pl.remove_actor(self._obs_actor)
            except Exception: pass

        self._obs_actor = self.pl.add_mesh(
            surf, color=(0.15, 0.15, 0.15), opacity=0.22, specular=0.0, backface_culling=True
        )

    def _clip_and_rerender_obstacles_around_path(self, path, margin=2):
        xs, ys, zs = zip(*[self._xyz(p) for p in path])
        xmin = max(0, int(min(xs)) - margin)
        xmax = min(int(self.env.x_range - 1), int(max(xs)) + margin)
        ymin = max(0, int(min(ys)) - margin)
        ymax = min(int(self.env.y_range - 1), int(max(ys)) + margin)
        zmin = max(0, int(min(zs)) - margin)
        zmax = min(int(self.env.z_range - 1), int(max(zs)) + margin)
        self._build_and_render_grid_obstacles(bounds=(xmin, xmax, ymin, ymax, zmin, zmax))
        self.pl.render()

    # ----------------------------- Helpers ------------------------------
    @staticmethod
    def _xyz(obj):
        if hasattr(obj, 'x') and hasattr(obj, 'y'):
            z = getattr(obj, 'z', 0.0)
            return float(obj.x), float(obj.y), float(z)
        if len(obj) == 2:
            return float(obj[0]), float(obj[1]), 0.0
        return float(obj[0]), float(obj[1]), float(obj[2])

    @staticmethod
    def _xyz_pose(pose):
        if len(pose) == 3:
            x, y, th = pose
            return float(x), float(y), 0.0
        x, y, z, _ = pose
        return float(x), float(y), float(z)

    @staticmethod
    def _rect3d(ox, oy, w, h, z=0.0):
        return [(ox, oy, z), (ox+w, oy, z), (ox+w, oy+h, z), (ox, oy+h, z)]

    @staticmethod
    def _circle3d(cx, cy, r, z=0.0, n=48):
        t = np.linspace(0, 2*np.pi, n, endpoint=True)
        return [(cx + r*np.cos(tt), cy + r*np.sin(tt), z) for tt in t]

    def _polyline(self, pts: np.ndarray, close: bool = False) -> pv.PolyData:
        n = int(pts.shape[0])
        poly = pv.PolyData(pts)
        if close:
            cells = np.hstack(([n+1], np.arange(n, dtype=np.int64), [0])).astype(np.int64)
        else:
            cells = np.hstack(([n],   np.arange(n, dtype=np.int64))).astype(np.int64)
        poly.lines = cells
        return poly

    def _ensure_window(self):
        if not self._shown:
            try:
                self.pl.show(interactive_update=True, auto_close=False)
            except TypeError:
                self.pl.show(auto_close=False)
            self._shown = True

    def _poll(self):
        try:
            if hasattr(self.pl, "update"):
                self.pl.update()
            else:
                self.pl.render()
        except Exception:
            pass

    def _window_alive(self) -> bool:
        if getattr(self.pl, "_closed", False):
            return False
        try:
            rw = getattr(self.pl, "ren_win", None)
            return rw is not None
        except Exception:
            return True

    def _replay_hotkey(self):
        if not self._last_anim:
            return
        p = self._last_anim
        self.plotEnv(p["title"])
        if p["expand"] is not None:
            self._animate_expand_batched(p["expand"], delay_sec=0.5)
        if p["history_pose"] is not None:
            self._draw_history_instant(p["history_pose"], p["predict_path"], p["lookahead_pts"])
        if p["path"] is not None:
            self.plotPath(p["path"])
            if self.clip_obstacles_to_path and isinstance(self.env, Grid3D):
                self._clip_and_rerender_obstacles_around_path(p["path"], margin=self.clip_margin)
        if p["ellipse"] is not None:
            self.plotEllipse(p["ellipse"])
        self.pl.render()
