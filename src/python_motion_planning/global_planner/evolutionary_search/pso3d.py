"""
@file: pso3d.py
@breif: Particle Swarm Optimization (PSO) motion planning in 3D (with diagnostics, adaptive inertia, projected sorting, stagnation kick)
@author: Yang Haodong, Wu Maojia, Joar Heimonen
@update: 2025.9.10
"""
import random, math
from copy import deepcopy

from .evolutionary_search3d import EvolutionarySearcher3D
from python_motion_planning.utils import Env3D, MathHelper, Grid3D
from python_motion_planning.curve_generation import BSpline3D

GEN_MODE_CIRCLE = 0   # interpreted as SPHERE in 3D
GEN_MODE_RANDOM = 1


class PSO3D(EvolutionarySearcher3D):
    """
    Particle Swarm Optimization (PSO) motion planning in 3D.
    """

    def __init__(
        self,
        start: tuple,
        goal: tuple,
        env: Grid3D,
        heuristic_type: str = "euclidean",
        n_particles: int = 300,
        point_num: int = 5,
        w_inertial: float = 1.0,
        w_cognitive: float = 1.0,
        w_social: float = 1.0,
        max_speed: int = 6,
        max_iter: int = 200,
        init_mode: int = GEN_MODE_RANDOM,
    ) -> None:
        super().__init__(start, goal, env, heuristic_type)
        self.max_iter = max_iter
        self.n_particles = n_particles
        self.w_inertial = w_inertial
        self.w_social = w_social
        self.w_cognitive = w_cognitive
        self.point_num = point_num
        self.init_mode = init_mode
        self.max_speed = max_speed

        self.particles = []
        self.inherited_particles = []
        self.best_particle = self.Particle()
        self.b_spline_gen = BSpline3D(step=0.01, k=4)

        # diagnostics knobs
        self._debug_every = 1      # print every iteration when verbose
        self._stall_eps = 1e-3     # movement threshold for "stalled" particles

        # path direction (for projected sorting)
        self._setup_direction()

    def __str__(self) -> str:
        return "Particle Swarm Optimization 3D (PSO3D)"

    class Particle:
        def __init__(self) -> None:
            self.reset()

        def reset(self):
            self.position = []       # list[(x,y,z)] (floats)
            self.velocity = []       # list[(vx,vy,vz)] (floats)
            self.fitness = -1.0
            self.best_pos = []       # list[(x,y,z)]
            self.best_fitness = -1.0

    # ---------------------------- Core PSO Loop ---------------------------- #
    def plan(self, verbose: bool = False):
        """
        PSO motion plan function in 3D.

        Parameters:
            verbose (bool): print detailed per-iteration diagnostics

        Returns:
            cost (float): path cost
            path (list[tuple]): resulting path as (x, y, z) samples
            fitness_history (list[float]): best fitness per iteration
        """
        # (Re)initialize containers each run
        self.particles = []
        self.best_particle = self.Particle()
        self._setup_direction()

        # Generate initial control points for particle swarm
        init_positions = self.initializePositions()

        # Particle initialization
        for i in range(self.n_particles):
            init_fitness = self.calFitnessValue(init_positions[i])

            if i == 0 or init_fitness > self.best_particle.fitness:
                self.best_particle.fitness = init_fitness
                self.best_particle.position = deepcopy(init_positions[i])

            p = self.Particle()
            # IMPORTANT: deep copies to avoid aliasing with best_pos
            p.position = deepcopy(init_positions[i])          # keep as floats downstream
            p.best_pos = deepcopy(init_positions[i])

            # Seed non-zero velocities so swarm explores
            vel_scale = max(1.0, 0.5 * self.max_speed)
            p.velocity = [(
                random.uniform(-vel_scale, vel_scale),
                random.uniform(-vel_scale, vel_scale),
                random.uniform(-vel_scale, vel_scale),
            ) for _ in range(self.point_num)]

            p.fitness = init_fitness
            p.best_fitness = init_fitness
            self.particles.append(p)

        if verbose:
            print(f"[init] start={self.start.current}, goal={self.goal.current}, "
                  f"grid=({self.env.x_range},{self.env.y_range},{self.env.z_range}), "
                  f"particles={self.n_particles}, points/particle={self.point_num}, "
                  f"w=(inertial={self.w_inertial}, cognitive={self.w_cognitive}, social={self.w_social}), "
                  f"max_speed={self.max_speed}")

        # Adaptive inertia schedule (explore -> exploit)
        w_hi, w_lo = 0.9, 0.4
        patience = 40
        no_improve = 0

        # Iterative optimization
        fitness_history = []
        for it in range(self.max_iter):
            # set inertia for this iteration
            self._w_now = w_hi - (w_hi - w_lo) * (it / max(1, self.max_iter - 1))

            gbest_before = self.best_particle.fitness
            moves = []          # avg movement per particle (mean control-point displacement)
            vel_mags = []       # velocity magnitudes across all control points
            improved_personal = 0

            for p in self.particles:
                prev = p.position[:]  # shallow copy of list of tuples
                improved = self.optimizeParticle(p)
                if improved:
                    improved_personal += 1

                # movement magnitude (average over control points)
                if self.point_num > 0:
                    dsum = 0.0
                    for j in range(self.point_num):
                        dsum += math.dist(prev[j], p.position[j])
                    moves.append(dsum / self.point_num)

                # gather velocity magnitudes
                for (vx, vy, vz) in p.velocity:
                    vel_mags.append(math.sqrt(vx*vx + vy*vy + vz*vz))

            fitness_history.append(self.best_particle.fitness)

            # stagnation kick if no improvement for `patience` iters
            if self.best_particle.fitness > gbest_before:
                no_improve = 0
            else:
                no_improve += 1
                if no_improve == patience:
                    if verbose:
                        print(f"  [kick] no_improve={no_improve}, re-seeding worst 20% near gbest")
                    self._kick_swarm(frac=0.2, pos_sigma=2.0)
                    no_improve = 0

            if verbose and (it % self._debug_every == 0):
                stalled_frac = (sum(1 for m in moves if m < self._stall_eps) / len(moves)) if moves else 0.0
                mean_move = (sum(moves) / len(moves)) if moves else 0.0
                mean_vel = (sum(vel_mags) / len(vel_mags)) if vel_mags else 0.0
                min_vel = min(vel_mags) if vel_mags else 0.0
                max_vel = max(vel_mags) if vel_mags else 0.0
                blen, bcol = self._diagnose_best()

                print(f"iteration {it}: best fitness = {self.best_particle.fitness}")
                print(f"  gbest_improved={self.best_particle.fitness > gbest_before} | "
                      f"personal_improvements={improved_personal}/{self.n_particles}")
                print(f"  movement: mean={mean_move:.6f}, stalled(<{self._stall_eps})={stalled_frac:.2%}")
                print(f"  velocity: mean={mean_vel:.6f}, min={min_vel:.6f}, max={max_vel:.6f}")
                print(f"  best_path: length={blen:.3f}, collisions={bcol}")

        # Build B-spline path from best particle
        points = [self.start.current] + self.best_particle.position + [self.goal.current]
        points = sorted(set(points), key=points.index)  # preserve order, remove dupes
        path = self._run_spline(points)

        # Cost is the path length
        return self._path_length(path), path, fitness_history

    # ---------------------------- Initialization ---------------------------- #
    def initializePositions(self) -> list:
        """
        Generate n_particles sequences, each with point_num control points in 3D.
        """
        init_positions = []

        # Sphere generation (3D analogue of circle)
        cx = cy = cz = radius = None
        if self.init_mode == GEN_MODE_CIRCLE:
            cx = (self.start.x + self.goal.x) / 2.0
            cy = (self.start.y + self.goal.y) / 2.0
            cz = (self.start.z + self.goal.z) / 2.0
            d = self.dist(self.start, self.goal) / 2.0
            radius = 5 if d < 5 else d

        # Axis-aligned bounds for random mode
        x_lo, x_hi = sorted([self.start.x, self.goal.x])
        y_lo, y_hi = sorted([self.start.y, self.goal.y])
        z_lo, z_hi = sorted([self.start.z, self.goal.z])

        xr, yr, zr = self.env.x_range, self.env.y_range, self.env.z_range

        for _ in range(self.n_particles):
            point_id, visited = 0, set()
            pts_x, pts_y, pts_z = [], [], []

            while point_id < self.point_num:
                if self.init_mode == GEN_MODE_RANDOM:
                    pt_x = random.randint(x_lo, x_hi)
                    pt_y = random.randint(y_lo, y_hi)
                    pt_z = random.randint(z_lo, z_hi)
                    pos_id = pt_x + xr * (pt_y + yr * pt_z)
                else:
                    # sample uniformly inside sphere via cubic root of random
                    u = random.random()
                    r = radius * (u ** (1.0 / 3.0))
                    theta = random.random() * 2.0 * math.pi
                    phi = math.acos(2.0 * random.random() - 1.0)  # [0, pi]
                    sx = int(cx + r * math.sin(phi) * math.cos(theta))
                    sy = int(cy + r * math.sin(phi) * math.sin(theta))
                    sz = int(cz + r * math.cos(phi))
                    # bounds check
                    if 0 <= sx < xr and 0 <= sy < yr and 0 <= sz < zr:
                        pt_x, pt_y, pt_z = sx, sy, sz
                        pos_id = pt_x + xr * (pt_y + yr * pt_z)
                    else:
                        continue

                if pos_id not in visited:
                    visited.add(pos_id)
                    pts_x.append(float(pt_x))
                    pts_y.append(float(pt_y))
                    pts_z.append(float(pt_z))
                    point_id += 1

            # PROJECTED SORT along start→goal direction (keeps natural path order)
            pts = [(ix, iy, iz) for (ix, iy, iz) in zip(pts_x, pts_y, pts_z)]
            pts.sort(key=self._project_t)
            init_positions.append(pts)

        return init_positions

    # ---------------------------- Fitness ---------------------------- #
    def calFitnessValue(self, position: list) -> float:
        """
        Fitness = high for short, collision-free smooth curves.
        """
        points = [self.start.current] + position + [self.goal.current]
        points = sorted(set(points), key=points.index)
        try:
            path = self._run_spline(points)
        except Exception:
            return float("inf")

        # collision detection along discretized spline (use rounded grid samples)
        obs_cost = 0
        for i in range(len(path) - 1):
            p1 = (round(path[i][0]), round(path[i][1]), round(path[i][2]))
            p2 = (round(path[i + 1][0]), round(path[i + 1][1]), round(path[i + 1][2]))
            if self.isCollision(p1, p2):
                obs_cost += 1

        length = self._path_length(path)
        return 100000.0 / (length + 50000 * obs_cost)

    # ---------------------------- Particle Updates ---------------------------- #
    def updateParticleVelocity(self, particle):
        """
        Update particle velocity (vx, vy, vz) with aspect-ratio scaling and limits.
        """
        xr, yr, zr = self.env.x_range, self.env.y_range, self.env.z_range
        max_r = max(xr, yr, zr)
        sx = max_r / xr if xr > 0 else 1.0
        sy = max_r / yr if yr > 0 else 1.0
        sz = max_r / zr if zr > 0 else 1.0

        # adaptive inertia this iteration
        w = getattr(self, "_w_now", self.w_inertial)

        for i in range(self.point_num):
            rand1, rand2 = random.random(), random.random()
            vx, vy, vz = particle.velocity[i]
            px, py, pz = particle.position[i]

            vx_new = (
                w * vx
                + self.w_cognitive * rand1 * (particle.best_pos[i][0] - px)
                + self.w_social * rand2 * (self.best_particle.position[i][0] - px)
            )
            vy_new = (
                w * vy
                + self.w_cognitive * rand1 * (particle.best_pos[i][1] - py)
                + self.w_social * rand2 * (self.best_particle.position[i][1] - py)
            )
            vz_new = (
                w * vz
                + self.w_cognitive * rand1 * (particle.best_pos[i][2] - pz)
                + self.w_social * rand2 * (self.best_particle.position[i][2] - pz)
            )

            # Aspect scaling
            vx_new *= sx
            vy_new *= sy
            vz_new *= sz

            # Velocity clamp
            vx_new = MathHelper.clamp(vx_new, -self.max_speed, self.max_speed)
            vy_new = MathHelper.clamp(vy_new, -self.max_speed, self.max_speed)
            vz_new = MathHelper.clamp(vz_new, -self.max_speed, self.max_speed)

            particle.velocity[i] = (vx_new, vy_new, vz_new)

    def updateParticlePosition(self, particle):
        """
        Euler-integrate positions in *float* space and clamp within grid bounds.
        """
        for i in range(self.point_num):
            px = particle.position[i][0] + particle.velocity[i][0]
            py = particle.position[i][1] + particle.velocity[i][1]
            pz = particle.position[i][2] + particle.velocity[i][2]

            # Position limit (floats allowed)
            px = MathHelper.clamp(px, 0.0, float(self.env.x_range - 1))
            py = MathHelper.clamp(py, 0.0, float(self.env.y_range - 1))
            pz = MathHelper.clamp(pz, 0.0, float(self.env.z_range - 1))

            particle.position[i] = (px, py, pz)

        # Keep order stable by projection along start→goal
        particle.position.sort(key=self._project_t)

    def optimizeParticle(self, particle):
        """
        One PSO update step (velocity, position, fitness, and bests).
        Returns:
            improved_personal (bool): True if personal best improved
        """
        self.updateParticleVelocity(particle)
        self.updateParticlePosition(particle)

        # Evaluate
        prev_best = particle.best_fitness
        particle.fitness = self.calFitnessValue(particle.position)

        # Update personal best
        improved_personal = False
        if particle.fitness > particle.best_fitness:
            particle.best_fitness = particle.fitness
            particle.best_pos = deepcopy(particle.position)
            improved_personal = True
        else:
            # optional tiny nudge to escape stagnation (low probability)
            if random.random() < 0.1 and self.point_num > 0:
                j = random.randrange(self.point_num)
                px, py, pz = particle.position[j]
                particle.position[j] = (
                    MathHelper.clamp(px + random.choice([-1.0, 1.0]), 0.0, float(self.env.x_range - 1)),
                    MathHelper.clamp(py + random.choice([-1.0, 1.0]), 0.0, float(self.env.y_range - 1)),
                    MathHelper.clamp(pz + random.choice([-1.0, 1.0]), 0.0, float(self.env.z_range - 1)),
                )

        # Update global best
        if particle.best_fitness > self.best_particle.fitness:
            self.best_particle.fitness = particle.best_fitness
            self.best_particle.position = deepcopy(particle.position)

        return improved_personal

    # ---------------------------- Runner ---------------------------- #
    def run(self):
        """
        Run both planning and animation.
        """
        cost, path, fitness_history = self.plan(verbose=True)
        cost_curve = [-f for f in fitness_history]
        self.plot.animation(path, str(self), cost, cost_curve=cost_curve)

    # ---------------------------- Collision (3D Bresenham) ---------------------------- #
    def isCollision(self, p1: tuple, p2: tuple) -> bool:
        """
        Judge collision when moving from p1 to p2 using 3D Bresenham.
        """
        if p1 in self.obstacles or p2 in self.obstacles:
            return True

        x1, y1, z1 = p1
        x2, y2, z2 = p2

        xr, yr, zr = self.env.x_range, self.env.y_range, self.env.z_range
        # bounds check
        if not (0 <= x1 < xr and 0 <= y1 < yr and 0 <= z1 < zr):
            return True
        if not (0 <= x2 < xr and 0 <= y2 < yr and 0 <= z2 < zr):
            return True

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        dz = abs(z2 - z1)
        sx = 1 if x2 >= x1 else -1
        sy = 1 if y2 >= y1 else -1
        sz = 1 if z2 >= z1 else -1

        x, y, z = x1, y1, z1

        # 3D Bresenham variant
        if dx >= dy and dx >= dz:
            p1_err = 2 * dy - dx
            p2_err = 2 * dz - dx
            while x != x2:
                x += sx
                if p1_err >= 0:
                    y += sy
                    p1_err -= 2 * dx
                if p2_err >= 0:
                    z += sz
                    p2_err -= 2 * dx
                p1_err += 2 * dy
                p2_err += 2 * dz
                if (x, y, z) in self.obstacles:
                    return True
        elif dy >= dx and dy >= dz:
            p1_err = 2 * dx - dy
            p2_err = 2 * dz - dy
            while y != y2:
                y += sy
                if p1_err >= 0:
                    x += sx
                    p1_err -= 2 * dy
                if p2_err >= 0:
                    z += sz
                    p2_err -= 2 * dy
                p1_err += 2 * dx
                p2_err += 2 * dz
                if (x, y, z) in self.obstacles:
                    return True
        else:
            p1_err = 2 * dy - dz
            p2_err = 2 * dx - dz
            while z != z2:
                z += sz
                if p1_err >= 0:
                    y += sy
                    p1_err -= 2 * dz
                if p2_err >= 0:
                    x += sx
                    p2_err -= 2 * dz
                p1_err += 2 * dy
                p2_err += 2 * dx
                if (x, y, z) in self.obstacles:
                    return True

        return False

    # ---------------------------- Internals ---------------------------- #
    def _setup_direction(self):
        sx, sy, sz = self.start.current
        gx, gy, gz = self.goal.current
        self._dir = (gx - sx, gy - sy, gz - sz)
        self._dir_len = math.sqrt(self._dir[0]**2 + self._dir[1]**2 + self._dir[2]**2) or 1.0
        self._s0 = (sx, sy, sz)

    def _project_t(self, p):
        # scalar projection of (p - start) onto start->goal
        vx, vy, vz = p[0] - self._s0[0], p[1] - self._s0[1], p[2] - self._s0[2]
        return (vx * self._dir[0] + vy * self._dir[1] + vz * self._dir[2]) / self._dir_len

    def _run_spline(self, points):
        """
        Run BSpline with 3D points. If the BSpline length method doesn't support 3D,
        we still use its sampled points and compute length ourselves.
        """
        return self.b_spline_gen.run(points, display=False)

    def _path_length(self, path):
        """
        Robust 3D length. Uses BSpline.length if available; otherwise computes manually.
        """
        try:
            return self.b_spline_gen.length(path)
        except Exception:
            length = 0.0
            for i in range(len(path) - 1):
                x1, y1, z1 = path[i]
                x2, y2, z2 = path[i + 1]
                dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
                length += math.sqrt(dx * dx + dy * dy + dz * dz)
            return length

    def _diagnose_best(self):
        """
        Recompute best path length and collision count for diagnostics.
        """
        pts = [self.start.current] + self.best_particle.position + [self.goal.current]
        pts = sorted(set(pts), key=pts.index)
        try:
            path = self._run_spline(pts)
        except Exception:
            return float("inf"), float("inf")

        # collisions
        col = 0
        for i in range(len(path) - 1):
            p1 = (round(path[i][0]), round(path[i][1]), round(path[i][2]))
            p2 = (round(path[i + 1][0]), round(path[i + 1][1]), round(path[i + 1][2]))
            if self.isCollision(p1, p2):
                col += 1
        return self._path_length(path), col

    def _kick_swarm(self, frac: float = 0.2, pos_sigma: float = 1.5, vel_scale: float = None):
        """
        Re-seed a fraction of the worst particles near the current global best.
        """
        if not self.particles or not self.best_particle.position:
            return
        k = max(1, int(frac * len(self.particles)))
        # worst by personal best fitness
        worst = sorted(self.particles, key=lambda p: p.best_fitness)[:k]
        gbest = self.best_particle.position
        if vel_scale is None:
            vel_scale = max(1.0, 0.5 * self.max_speed)

        for p in worst:
            # re-seed around gbest with Gaussian noise, keep order via projection
            new_pos = []
            for (gx, gy, gz) in gbest:
                nx = MathHelper.clamp(gx + random.gauss(0, pos_sigma), 0.0, float(self.env.x_range - 1))
                ny = MathHelper.clamp(gy + random.gauss(0, pos_sigma), 0.0, float(self.env.y_range - 1))
                nz = MathHelper.clamp(gz + random.gauss(0, pos_sigma), 0.0, float(self.env.z_range - 1))
                new_pos.append((nx, ny, nz))
            new_pos.sort(key=self._project_t)

            p.position = new_pos
            p.velocity = [(
                random.uniform(-vel_scale, vel_scale),
                random.uniform(-vel_scale, vel_scale),
                random.uniform(-vel_scale, vel_scale),
            ) for _ in range(self.point_num)]
            p.fitness = self.calFitnessValue(p.position)
            p.best_pos = deepcopy(p.position)
            p.best_fitness = p.fitness
