"""
@file: bspline_curve3d.py
@breif: B-Spline curve generation in 3D (also works for 2D generically)
@author: Winter, Joar
@update: 2025.9.10
"""
import math
import numpy as np

from .curve import Curve


class BSpline3D(Curve):
    """
    B-Spline curve generation in 3D (dimension-agnostic; 2D works too).

    Parameters:
        step (float): Simulation or interpolation size (0<step<=1)
        k (int): Degree of curve
        param_mode (str): "centripetal" | "chord_length" | "uniform_spaced"
        spline_mode (str): "interpolation" | "approximation"

    Examples:
        >>> from python_motion_planning.curve_generation.bspline_curve_3d import BSpline3D
        >>> points = [(0, 0, 0), (10, 10, -5), (20, 5, 6), (30, 12, 0)]
        >>> gen = BSpline3D(step=0.01, k=3)
        >>> path = gen.run(points, display=False)
    """
    def __init__(self, step: float, k: int, param_mode: str = "centripetal",
                 spline_mode: str = "interpolation") -> None:
        super().__init__(step)
        self.k = k

        assert param_mode in ("centripetal", "chord_length", "uniform_spaced"), \
            "Parameter selection mode error!"
        self.param_mode = param_mode

        assert spline_mode in ("interpolation", "approximation"), \
            "Spline mode selection error!"
        self.spline_mode = spline_mode
    
    def __str__(self) -> str:
        return "B-Spline Curve 3D"

    # ------------------ Core B-spline utilities ------------------ #
    def baseFunction(self, i: int, k: int, t: float, knot: list):
        """
        Cox–de Boor recursion for basis function N_{i,k}(t).
        """
        if k == 0:
            return 1.0 if (knot[i] <= t < knot[i + 1]) else 0.0
        length1 = knot[i + k] - knot[i]
        length2 = knot[i + k + 1] - knot[i + 1]
        term1 = 0.0 if length1 == 0 else (t - knot[i]) / length1 * self.baseFunction(i, k - 1, t, knot)
        term2 = 0.0 if length2 == 0 else (knot[i + k + 1] - t) / length2 * self.baseFunction(i + 1, k - 1, t, knot)
        return term1 + term2

    def paramSelection(self, points: list):
        """
        Parameterization: uniform, chord-length, or centripetal over Euclidean distances
        in the native dimension (2D or 3D).
        """
        n = len(points)
        if n == 1:
            return [0.0]
        P = np.asarray(points, dtype=float)  # shape (n, d)
        diffs = np.diff(P, axis=0)          # (n-1, d)
        seg_len = np.linalg.norm(diffs, axis=1)  # Euclidean lengths

        if self.param_mode == "uniform_spaced":
            return np.linspace(0, 1, n).tolist()

        if self.param_mode == "chord_length":
            s = np.cumsum(seg_len)
            params = np.zeros(n)
            if s[-1] > 0:
                params[1:] = s / s[-1]
            return params.tolist()

        # centripetal
        alpha = 0.5
        s = np.cumsum(np.power(seg_len, alpha))
        params = np.zeros(n)
        if s[-1] > 0:
            params[1:] = s / s[-1]
        return params.tolist()

    def knotGeneration(self, param: list, n: int):
        """
        Averaging knot vector with clamped ends.
        """
        m = n + self.k + 1  # knot count
        knot = np.zeros(m)
        knot[:self.k + 1] = 0.0
        knot[n:] = 1.0
        for i in range(self.k + 1, n):
            # average the previous k parameters
            s = 0.0
            for j in range(i - self.k, i):
                s += param[j]
            knot[i] = s / self.k
        return knot.tolist()

    def interpolation(self, points: list, param: list, knot: list):
        """
        Solve for control points so the curve interpolates the data points.
        """
        n = len(points)
        N = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                N[i, j] = self.baseFunction(j, self.k, param[i], knot)
        N[-1, -1] = 1.0  # ensure endpoint interpolation

        D = np.asarray(points, dtype=float)  # (n, d)
        control_pts = np.linalg.inv(N) @ D   # (n, d)
        return control_pts

    def approximation(self, points: list, param: list, knot: list):
        """
        Least-squares B-spline approximation with end-point interpolation.
        """
        D = np.asarray(points, dtype=float)  # (n, d)
        n = D.shape[0]
        d = D.shape[1]

        # Heuristic: number of control points
        h = n - 1
        N = np.zeros((n, h))
        for i in range(n):
            for j in range(h):
                N[i, j] = self.baseFunction(j, self.k, param[i], knot)

        # internal blocks (exclude endpoints to fix them)
        N_ = N[1:n-1, 1:h-1]  # (n-2, h-2)

        # rhs for each interior row
        qk = np.zeros((n - 2, d))
        for i in range(1, n - 1):
            qk[i - 1, :] = D[i, :] - N[i, 0] * D[0, :] - N[i, h - 1] * D[-1, :]

        # normal equations
        A = N_.T @ N_
        B = N_.T @ qk
        P_int = np.linalg.inv(A) @ B  # (h-2, d)

        # assemble full control net with fixed endpoints
        P = np.vstack([D[0, :], P_int, D[-1, :]])  # (h, d)
        return P

    def generation(self, t: np.ndarray, k: int, knot: list, control_pts: np.ndarray):
        """
        Evaluate the B-spline curve at parameter values t.
        """
        n_ctrl = len(control_pts)
        N = np.zeros((len(t), n_ctrl))
        for i in range(len(t)):
            for j in range(n_ctrl):
                N[i, j] = self.baseFunction(j, k, t[i], knot)
        N[-1, -1] = 1.0  # exact end hit
        return N @ control_pts  # (len(t), d)

    # ------------------ Public API ------------------ #
    def run(self, points: list, display: bool = True):
        """
        Generate the B-spline path. Supports 2D and 3D points.
        """
        assert len(points) >= 2, "Number of points should be at least 2."

        P = np.asarray(points, dtype=float)
        d = P.shape[1]
        assert d in (2, 3), "BSpline3D supports 2D or 3D points."

        # parameter samples along curve
        n_samples = max(2, int(1 / self.step))
        t = np.linspace(0, 1, n_samples)

        params = self.paramSelection(points)
        knot = self.knotGeneration(params, len(points))

        if self.spline_mode == "interpolation":
            control_pts = self.interpolation(points, params, knot)
        elif self.spline_mode == "approximation":
            control_pts = self.approximation(points, params, knot)
            # Recompute knots for the new control polygon size if desired
            # (the original code recomputed; we mimic that)
            new_points = control_pts.tolist()
            params = self.paramSelection(new_points)
            knot = self.knotGeneration(params, len(new_points))
        else:
            raise NotImplementedError

        path = self.generation(t, self.k, knot, control_pts)  # (n_samples, d)

        if display:
            import matplotlib.pyplot as plt
            if d == 3:
                from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D)
                fig = plt.figure("curve generation (3D)")
                ax = fig.add_subplot(111, projection="3d")
                ax.plot(path[:, 0], path[:, 1], path[:, 2], linewidth=2)
                ax.plot(control_pts[:, 0], control_pts[:, 1], control_pts[:, 2],
                        '--o', label="Control Points")
                for x, y, z in points:
                    ax.plot([x], [y], [z], "xr")
                ax.set_title(str(self))
                ax.legend()
            else:
                plt.figure("curve generation (2D)")
                plt.plot(path[:, 0], path[:, 1], linewidth=2)
                plt.plot(control_pts[:, 0], control_pts[:, 1], '--o', label="Control Points")
                for x, y in points:
                    plt.plot(x, y, "xr", linewidth=2)
                plt.axis("equal")
                plt.legend()
                plt.title(str(self))
            plt.show()

        # Return list of tuples in native dimension
        return [tuple(row) for row in path.tolist()]
