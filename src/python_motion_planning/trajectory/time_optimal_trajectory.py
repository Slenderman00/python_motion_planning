import numpy as np
from typing import List, Tuple, Optional
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
from .trajectory_base import Trajectory3D, TrajectoryPoint, TrajectoryConstraints


class TimeOptimalTrajectory3D(Trajectory3D):
    """
    Generates time-optimal trajectories along a given path.

    This implementation uses the path-velocity decomposition method
    to find the fastest trajectory that respects velocity and
    acceleration constraints.
    """

    def __init__(self,
                 path: List[Tuple],
                 constraints: Optional[TrajectoryConstraints] = None,
                 path_resolution: float = 0.01):
        """
        Initialize time-optimal trajectory generator.

        Args:
            path: List of waypoints
            constraints: Physical constraints
            path_resolution: Resolution for path parameterization
        """
        super().__init__(path, constraints)
        self.path_resolution = path_resolution

        # Path parameterization
        self.s_values = None  # Path parameter values
        self.path_length = 0.0
        self.path_interp = None  # Interpolated path functions

        # Velocity profile
        self.s_dot_profile = None  # Velocity along path
        self.s_ddot_profile = None  # Acceleration along path
        self.time_profile = None  # Time at each path point

    def _parameterize_path(self):
        """
        Parameterize the path using arc length.
        Creates interpolation functions for smooth path evaluation.
        """
        # Compute cumulative arc length
        n_points = len(self.path)
        arc_lengths = np.zeros(n_points)

        for i in range(1, n_points):
            segment_length = np.linalg.norm(self.path[i] - self.path[i-1])
            arc_lengths[i] = arc_lengths[i-1] + segment_length

        self.path_length = arc_lengths[-1]

        # Create dense parameterization
        num_samples = max(int(self.path_length / self.path_resolution), 100)
        self.s_values = np.linspace(0, self.path_length, num_samples)

        # Create interpolation functions for each dimension
        self.path_interp = []
        self.path_deriv_interp = []
        self.path_second_deriv_interp = []

        for dim in range(self.dimension):
            coords = self.path[:, dim]

            # Use cubic spline interpolation
            from scipy.interpolate import CubicSpline
            spline = CubicSpline(arc_lengths, coords)

            self.path_interp.append(spline)
            self.path_deriv_interp.append(spline.derivative(1))
            self.path_second_deriv_interp.append(spline.derivative(2))

    def _evaluate_path(self, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate path at parameter s.

        Args:
            s: Path parameter (0 <= s <= path_length)

        Returns:
            Tuple of (position, first_derivative, second_derivative)
        """
        s = np.clip(s, 0, self.path_length)

        position = np.array([interp(s) for interp in self.path_interp])
        first_deriv = np.array([interp(s) for interp in self.path_deriv_interp])
        second_deriv = np.array([interp(s) for interp in self.path_second_deriv_interp])

        return position, first_deriv, second_deriv

    def _compute_max_velocity(self, s: float) -> float:
        """
        Compute maximum velocity at path point s based on constraints.

        Args:
            s: Path parameter

        Returns:
            Maximum allowed velocity
        """
        _, q_prime, _ = self._evaluate_path(s)

        # Maximum velocity based on path tangent
        max_velocities = []

        for dim in range(self.dimension):
            if abs(q_prime[dim]) > 1e-10:
                max_v = self.constraints.max_velocity[dim] / abs(q_prime[dim])
                max_velocities.append(max_v)

        if max_velocities:
            return min(max_velocities)
        else:
            return np.min(self.constraints.max_velocity)

    def _compute_max_acceleration(self, s: float, s_dot: float) -> Tuple[float, float]:
        """
        Compute maximum forward and backward accelerations at path point s.

        Args:
            s: Path parameter
            s_dot: Velocity along path

        Returns:
            Tuple of (max_forward_acceleration, max_backward_acceleration)
        """
        _, q_prime, q_double_prime = self._evaluate_path(s)

        # Acceleration constraint: a = q''*s_dot^2 + q'*s_ddot
        # Rearranging: s_ddot = (a - q''*s_dot^2) / q'

        s_ddot_max = float('inf')
        s_ddot_min = float('-inf')

        for dim in range(self.dimension):
            if abs(q_prime[dim]) > 1e-10:
                # Contribution from centripetal acceleration
                centripetal = q_double_prime[dim] * s_dot * s_dot

                # Available acceleration for tangential component
                a_max = self.constraints.max_acceleration[dim]

                # Forward acceleration
                s_ddot_forward = (a_max - centripetal) / q_prime[dim]
                s_ddot_backward = (-a_max - centripetal) / q_prime[dim]

                # Update limits
                if q_prime[dim] > 0:
                    s_ddot_max = min(s_ddot_max, s_ddot_forward)
                    s_ddot_min = max(s_ddot_min, s_ddot_backward)
                else:
                    s_ddot_max = min(s_ddot_max, -s_ddot_backward)
                    s_ddot_min = max(s_ddot_min, -s_ddot_forward)

        return s_ddot_max, s_ddot_min

    def _forward_integration(self) -> np.ndarray:
        """
        Forward integration to compute maximum velocity profile.

        Returns:
            Array of maximum velocities at each path point
        """
        n = len(self.s_values)
        s_dot_max = np.zeros(n)

        # Start from rest
        s_dot_max[0] = 0.0

        for i in range(1, n):
            ds = self.s_values[i] - self.s_values[i-1]
            s_mid = (self.s_values[i] + self.s_values[i-1]) / 2.0

            # Maximum velocity from constraints
            v_max_constraint = self._compute_max_velocity(s_mid)

            # Maximum acceleration at previous point
            s_ddot_max, _ = self._compute_max_acceleration(
                self.s_values[i-1], s_dot_max[i-1]
            )

            # Apply kinematic equation: v^2 = v0^2 + 2*a*ds
            if s_ddot_max > 0:
                v_squared = s_dot_max[i-1]**2 + 2 * s_ddot_max * ds
                s_dot_max[i] = min(np.sqrt(max(v_squared, 0)), v_max_constraint)
            else:
                s_dot_max[i] = min(s_dot_max[i-1], v_max_constraint)

        return s_dot_max

    def _backward_integration(self, s_dot_forward: np.ndarray) -> np.ndarray:
        """
        Backward integration to ensure we can decelerate to stop.

        Args:
            s_dot_forward: Forward integrated velocity profile

        Returns:
            Array of final velocities at each path point
        """
        n = len(self.s_values)
        s_dot_final = np.copy(s_dot_forward)

        # End at rest
        s_dot_final[-1] = 0.0

        for i in range(n-2, -1, -1):
            ds = self.s_values[i+1] - self.s_values[i]
            s_mid = (self.s_values[i+1] + self.s_values[i]) / 2.0

            # Maximum deceleration at next point
            _, s_ddot_min = self._compute_max_acceleration(
                self.s_values[i+1], s_dot_final[i+1]
            )

            # Apply kinematic equation backwards
            if s_ddot_min < 0:
                v_squared = s_dot_final[i+1]**2 - 2 * s_ddot_min * ds
                v_back = np.sqrt(max(v_squared, 0))
                s_dot_final[i] = min(s_dot_final[i], v_back)

        return s_dot_final

    def _compute_velocity_profile(self):
        """
        Compute the time-optimal velocity profile using forward-backward integration.
        """
        # Forward pass: accelerate as much as possible
        s_dot_forward = self._forward_integration()

        # Backward pass: ensure we can decelerate
        self.s_dot_profile = self._backward_integration(s_dot_forward)

        # Compute acceleration profile
        n = len(self.s_values)
        self.s_ddot_profile = np.zeros(n)

        for i in range(1, n-1):
            ds = self.s_values[i+1] - self.s_values[i-1]
            if ds > 0:
                self.s_ddot_profile[i] = (
                    self.s_dot_profile[i+1]**2 - self.s_dot_profile[i-1]**2
                ) / (2 * ds)

        # Compute time profile
        self.time_profile = np.zeros(n)
        for i in range(1, n):
            ds = self.s_values[i] - self.s_values[i-1]
            avg_velocity = (self.s_dot_profile[i] + self.s_dot_profile[i-1]) / 2.0

            if avg_velocity > 1e-10:
                dt = ds / avg_velocity
            else:
                dt = ds / 0.1  # Default slow speed

            self.time_profile[i] = self.time_profile[i-1] + dt

        self.total_time = self.time_profile[-1]

    def generate(self) -> List[TrajectoryPoint]:
        """Generate the time-optimal trajectory."""
        if len(self.path) < 2:
            raise ValueError("Need at least 2 waypoints for trajectory generation")

        # Parameterize the path
        self._parameterize_path()

        # Compute velocity profile
        self._compute_velocity_profile()

        # Create interpolation functions for time-based evaluation
        self.s_of_t = interp1d(
            self.time_profile, self.s_values,
            kind='linear', fill_value='extrapolate'
        )
        self.s_dot_of_t = interp1d(
            self.time_profile, self.s_dot_profile,
            kind='linear', fill_value='extrapolate'
        )
        self.s_ddot_of_t = interp1d(
            self.time_profile, self.s_ddot_profile,
            kind='linear', fill_value='extrapolate'
        )

        # Sample trajectory at regular time intervals
        dt = self.constraints.min_time_step
        self.trajectory_points = []

        t = 0.0
        while t <= self.total_time:
            point = self.evaluate(t)
            self.trajectory_points.append(point)
            t += dt

        # Ensure final point is included
        if self.trajectory_points[-1].time < self.total_time:
            self.trajectory_points.append(self.evaluate(self.total_time))

        # Compute yaw from velocity
        self.compute_yaw_from_velocity()

        return self.trajectory_points

    def evaluate(self, t: float) -> TrajectoryPoint:
        """Evaluate trajectory at a specific time."""
        # Clamp time
        t = np.clip(t, 0, self.total_time)

        # Get path parameter at time t
        s = float(self.s_of_t(t))
        s_dot = float(self.s_dot_of_t(t))
        s_ddot = float(self.s_ddot_of_t(t))

        # Evaluate path
        position, q_prime, q_double_prime = self._evaluate_path(s)

        # Compute velocity: v = q' * s_dot
        velocity = q_prime * s_dot

        # Compute acceleration: a = q'' * s_dot^2 + q' * s_ddot
        acceleration = q_double_prime * (s_dot ** 2) + q_prime * s_ddot

        # Compute jerk (if needed)
        jerk = None  # Could be computed from third derivative

        return TrajectoryPoint(
            time=t,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk
        )

    def get_velocity_profile_plot_data(self) -> dict:
        """
        Get data for plotting the velocity profile.

        Returns:
            Dictionary with plot data
        """
        return {
            's_values': self.s_values,
            's_dot_profile': self.s_dot_profile,
            's_ddot_profile': self.s_ddot_profile,
            'time_profile': self.time_profile,
            'max_velocities': [self._compute_max_velocity(s) for s in self.s_values]
        }

    def optimize_waypoints(self, margin: float = 0.1):
        """
        Optimize waypoint placement for smoother trajectory.

        Args:
            margin: Margin for waypoint adjustment (fraction of segment length)
        """
        # This could implement waypoint adjustment to reduce acceleration peaks
        # For now, we keep the original waypoints
        pass
