import numpy as np
from typing import List, Tuple, Optional
from scipy import interpolate
from scipy.optimize import minimize_scalar
from .trajectory_base import Trajectory3D, TrajectoryPoint, TrajectoryConstraints


class SplineTrajectory3D(Trajectory3D):
    """
    Generates smooth trajectories using B-splines.

    B-splines provide excellent smoothness properties and local control,
    making them ideal for trajectory generation in robotics.
    """

    def __init__(self,
                 path: List[Tuple],
                 constraints: Optional[TrajectoryConstraints] = None,
                 spline_degree: int = 5,
                 smoothing_factor: float = 0.0):
        """
        Initialize B-spline trajectory generator.

        Args:
            path: List of waypoints
            constraints: Physical constraints
            spline_degree: Degree of the B-spline (3-5 recommended)
            smoothing_factor: Smoothing factor for spline fitting (0 = interpolation)
        """
        super().__init__(path, constraints)
        self.spline_degree = min(spline_degree, len(path) - 1)  # Can't exceed number of points - 1
        self.smoothing_factor = smoothing_factor

        # Spline representations for each dimension
        self.splines = []
        self.spline_derivatives = []
        self.spline_second_derivatives = []

        # Parameter values for waypoints
        self.u_waypoints = None
        self.u_max = 1.0

    def _compute_chord_length_parameterization(self) -> np.ndarray:
        """
        Compute parameter values using chord length parameterization.

        Returns:
            Array of parameter values for each waypoint
        """
        distances = np.zeros(len(self.path))
        for i in range(1, len(self.path)):
            distances[i] = distances[i-1] + np.linalg.norm(self.path[i] - self.path[i-1])

        # Normalize to [0, 1]
        if distances[-1] > 0:
            return distances / distances[-1]
        else:
            return np.linspace(0, 1, len(self.path))

    def _compute_centripetal_parameterization(self) -> np.ndarray:
        """
        Compute parameter values using centripetal parameterization.
        Better for paths with varying curvature.

        Returns:
            Array of parameter values for each waypoint
        """
        distances = np.zeros(len(self.path))
        for i in range(1, len(self.path)):
            chord = np.linalg.norm(self.path[i] - self.path[i-1])
            distances[i] = distances[i-1] + np.sqrt(chord)

        # Normalize to [0, 1]
        if distances[-1] > 0:
            return distances / distances[-1]
        else:
            return np.linspace(0, 1, len(self.path))

    def _fit_splines(self):
        """Fit B-splines to the waypoints for each dimension."""
        # Use centripetal parameterization for better curve properties
        self.u_waypoints = self._compute_centripetal_parameterization()

        self.splines = []
        self.spline_derivatives = []
        self.spline_second_derivatives = []

        for dim in range(self.dimension):
            # Extract coordinates for this dimension
            coords = self.path[:, dim]

            # Create B-spline
            if self.smoothing_factor > 0:
                # Smoothing spline
                spline = interpolate.UnivariateSpline(
                    self.u_waypoints, coords,
                    k=self.spline_degree,
                    s=self.smoothing_factor
                )
            else:
                # Interpolating spline
                if len(self.path) > self.spline_degree:
                    spline = interpolate.UnivariateSpline(
                        self.u_waypoints, coords,
                        k=self.spline_degree,
                        s=0
                    )
                else:
                    # Use lower degree if not enough points
                    degree = min(self.spline_degree, len(self.path) - 1)
                    spline = interpolate.UnivariateSpline(
                        self.u_waypoints, coords,
                        k=degree,
                        s=0
                    )

            # Store spline and its derivatives
            self.splines.append(spline)
            self.spline_derivatives.append(spline.derivative(1))
            self.spline_second_derivatives.append(spline.derivative(2))

    def _compute_arc_length_parameterization(self, num_samples: int = 1000) -> interpolate.interp1d:
        """
        Compute arc length parameterization for constant speed.

        Args:
            num_samples: Number of samples for arc length computation

        Returns:
            Interpolation function from arc length to parameter u
        """
        # Sample the curve
        u_samples = np.linspace(0, 1, num_samples)
        arc_lengths = np.zeros(num_samples)

        for i in range(1, num_samples):
            u_prev = u_samples[i-1]
            u_curr = u_samples[i]

            # Compute positions
            pos_prev = np.array([spline(u_prev) for spline in self.splines])
            pos_curr = np.array([spline(u_curr) for spline in self.splines])

            # Accumulate arc length
            arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(pos_curr - pos_prev)

        # Normalize arc length
        if arc_lengths[-1] > 0:
            arc_lengths = arc_lengths / arc_lengths[-1]

        # Create interpolation from normalized arc length to u
        return interpolate.interp1d(arc_lengths, u_samples, kind='linear', fill_value='extrapolate')

    def _compute_time_parameterization(self) -> Tuple[float, interpolate.interp1d]:
        """
        Compute time parameterization respecting velocity constraints.

        Returns:
            Tuple of (total_time, time_to_parameter_function)
        """
        # Use simplified trapezoidal velocity profile
        num_samples = 500
        u_samples = np.linspace(0, 1, num_samples)

        # Compute velocities along the curve
        velocities = []
        for u in u_samples:
            # Compute tangent vector
            tangent = np.array([deriv(u) for deriv in self.spline_derivatives])
            speed = np.linalg.norm(tangent)

            if speed > 0:
                # Normalize and apply velocity constraints
                tangent_normalized = tangent / speed
                # Use minimum of constraints for each dimension
                max_speed = np.min(np.abs(self.constraints.max_velocity))
                velocities.append(min(speed, max_speed))
            else:
                velocities.append(0.0)

        velocities = np.array(velocities)

        # Compute time by integration
        times = np.zeros(num_samples)
        for i in range(1, num_samples):
            du = u_samples[i] - u_samples[i-1]
            avg_velocity = (velocities[i] + velocities[i-1]) / 2.0

            if avg_velocity > 0:
                # Arc length approximation
                pos_prev = np.array([spline(u_samples[i-1]) for spline in self.splines])
                pos_curr = np.array([spline(u_samples[i]) for spline in self.splines])
                ds = np.linalg.norm(pos_curr - pos_prev)
                dt = ds / avg_velocity
            else:
                dt = 0.1  # Default time step if stationary

            times[i] = times[i-1] + dt

        self.total_time = times[-1]

        # Create interpolation from time to parameter u
        return self.total_time, interpolate.interp1d(times, u_samples, kind='linear', fill_value='extrapolate')

    def generate(self) -> List[TrajectoryPoint]:
        """Generate the B-spline trajectory."""
        if len(self.path) < 2:
            raise ValueError("Need at least 2 waypoints for trajectory generation")

        # Fit splines to waypoints
        self._fit_splines()

        # Compute time parameterization
        self.total_time, time_to_u = self._compute_time_parameterization()

        # Sample trajectory at regular time intervals
        dt = self.constraints.min_time_step
        self.trajectory_points = []

        t = 0.0
        while t <= self.total_time:
            # Get parameter value for this time
            u = float(time_to_u(t))
            u = np.clip(u, 0, 1)

            # Evaluate splines
            position = np.array([spline(u) for spline in self.splines])

            # Compute velocity (chain rule: dq/dt = dq/du * du/dt)
            du_dt = 1.0 / self.total_time if self.total_time > 0 else 0.0
            tangent = np.array([deriv(u) for deriv in self.spline_derivatives])
            velocity = tangent * du_dt

            # Compute acceleration
            second_deriv = np.array([deriv2(u) for deriv2 in self.spline_second_derivatives])
            acceleration = second_deriv * (du_dt ** 2)

            # Create trajectory point
            point = TrajectoryPoint(
                time=t,
                position=position,
                velocity=velocity,
                acceleration=acceleration,
                jerk=None  # Could compute third derivative if needed
            )

            self.trajectory_points.append(point)
            t += dt

        # Ensure final point is included
        if self.trajectory_points[-1].time < self.total_time:
            u = 1.0
            position = np.array([spline(u) for spline in self.splines])
            velocity = np.zeros(self.dimension)  # Zero velocity at end
            acceleration = np.zeros(self.dimension)

            point = TrajectoryPoint(
                time=self.total_time,
                position=position,
                velocity=velocity,
                acceleration=acceleration
            )
            self.trajectory_points.append(point)

        # Post-process to ensure constraint satisfaction
        self._enforce_constraints()

        # Compute yaw from velocity
        self.compute_yaw_from_velocity()

        return self.trajectory_points

    def evaluate(self, t: float) -> TrajectoryPoint:
        """Evaluate trajectory at a specific time."""
        if not self.splines:
            self.generate()

        # Clamp time
        t = np.clip(t, 0, self.total_time)

        # Find the parameter value for this time (linear interpolation for now)
        u = t / self.total_time if self.total_time > 0 else 0.0
        u = np.clip(u, 0, 1)

        # Evaluate splines
        position = np.array([spline(u) for spline in self.splines])

        # Compute velocity
        du_dt = 1.0 / self.total_time if self.total_time > 0 else 0.0
        tangent = np.array([deriv(u) for deriv in self.spline_derivatives])
        velocity = tangent * du_dt

        # Compute acceleration
        second_deriv = np.array([deriv2(u) for deriv2 in self.spline_second_derivatives])
        acceleration = second_deriv * (du_dt ** 2)

        return TrajectoryPoint(
            time=t,
            position=position,
            velocity=velocity,
            acceleration=acceleration
        )

    def _enforce_constraints(self):
        """Post-process trajectory to enforce constraints."""
        velocities = self.get_velocities()
        accelerations = self.get_accelerations()

        # Find maximum violations
        max_v_ratio = np.max(np.abs(velocities) / self.constraints.max_velocity)
        max_a_ratio = np.max(np.abs(accelerations) / self.constraints.max_acceleration)

        if max_v_ratio > 1.0 or max_a_ratio > 1.0:
            # Scale time to satisfy constraints
            time_scale = max(max_v_ratio, np.sqrt(max_a_ratio))

            # Rescale trajectory in time
            self.total_time *= time_scale

            for point in self.trajectory_points:
                point.time *= time_scale
                point.velocity /= time_scale
                point.acceleration /= (time_scale ** 2)

    def reparameterize_constant_speed(self):
        """
        Reparameterize the trajectory for constant speed motion.
        Useful for some applications like painting or welding.
        """
        if not self.splines:
            self.generate()

        # Compute arc length parameterization
        arc_length_to_u = self._compute_arc_length_parameterization()

        # Compute constant speed
        total_length = self.get_path_length()
        constant_speed = min(
            total_length / self.total_time if self.total_time > 0 else 1.0,
            np.min(self.constraints.max_velocity)
        )

        # Recompute total time for constant speed
        self.total_time = total_length / constant_speed

        # Regenerate trajectory points
        dt = self.constraints.min_time_step
        self.trajectory_points = []

        t = 0.0
        while t <= self.total_time:
            # Arc length at time t
            s = (t / self.total_time) if self.total_time > 0 else 0.0

            # Get parameter value
            u = float(arc_length_to_u(s))
            u = np.clip(u, 0, 1)

            # Evaluate position
            position = np.array([spline(u) for spline in self.splines])

            # Constant velocity in direction of motion
            tangent = np.array([deriv(u) for deriv in self.spline_derivatives])
            if np.linalg.norm(tangent) > 0:
                velocity = (tangent / np.linalg.norm(tangent)) * constant_speed
            else:
                velocity = np.zeros(self.dimension)

            # Acceleration (approximately zero for constant speed)
            acceleration = np.zeros(self.dimension)

            point = TrajectoryPoint(
                time=t,
                position=position,
                velocity=velocity,
                acceleration=acceleration
            )

            self.trajectory_points.append(point)
            t += dt

        # Add final point
        if self.trajectory_points[-1].time < self.total_time:
            position = np.array([spline(1.0) for spline in self.splines])
            self.trajectory_points.append(
                TrajectoryPoint(
                    time=self.total_time,
                    position=position,
                    velocity=np.zeros(self.dimension),
                    acceleration=np.zeros(self.dimension)
                )
            )
