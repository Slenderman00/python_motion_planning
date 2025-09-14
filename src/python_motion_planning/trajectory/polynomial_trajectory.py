import numpy as np
from typing import List, Tuple, Optional
from .trajectory_base import TrajectoryBase, Trajectory3D, TrajectoryPoint, TrajectoryConstraints


class PolynomialSegment:
    """Represents a polynomial segment between two waypoints."""

    def __init__(self, coeffs: np.ndarray, duration: float, start_time: float = 0.0):
        """
        Initialize polynomial segment.

        Args:
            coeffs: Polynomial coefficients [a0, a1, a2, a3, a4, a5] for each dimension
                   Shape: (dimension, 6) for quintic polynomial
            duration: Duration of this segment
            start_time: Start time of this segment in the overall trajectory
        """
        self.coeffs = coeffs
        self.duration = duration
        self.start_time = start_time
        self.dimension = coeffs.shape[0]
        self.order = coeffs.shape[1] - 1

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate polynomial at time t.

        Args:
            t: Time within segment (0 <= t <= duration)

        Returns:
            Tuple of (position, velocity, acceleration, jerk)
        """
        # Clamp t to segment duration
        t = np.clip(t, 0, self.duration)

        # Time powers for quintic polynomial
        t_vec = np.array([1, t, t**2, t**3, t**4, t**5])
        dt_vec = np.array([0, 1, 2*t, 3*t**2, 4*t**3, 5*t**4])
        ddt_vec = np.array([0, 0, 2, 6*t, 12*t**2, 20*t**3])
        dddt_vec = np.array([0, 0, 0, 6, 24*t, 60*t**2])

        position = self.coeffs @ t_vec
        velocity = self.coeffs @ dt_vec
        acceleration = self.coeffs @ ddt_vec
        jerk = self.coeffs @ dddt_vec

        return position, velocity, acceleration, jerk


class PolynomialTrajectory3D(Trajectory3D):
    """
    Generates smooth polynomial trajectories through waypoints.

    Uses quintic polynomials to ensure continuous position, velocity,
    and acceleration at waypoints.
    """

    def __init__(self,
                 path: List[Tuple],
                 constraints: Optional[TrajectoryConstraints] = None,
                 boundary_conditions: Optional[dict] = None):
        """
        Initialize polynomial trajectory generator.

        Args:
            path: List of waypoints
            constraints: Physical constraints
            boundary_conditions: Initial and final conditions
                {'initial_velocity': np.array, 'final_velocity': np.array,
                 'initial_acceleration': np.array, 'final_acceleration': np.array}
        """
        super().__init__(path, constraints)

        # Default boundary conditions (start and end at rest)
        self.boundary_conditions = boundary_conditions or {}
        if 'initial_velocity' not in self.boundary_conditions:
            self.boundary_conditions['initial_velocity'] = np.zeros(3)
        if 'final_velocity' not in self.boundary_conditions:
            self.boundary_conditions['final_velocity'] = np.zeros(3)
        if 'initial_acceleration' not in self.boundary_conditions:
            self.boundary_conditions['initial_acceleration'] = np.zeros(3)
        if 'final_acceleration' not in self.boundary_conditions:
            self.boundary_conditions['final_acceleration'] = np.zeros(3)

        self.segments: List[PolynomialSegment] = []

    def _compute_segment_times(self) -> List[float]:
        """
        Compute time allocation for each segment based on distance and constraints.

        Returns:
            List of segment durations
        """
        segment_times = []

        for i in range(len(self.path) - 1):
            # Distance between waypoints
            distance = np.linalg.norm(self.path[i+1] - self.path[i])

            # Estimate time based on max velocity (with some margin for acceleration)
            avg_max_velocity = np.min(self.constraints.max_velocity) * 0.7
            time_estimate = distance / avg_max_velocity if avg_max_velocity > 0 else 1.0

            # Ensure minimum time for acceleration/deceleration
            min_time = 2.0 * np.sqrt(distance / np.min(self.constraints.max_acceleration))

            segment_times.append(max(time_estimate, min_time, 0.5))

        return segment_times

    def _solve_quintic_coefficients(self,
                                   p0: np.ndarray, p1: np.ndarray,
                                   v0: np.ndarray, v1: np.ndarray,
                                   a0: np.ndarray, a1: np.ndarray,
                                   T: float) -> np.ndarray:
        """
        Solve for quintic polynomial coefficients.

        The polynomial is: p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5

        Args:
            p0, p1: Initial and final positions
            v0, v1: Initial and final velocities
            a0, a1: Initial and final accelerations
            T: Segment duration

        Returns:
            Coefficients array of shape (dimension, 6)
        """
        # Solve for each dimension independently
        coeffs = np.zeros((self.dimension, 6))

        for dim in range(self.dimension):
            # Boundary conditions matrix
            # p(0) = p0, p(T) = p1
            # v(0) = v0, v(T) = v1
            # a(0) = a0, a(T) = a1

            # Direct solution for quintic polynomial
            coeffs[dim, 0] = p0[dim]
            coeffs[dim, 1] = v0[dim]
            coeffs[dim, 2] = a0[dim] / 2.0

            if T > 0:
                T2 = T * T
                T3 = T2 * T
                T4 = T3 * T
                T5 = T4 * T

                # Solve for remaining coefficients
                coeffs[dim, 3] = (20*p1[dim] - 20*p0[dim] - (8*v1[dim] + 12*v0[dim])*T -
                                 (3*a0[dim] - a1[dim])*T2) / (2*T3)
                coeffs[dim, 4] = (30*p0[dim] - 30*p1[dim] + (14*v1[dim] + 16*v0[dim])*T +
                                 (3*a0[dim] - 2*a1[dim])*T2) / (2*T4)
                coeffs[dim, 5] = (12*p1[dim] - 12*p0[dim] - (6*v1[dim] + 6*v0[dim])*T -
                                 (a0[dim] - a1[dim])*T2) / (2*T5)

        return coeffs

    def _generate_waypoint_velocities(self) -> List[np.ndarray]:
        """
        Generate velocities at intermediate waypoints using finite differences.

        Returns:
            List of velocities at each waypoint
        """
        velocities = []
        n = len(self.path)

        # Initial velocity
        velocities.append(self.boundary_conditions['initial_velocity'])

        # Intermediate waypoints
        for i in range(1, n - 1):
            # Use centered finite difference
            v = (self.path[i+1] - self.path[i-1]) / (self.segment_times[i] + self.segment_times[i-1])

            # Clamp to constraints
            v = np.clip(v, -self.constraints.max_velocity, self.constraints.max_velocity)
            velocities.append(v)

        # Final velocity
        velocities.append(self.boundary_conditions['final_velocity'])

        return velocities

    def generate(self) -> List[TrajectoryPoint]:
        """Generate the polynomial trajectory."""
        if len(self.path) < 2:
            raise ValueError("Need at least 2 waypoints for trajectory generation")

        # Compute segment times
        self.segment_times = self._compute_segment_times()
        self.total_time = sum(self.segment_times)

        # Generate velocities at waypoints
        waypoint_velocities = self._generate_waypoint_velocities()

        # Generate polynomial segments
        self.segments = []
        current_time = 0.0

        for i in range(len(self.path) - 1):
            # Positions
            p0 = self.path[i]
            p1 = self.path[i + 1]

            # Velocities
            v0 = waypoint_velocities[i]
            v1 = waypoint_velocities[i + 1]

            # Accelerations (zero at intermediate waypoints for smoothness)
            if i == 0:
                a0 = self.boundary_conditions['initial_acceleration']
            else:
                a0 = np.zeros(self.dimension)

            if i == len(self.path) - 2:
                a1 = self.boundary_conditions['final_acceleration']
            else:
                a1 = np.zeros(self.dimension)

            # Solve for polynomial coefficients
            T = self.segment_times[i]
            coeffs = self._solve_quintic_coefficients(p0, p1, v0, v1, a0, a1, T)

            # Create segment
            segment = PolynomialSegment(coeffs, T, current_time)
            self.segments.append(segment)
            current_time += T

        # Sample trajectory at regular intervals
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
        if not self.segments:
            self.generate()

        # Clamp time
        t = np.clip(t, 0, self.total_time)

        # Find the appropriate segment
        segment_idx = 0
        local_t = t

        for i, segment in enumerate(self.segments):
            if t <= segment.start_time + segment.duration:
                segment_idx = i
                local_t = t - segment.start_time
                break
        else:
            # Use last segment
            segment_idx = len(self.segments) - 1
            segment = self.segments[segment_idx]
            local_t = segment.duration

        # Evaluate polynomial
        segment = self.segments[segment_idx]
        position, velocity, acceleration, jerk = segment.evaluate(local_t)

        return TrajectoryPoint(
            time=t,
            position=position,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk
        )

    def optimize_time_allocation(self, max_iterations: int = 10):
        """
        Iteratively optimize time allocation to better respect constraints.

        Args:
            max_iterations: Maximum optimization iterations
        """
        for iteration in range(max_iterations):
            # Generate trajectory with current time allocation
            self.generate()

            # Check constraint violations
            constraint_status = self.check_constraints()

            if all(constraint_status.values()):
                break

            # Adjust segment times based on violations
            velocities = self.get_velocities()
            accelerations = self.get_accelerations()

            for i, segment in enumerate(self.segments):
                # Find max violation ratio in this segment
                start_idx = int(segment.start_time / self.constraints.min_time_step)
                end_idx = min(
                    int((segment.start_time + segment.duration) / self.constraints.min_time_step),
                    len(velocities) - 1
                )

                if end_idx > start_idx:
                    seg_velocities = velocities[start_idx:end_idx+1]
                    seg_accelerations = accelerations[start_idx:end_idx+1]

                    # Compute violation ratios
                    v_ratio = np.max(np.abs(seg_velocities) / self.constraints.max_velocity)
                    a_ratio = np.max(np.abs(seg_accelerations) / self.constraints.max_acceleration)

                    # Increase time if constraints are violated
                    max_ratio = max(v_ratio, a_ratio)
                    if max_ratio > 1.0:
                        self.segment_times[i] *= (max_ratio ** 0.5)  # Use square root for gradual adjustment

            self.total_time = sum(self.segment_times)
