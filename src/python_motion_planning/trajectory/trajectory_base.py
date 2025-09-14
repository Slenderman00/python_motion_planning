import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import math


@dataclass
class TrajectoryPoint:
    """Single point in a trajectory with full state information."""
    time: float
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    acceleration: np.ndarray  # [ax, ay, az]
    jerk: Optional[np.ndarray] = None  # [jx, jy, jz]
    yaw: Optional[float] = None  # orientation (for robots)
    yaw_rate: Optional[float] = None  # angular velocity

    def to_tuple(self) -> Tuple:
        """Convert to tuple format for compatibility."""
        return tuple(self.position)

    def __repr__(self) -> str:
        return f"TrajectoryPoint(t={self.time:.2f}, pos={self.position})"


@dataclass
class TrajectoryConstraints:
    """Constraints for trajectory generation."""
    max_velocity: np.ndarray  # [vx_max, vy_max, vz_max]
    max_acceleration: np.ndarray  # [ax_max, ay_max, az_max]
    max_jerk: Optional[np.ndarray] = None  # [jx_max, jy_max, jz_max]
    min_time_step: float = 0.01  # minimum time step for discretization

    def __post_init__(self):
        """Ensure arrays are numpy arrays."""
        self.max_velocity = np.asarray(self.max_velocity)
        self.max_acceleration = np.asarray(self.max_acceleration)
        if self.max_jerk is not None:
            self.max_jerk = np.asarray(self.max_jerk)


class TrajectoryBase(ABC):
    """
    Base class for trajectory planning.

    This class provides the interface and common functionality for
    generating smooth trajectories from paths.
    """

    def __init__(self,
                 path: List[Tuple],
                 constraints: Optional[TrajectoryConstraints] = None,
                 dimension: int = 3):
        """
        Initialize trajectory base.

        Args:
            path: List of waypoints [(x,y,z), ...] or [(x,y), ...]
            constraints: Physical constraints for the trajectory
            dimension: Spatial dimension (2 or 3)
        """
        self.dimension = dimension
        self.path = self._process_path(path)

        # Default constraints if not provided
        if constraints is None:
            # Default to reasonable values
            max_v = np.array([2.0] * dimension)
            max_a = np.array([1.0] * dimension)
            self.constraints = TrajectoryConstraints(max_v, max_a)
        else:
            self.constraints = constraints

        self.trajectory_points: List[TrajectoryPoint] = []
        self.total_time: float = 0.0
        self.segment_times: List[float] = []

    def _process_path(self, path: List[Tuple]) -> np.ndarray:
        """Convert path to numpy array, handling 2D/3D conversion."""
        if not path:
            raise ValueError("Path cannot be empty")

        path_array = []
        for point in path:
            if len(point) == 2 and self.dimension == 3:
                # Convert 2D to 3D by adding z=0
                path_array.append([point[0], point[1], 0.0])
            elif len(point) == 3 and self.dimension == 2:
                # Convert 3D to 2D by dropping z
                path_array.append([point[0], point[1]])
            else:
                path_array.append(list(point))

        return np.array(path_array, dtype=float)

    @abstractmethod
    def generate(self) -> List[TrajectoryPoint]:
        """
        Generate the trajectory from the path.

        Returns:
            List of TrajectoryPoint objects representing the trajectory
        """
        pass

    @abstractmethod
    def evaluate(self, t: float) -> TrajectoryPoint:
        """
        Evaluate trajectory at a specific time.

        Args:
            t: Time in seconds

        Returns:
            TrajectoryPoint at time t
        """
        pass

    def get_positions(self) -> np.ndarray:
        """Get all positions from the trajectory."""
        if not self.trajectory_points:
            self.generate()
        return np.array([tp.position for tp in self.trajectory_points])

    def get_velocities(self) -> np.ndarray:
        """Get all velocities from the trajectory."""
        if not self.trajectory_points:
            self.generate()
        return np.array([tp.velocity for tp in self.trajectory_points])

    def get_accelerations(self) -> np.ndarray:
        """Get all accelerations from the trajectory."""
        if not self.trajectory_points:
            self.generate()
        return np.array([tp.acceleration for tp in self.trajectory_points])

    def get_times(self) -> np.ndarray:
        """Get all time stamps from the trajectory."""
        if not self.trajectory_points:
            self.generate()
        return np.array([tp.time for tp in self.trajectory_points])

    def get_path_length(self) -> float:
        """Calculate the total path length."""
        length = 0.0
        for i in range(1, len(self.path)):
            length += np.linalg.norm(self.path[i] - self.path[i-1])
        return length

    def check_constraints(self) -> Dict[str, bool]:
        """
        Check if trajectory satisfies constraints.

        Returns:
            Dictionary with constraint satisfaction status
        """
        if not self.trajectory_points:
            self.generate()

        velocities = self.get_velocities()
        accelerations = self.get_accelerations()

        results = {
            'velocity_satisfied': True,
            'acceleration_satisfied': True,
            'jerk_satisfied': True
        }

        # Check velocity constraints
        max_v = np.max(np.abs(velocities), axis=0)
        if np.any(max_v > self.constraints.max_velocity):
            results['velocity_satisfied'] = False

        # Check acceleration constraints
        max_a = np.max(np.abs(accelerations), axis=0)
        if np.any(max_a > self.constraints.max_acceleration):
            results['acceleration_satisfied'] = False

        # Check jerk constraints if available
        if self.constraints.max_jerk is not None:
            jerks = []
            for tp in self.trajectory_points:
                if tp.jerk is not None:
                    jerks.append(tp.jerk)
            if jerks:
                max_j = np.max(np.abs(np.array(jerks)), axis=0)
                if np.any(max_j > self.constraints.max_jerk):
                    results['jerk_satisfied'] = False

        return results

    def resample(self, dt: float) -> List[TrajectoryPoint]:
        """
        Resample trajectory at a fixed time step.

        Args:
            dt: Time step for resampling

        Returns:
            Resampled trajectory points
        """
        if not self.trajectory_points:
            self.generate()

        resampled = []
        t = 0.0
        while t <= self.total_time:
            resampled.append(self.evaluate(t))
            t += dt

        # Ensure we include the final point
        if resampled[-1].time < self.total_time:
            resampled.append(self.evaluate(self.total_time))

        return resampled

    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary for serialization."""
        if not self.trajectory_points:
            self.generate()

        return {
            'path': self.path.tolist(),
            'total_time': self.total_time,
            'trajectory': [
                {
                    'time': tp.time,
                    'position': tp.position.tolist(),
                    'velocity': tp.velocity.tolist(),
                    'acceleration': tp.acceleration.tolist(),
                    'jerk': tp.jerk.tolist() if tp.jerk is not None else None,
                    'yaw': tp.yaw,
                    'yaw_rate': tp.yaw_rate
                }
                for tp in self.trajectory_points
            ],
            'constraints': {
                'max_velocity': self.constraints.max_velocity.tolist(),
                'max_acceleration': self.constraints.max_acceleration.tolist(),
                'max_jerk': self.constraints.max_jerk.tolist() if self.constraints.max_jerk is not None else None
            }
        }

    def compute_yaw_from_velocity(self):
        """Compute yaw angle from velocity direction for each trajectory point."""
        for tp in self.trajectory_points:
            if np.linalg.norm(tp.velocity[:2]) > 1e-6:  # Check if there's motion in x-y plane
                tp.yaw = np.arctan2(tp.velocity[1], tp.velocity[0])

        # Compute yaw rate
        for i in range(1, len(self.trajectory_points)):
            dt = self.trajectory_points[i].time - self.trajectory_points[i-1].time
            if dt > 0 and self.trajectory_points[i].yaw is not None and self.trajectory_points[i-1].yaw is not None:
                dyaw = self.trajectory_points[i].yaw - self.trajectory_points[i-1].yaw
                # Handle angle wrapping
                while dyaw > math.pi:
                    dyaw -= 2 * math.pi
                while dyaw < -math.pi:
                    dyaw += 2 * math.pi
                self.trajectory_points[i].yaw_rate = dyaw / dt


class Trajectory3D(TrajectoryBase):
    """Specific class for 3D trajectories."""

    def __init__(self,
                 path: List[Tuple],
                 constraints: Optional[TrajectoryConstraints] = None):
        """
        Initialize 3D trajectory.

        Args:
            path: List of 3D waypoints [(x,y,z), ...]
            constraints: Physical constraints for the trajectory
        """
        super().__init__(path, constraints, dimension=3)

    def get_plot_data(self) -> Dict[str, np.ndarray]:
        """
        Get data formatted for Plot3D visualization.

        Returns:
            Dictionary with visualization data
        """
        if not self.trajectory_points:
            self.generate()

        positions = self.get_positions()
        velocities = self.get_velocities()
        times = self.get_times()

        # Create velocity magnitude for coloring
        velocity_magnitude = np.linalg.norm(velocities, axis=1)

        return {
            'positions': positions,
            'velocities': velocities,
            'velocity_magnitude': velocity_magnitude,
            'times': times,
            'path': self.path
        }
