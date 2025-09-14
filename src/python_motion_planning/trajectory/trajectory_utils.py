import numpy as np
from typing import List, Tuple, Optional, Union
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
import math


def compute_velocity_profile(positions: np.ndarray,
                            times: np.ndarray) -> np.ndarray:
    """
    Compute velocity profile from position and time data.

    Args:
        positions: Array of positions, shape (n_points, dimension)
        times: Array of time stamps, shape (n_points,)

    Returns:
        Array of velocities, shape (n_points, dimension)
    """
    if len(positions) != len(times):
        raise ValueError("Positions and times must have same length")

    n_points = len(positions)
    dimension = positions.shape[1] if positions.ndim > 1 else 1

    if dimension == 1:
        positions = positions.reshape(-1, 1)

    velocities = np.zeros_like(positions)

    # Use central differences for interior points
    for i in range(1, n_points - 1):
        dt = times[i+1] - times[i-1]
        if dt > 0:
            velocities[i] = (positions[i+1] - positions[i-1]) / dt

    # Use forward difference for first point
    if n_points > 1:
        dt = times[1] - times[0]
        if dt > 0:
            velocities[0] = (positions[1] - positions[0]) / dt

    # Use backward difference for last point
    if n_points > 1:
        dt = times[-1] - times[-2]
        if dt > 0:
            velocities[-1] = (positions[-1] - positions[-2]) / dt

    return velocities


def compute_acceleration_profile(velocities: np.ndarray,
                                times: np.ndarray) -> np.ndarray:
    """
    Compute acceleration profile from velocity and time data.

    Args:
        velocities: Array of velocities, shape (n_points, dimension)
        times: Array of time stamps, shape (n_points,)

    Returns:
        Array of accelerations, shape (n_points, dimension)
    """
    if len(velocities) != len(times):
        raise ValueError("Velocities and times must have same length")

    n_points = len(velocities)
    dimension = velocities.shape[1] if velocities.ndim > 1 else 1

    if dimension == 1:
        velocities = velocities.reshape(-1, 1)

    accelerations = np.zeros_like(velocities)

    # Use central differences for interior points
    for i in range(1, n_points - 1):
        dt = times[i+1] - times[i-1]
        if dt > 0:
            accelerations[i] = (velocities[i+1] - velocities[i-1]) / dt

    # Use forward difference for first point
    if n_points > 1:
        dt = times[1] - times[0]
        if dt > 0:
            accelerations[0] = (velocities[1] - velocities[0]) / dt

    # Use backward difference for last point
    if n_points > 1:
        dt = times[-1] - times[-2]
        if dt > 0:
            accelerations[-1] = (velocities[-1] - velocities[-2]) / dt

    return accelerations


def interpolate_path(path: List[Tuple],
                    num_points: Optional[int] = None,
                    resolution: Optional[float] = None,
                    method: str = 'cubic') -> np.ndarray:
    """
    Interpolate a path to increase point density.

    Args:
        path: List of waypoints
        num_points: Target number of points (if specified)
        resolution: Target resolution (distance between points)
        method: Interpolation method ('linear', 'cubic', 'quintic')

    Returns:
        Interpolated path as numpy array
    """
    path_array = np.array(path)
    n_waypoints = len(path_array)

    if n_waypoints < 2:
        return path_array

    # Compute arc length parameterization
    distances = np.zeros(n_waypoints)
    for i in range(1, n_waypoints):
        distances[i] = distances[i-1] + np.linalg.norm(
            path_array[i] - path_array[i-1]
        )

    total_length = distances[-1]

    # Determine number of interpolation points
    if num_points is not None:
        n_interp = num_points
    elif resolution is not None and resolution > 0:
        n_interp = int(total_length / resolution) + 1
    else:
        # Default: 10 points per unit length
        n_interp = max(int(total_length * 10), 100)

    # Create interpolation parameter values
    s_interp = np.linspace(0, total_length, n_interp)

    # Interpolate each dimension
    dimension = path_array.shape[1]
    interpolated = np.zeros((n_interp, dimension))

    for dim in range(dimension):
        if method == 'cubic' and n_waypoints >= 4:
            interp_func = CubicSpline(distances, path_array[:, dim])
        elif method == 'quintic' and n_waypoints >= 6:
            from scipy.interpolate import UnivariateSpline
            interp_func = UnivariateSpline(
                distances, path_array[:, dim], k=5, s=0
            )
        else:
            # Fall back to linear for insufficient points
            interp_func = interp1d(
                distances, path_array[:, dim],
                kind='linear', fill_value='extrapolate'
            )

        interpolated[:, dim] = interp_func(s_interp)

    return interpolated


def smooth_path(path: Union[List[Tuple], np.ndarray],
                method: str = 'savgol',
                window_size: Optional[int] = None,
                poly_order: int = 3,
                sigma: float = 1.0) -> np.ndarray:
    """
    Smooth a path using various filtering methods.

    Args:
        path: Path to smooth
        method: Smoothing method ('savgol', 'moving_average', 'gaussian')
        window_size: Window size for filtering (auto if None)
        poly_order: Polynomial order for Savitzky-Golay filter
        sigma: Standard deviation for Gaussian filter

    Returns:
        Smoothed path as numpy array
    """
    path_array = np.array(path) if isinstance(path, list) else path.copy()
    n_points = len(path_array)

    if n_points < 3:
        return path_array

    # Auto-determine window size if not specified
    if window_size is None:
        window_size = min(n_points // 4, 21)
        if window_size % 2 == 0:
            window_size += 1  # Ensure odd window size
        window_size = max(3, window_size)

    dimension = path_array.shape[1] if path_array.ndim > 1 else 1

    if dimension == 1:
        path_array = path_array.reshape(-1, 1)

    smoothed = np.zeros_like(path_array)

    for dim in range(dimension):
        if method == 'savgol':
            # Savitzky-Golay filter
            if window_size > n_points:
                window_size = n_points if n_points % 2 == 1 else n_points - 1
            poly_order = min(poly_order, window_size - 1)
            smoothed[:, dim] = savgol_filter(
                path_array[:, dim], window_size, poly_order
            )

        elif method == 'moving_average':
            # Simple moving average
            smoothed[:, dim] = uniform_filter1d(
                path_array[:, dim], size=window_size, mode='nearest'
            )

        elif method == 'gaussian':
            # Gaussian filter
            from scipy.ndimage import gaussian_filter1d
            smoothed[:, dim] = gaussian_filter1d(
                path_array[:, dim], sigma=sigma, mode='nearest'
            )

        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    # Ensure endpoints remain unchanged
    smoothed[0] = path_array[0]
    smoothed[-1] = path_array[-1]

    return smoothed


def compute_path_curvature(path: np.ndarray) -> np.ndarray:
    """
    Compute curvature at each point along the path.

    Args:
        path: Path array, shape (n_points, dimension)

    Returns:
        Array of curvature values, shape (n_points,)
    """
    n_points = len(path)
    curvatures = np.zeros(n_points)

    for i in range(1, n_points - 1):
        # Use three consecutive points
        p0 = path[i-1]
        p1 = path[i]
        p2 = path[i+1]

        # Compute vectors
        v1 = p1 - p0
        v2 = p2 - p1

        # Compute curvature using cross product
        cross = np.cross(v1, v2) if len(p0) >= 2 else 0
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))

        if denom > 1e-10:
            if isinstance(cross, np.ndarray):
                curvatures[i] = np.linalg.norm(cross) / denom
            else:
                curvatures[i] = abs(cross) / denom

    return curvatures


def compute_path_length(path: Union[List[Tuple], np.ndarray]) -> float:
    """
    Compute the total length of a path.

    Args:
        path: Path as list of tuples or numpy array

    Returns:
        Total path length
    """
    path_array = np.array(path) if isinstance(path, list) else path

    length = 0.0
    for i in range(1, len(path_array)):
        length += np.linalg.norm(path_array[i] - path_array[i-1])

    return length


def resample_path_uniform(path: Union[List[Tuple], np.ndarray],
                         num_points: int) -> np.ndarray:
    """
    Resample a path with uniform spacing.

    Args:
        path: Original path
        num_points: Number of points in resampled path

    Returns:
        Resampled path with uniform spacing
    """
    path_array = np.array(path) if isinstance(path, list) else path

    if len(path_array) < 2:
        return path_array

    # Compute cumulative distances
    n_original = len(path_array)
    distances = np.zeros(n_original)

    for i in range(1, n_original):
        distances[i] = distances[i-1] + np.linalg.norm(
            path_array[i] - path_array[i-1]
        )

    total_length = distances[-1]

    # Create uniform spacing
    uniform_distances = np.linspace(0, total_length, num_points)

    # Interpolate positions
    dimension = path_array.shape[1] if path_array.ndim > 1 else 1
    resampled = np.zeros((num_points, dimension))

    for dim in range(dimension):
        interp_func = interp1d(
            distances, path_array[:, dim],
            kind='linear', fill_value='extrapolate'
        )
        resampled[:, dim] = interp_func(uniform_distances)

    return resampled


def remove_redundant_points(path: Union[List[Tuple], np.ndarray],
                           tolerance: float = 0.01) -> np.ndarray:
    """
    Remove redundant points that are collinear within tolerance.

    Args:
        path: Original path
        tolerance: Maximum deviation from straight line

    Returns:
        Simplified path with redundant points removed
    """
    path_array = np.array(path) if isinstance(path, list) else path

    if len(path_array) <= 2:
        return path_array

    kept_indices = [0]  # Always keep first point

    i = 0
    while i < len(path_array) - 1:
        # Look ahead to find the furthest point we can reach
        # while maintaining straight line within tolerance
        j = i + 2

        while j < len(path_array):
            # Check if all points between i and j are within tolerance
            # of the straight line from path[i] to path[j]

            p1 = path_array[i]
            p2 = path_array[j]

            all_within_tolerance = True

            for k in range(i + 1, j):
                # Compute distance from point to line
                p = path_array[k]

                # Vector from p1 to p2
                v = p2 - p1
                v_norm = np.linalg.norm(v)

                if v_norm > 1e-10:
                    # Project p onto line
                    t = np.dot(p - p1, v) / (v_norm ** 2)
                    t = np.clip(t, 0, 1)
                    projection = p1 + t * v

                    # Distance from point to line
                    dist = np.linalg.norm(p - projection)

                    if dist > tolerance:
                        all_within_tolerance = False
                        break

            if all_within_tolerance:
                j += 1
            else:
                break

        # Keep the point just before we exceeded tolerance
        i = j - 1
        kept_indices.append(i)

    # Always keep last point if not already included
    if kept_indices[-1] != len(path_array) - 1:
        kept_indices.append(len(path_array) - 1)

    return path_array[kept_indices]


def estimate_derivatives(path: np.ndarray,
                        dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate velocity and acceleration from discrete path points.

    Args:
        path: Path array, shape (n_points, dimension)
        dt: Time step between points

    Returns:
        Tuple of (velocities, accelerations)
    """
    n_points = len(path)
    dimension = path.shape[1] if path.ndim > 1 else 1

    velocities = np.zeros_like(path)
    accelerations = np.zeros_like(path)

    # Compute velocities using central differences
    for i in range(n_points):
        if i == 0 and n_points > 1:
            # Forward difference
            velocities[i] = (path[1] - path[0]) / dt
        elif i == n_points - 1 and n_points > 1:
            # Backward difference
            velocities[i] = (path[-1] - path[-2]) / dt
        elif n_points > 2:
            # Central difference
            velocities[i] = (path[i+1] - path[i-1]) / (2 * dt)

    # Compute accelerations
    for i in range(n_points):
        if i == 0 and n_points > 1:
            # Forward difference
            accelerations[i] = (velocities[1] - velocities[0]) / dt
        elif i == n_points - 1 and n_points > 1:
            # Backward difference
            accelerations[i] = (velocities[-1] - velocities[-2]) / dt
        elif n_points > 2:
            # Central difference
            accelerations[i] = (velocities[i+1] - velocities[i-1]) / (2 * dt)

    return velocities, accelerations
