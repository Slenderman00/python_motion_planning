"""
@file: env3d.py
@breif: 3-dimension environment
@author: Winter, Heimonen
@update: 2025.9.6
"""
from math import sqrt
from abc import ABC, abstractmethod
from scipy.spatial import cKDTree
import numpy as np

from .node3d import Node3D

class Env3D(ABC):
    """
    Class for building 3-d workspace of robots.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
        z_range (int): z-axis range of enviroment
        eps (float): tolerance for float comparison

    Examples:
        >>> from python_motion_planning.utils import Env3d
        >>> env = Env(30, 40, 50)
    """
    def __init__(self, x_range: int, y_range: int, z_range: int, eps: float = 1e-6) -> None:
        # size of environment
        self.x_range = x_range  
        self.y_range = y_range
        self.z_range = z_range
        self.eps = eps

    @property
    def grid_map(self) -> set:
        return {(i, j, o) for i in range(self.x_range) for j in range(self.y_range) for o in range(self.z_range)}

    @abstractmethod
    def init(self) -> None:
        pass

class Grid3D(Env3D):
    """
    Class for discrete 3-d grid map.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
        z_range (int): z-axis range of enviroment
    """
    def __init__(self, x_range: int, y_range: int, z_range: int) -> None:
        super().__init__(x_range, y_range, z_range)
        
        # allowed motions
        self.motions = [
            Node3D((-1, 0, 0), None, 1, None), Node3D((-1, 1, 0), None, sqrt(2), None),
            Node3D((0, 1, 0), None, 1, None), Node3D((1, 1, 0), None, sqrt(2), None),
            Node3D((1, 0, 0), None, 1, None), Node3D((1, -1, 0), None, sqrt(2), None),
            Node3D((0, -1, 0), None, 1, None), Node3D((-1, -1, 0), None, sqrt(2), None),
            Node3D((0, 0, 1), None, 1, None), Node3D((0, 0, -1), None, 1, None),
            Node3D((-1, 0, 1), None, sqrt(2), None), Node3D((-1, 1, 1), None, sqrt(3), None),
            Node3D((0, 1, 1), None, sqrt(2), None), Node3D((1, 1, 1), None, sqrt(3), None),
            Node3D((1, 0, 1), None, sqrt(2), None), Node3D((1, -1, 1), None, sqrt(3), None),
            Node3D((0, -1, 1), None, sqrt(2), None), Node3D((-1, -1, 1), None, sqrt(3), None),
            Node3D((-1, 0, -1), None, sqrt(2), None), Node3D((-1, 1, -1), None, sqrt(3), None),
            Node3D((0, 1, -1), None, sqrt(2), None), Node3D((1, 1, -1), None, sqrt(3), None),
            Node3D((1, 0, -1), None, sqrt(2), None), Node3D((1, -1, -1), None, sqrt(3), None),
            Node3D((0, -1, -1), None, sqrt(2), None), Node3D((-1, -1, -1), None, sqrt(3), None)
        ]
        
        # obstacles
        self.obstacles = None
        self.obstacles_tree = None
        self.init()
    
    def init(self) -> None:
        """
        Initialize grid map.
        """
        x, y, z = self.x_range, self.y_range, self.z_range
        obstacles = set()

        # boundary of environment
        # TODO: We also need to create a top and bottom border
        for _z in range(z):
            for i in range(x):
                obstacles.add((i, 0, _z - 1))
                obstacles.add((i, y - 1, _z - 1))
            for i in range(y):
                obstacles.add((0, i, _z -1))
                obstacles.add((x - 1, i, _z - 1))

        self.update(obstacles)

    def update(self, obstacles):
        self.obstacles = obstacles 
        self.obstacles_tree = cKDTree(np.array(list(obstacles)))


class Map3D(Env3D):
    """
    Class for continuous 3-d map.

    Parameters:
        x_range (int): x-axis range of enviroment
        y_range (int): y-axis range of environmet
        z_range (int): z-axis range of enviroment
    """
    def __init__(self, x_range: int, y_range: int, z_range: int) -> None:
        super().__init__(x_range, y_range, z_range)
        self.boundary = None
        self.obs_circ = None
        self.obs_rect = None
        self.init()

    def init(self):
        """
        Initialize map.
        """
        x, y, z = self.x_range, self.y_range, self.z_range
        # TODO: what the fuck is this?
        # boundary of environment
        self.boundary = [
            [0, 0, 0, x, y, 0],
            [0, 0, z, x, y, z],
            [0, 0, 0, x, 0, z],
            [0, y, 0, x, y, z],
            [0, 0, 0, 0, y, z],
            [x, 0, 0, x, y, z]
        ]
        self.obs_rect = []
        self.obs_circ = []

    def update(self, boundary=None, obs_circ=None, obs_rect=None):
        self.boundary = boundary if boundary else self.boundary
        self.obs_circ = obs_circ if obs_circ else self.obs_circ
        self.obs_rect = obs_rect if obs_rect else self.obs_rect
