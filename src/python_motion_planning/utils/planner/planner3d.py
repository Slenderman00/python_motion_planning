"""
@file: planner3d.py
@breif: Abstract class for planner
@author: Winter, Joar Heimonen
@update: 2025.9.7
"""
import math
from abc import abstractmethod, ABC
from ..environment.env3d import Env3D, Node3D
from ..plot.plot3d import Plot3D

class Planner3D(ABC):
    def __init__(self, start: tuple, goal: tuple, env: Env3D) -> None:
        # plannig start and goal
        self.start = Node3D(start, start, 0, 0)
        self.goal = Node3D(goal, goal, 0, 0)
        # environment
        self.env = env
        # graph handler
        self.plot = Plot3D(start, goal, env)

    def dist(self, node1: Node3D, node2: Node3D) -> float:
        return math.sqrt(
            (node2.x - node1.x) ** 2 +
            (node2.y - node1.y) ** 2 +
            (node2.z - node1.z) ** 2
        )

    def angle(self, node1: Node3D, node2: Node3D) -> tuple[float, float]:
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        dz = node2.z - node1.z

        azimuth = math.atan2(dy, dx)
        elevation = math.atan2(dz, math.hypot(dx, dy))

        return azimuth, elevation

    @abstractmethod
    def plan(self):
        '''
        Interface for planning.
        '''
        pass

    @abstractmethod
    def run(self):
        '''
        Interface for running both plannig and animation.
        '''
        pass
