"""Utility functions for grids."""
from typing import Union

import numpy as np
from sc2.position import Point2


def modify_two_by_two(
    grid: np.ndarray, location: Point2, weight: Union[np.inf, int]
) -> None:
    """Set a 2x2 building as unpathable for the given grid."""
    for i in [-1, 0]:
        for j in [-1, 0]:
            grid[(location[0] + i, location[1] + j)] = weight


def modify_three_by_three(
    grid: np.ndarray, location: Point2, weight: Union[np.inf, int]
) -> None:
    """Set a 3x3 building as unpathable for the given grid."""
    for i in [-1.5, -0.5, 0.5]:
        for j in [-1.5, -0.5, 0.5]:
            grid[int(location[0] + i), int(location[1] + j)] = weight
