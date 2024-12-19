from math import cos, pi, sqrt, atan
from typing import Union, Optional

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

ndarray = Union[np.ndarray, cp.ndarray]

def compute_linear_transform(pos1, pos2, offset=None):
    # Compute vector pointing from pos1 to pos2
    beam_vec = pos2 - pos1

    magnitude = np.linalg.norm(beam_vec)

    #  matlab behaviour is to return nans when positions are the same.
    #  we choose to return the identity matrix and the offset in this case.
    #  TODO: we should open an issue and change our behaviour once matlab is fixed.

    # if np.isclose(magnitude, 0):

    #     # "pos1 and pos2 are the same"
    #     if (shape1 := np.shape(pos1)) == (shape2 := np.shape(pos2)):
    #         raise ValueError(f"pos1 and pos2 must have the same shape. Received shapes: {shape1} and {shape2}")
    #     return np.eye(3), np.zeros_like(pos1) if offset is None else offset * np.ones_like(pos1)

    # Normalise to give unit beam vector
    beam_vec = beam_vec / magnitude

    # Canonical normalised beam_vec (canonical pos1 is [0, 0, 1])
    beam_vec0 = np.array([0, 0, -1])

    # Find the rotation matrix for the bowl
    u = np.cross(beam_vec0, beam_vec)

    # Normalise the rotation matrix if not zero
    if any(u != 0):
        u = u / np.linalg.norm(u)

    # Find the axis-angle transformation between beam_vec and e1
    theta = np.arccos(np.dot(beam_vec0, beam_vec))

    # Convert axis-angle transformation to a rotation matrix
    A = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    rotMat = np.cos(theta) * np.eye(3) + np.sin(theta) * A + (1 - np.cos(theta)) * np.outer(u, u)

    # Compute an offset for the bowl, where bowl_centre = move from pos1
    # towards focus by radius
    if offset is not None:
        offsetPos = pos1 + offset * beam_vec
    else:
        offsetPos = 0

    return rotMat, offsetPos

def make_cart_bowl(
    bowl_pos: np.ndarray, radius: float, diameter: float, focus_pos: np.ndarray, num_points: int) -> np.ndarray:
    """
    Create evenly distributed Cartesian points covering a bowl.

    Args:
        bowl_pos:       Cartesian position of the centre of the rear surface of
                        the bowl given as a three element vector [bx, by, bz] [m].
        radius:         Radius of curvature of the bowl [m].
        diameter:       Diameter of the opening of the bowl [m].
        focus_pos:      Any point on the beam axis of the bowl given as a three
                        element vector [fx, fy, fz] [m].
        num_points:     Number of points on the bowl.
        plot_bowl:      Boolean controlling whether the Cartesian points are
                        plotted.

    Returns:
        3 x num_points array of Cartesian coordinates.

    Examples:
        bowl = makeCartBowl([0, 0, 0], 1, 2, [0, 0, 1], 100)
        bowl = makeCartBowl([0, 0, 0], 1, 2, [0, 0, 1], 100, True)
    """
    GOLDEN_ANGLE = pi * (3 - sqrt(5.))  # golden angle in radians

    # check input values
    if radius <= 0:
        raise ValueError("The radius must be positive.")
    if diameter <= 0:
        raise ValueError("The diameter must be positive.")
    if diameter > 2 * radius:
        raise ValueError("The diameter of the bowl must be equal or less than twice the radius of curvature.")
    if np.all(bowl_pos == focus_pos):
        raise ValueError("The focus_pos must be different from the bowl_pos.")

    # check for infinite radius of curvature, and call makeCartDisc instead
    if np.isinf(radius):
        # bowl = make_cart_disc(bowl_pos, diameter / 2, focus_pos, num_points, plot_bowl)
        # return bowl
        raise NotImplementedError("make_cart_disc")

    # compute arc angle from chord (ref: https://en.wikipedia.org/wiki/Chord_(geometry))
    varphi_max = np.arcsin(diameter / (2 * radius))

    # compute spiral parameters
    theta = lambda t: GOLDEN_ANGLE * t
    C = 2 * np.pi * (1 - np.cos(varphi_max)) / (num_points - 1)
    varphi = lambda t: np.arccos(1 - C * t / (2 * np.pi))

    # compute canonical spiral points
    t = np.linspace(0, num_points - 1, num_points)
    p0 = np.array([np.cos(theta(t)) * np.sin(varphi(t)), np.sin(theta(t)) * np.sin(varphi(t)), np.cos(varphi(t))])
    p0 = radius * p0

    # linearly transform the canonical spiral points to give bowl in correct orientation
    R, b = compute_linear_transform(bowl_pos, focus_pos, radius)

    if b.ndim == 1:
        b = np.expand_dims(b, axis=-1)  # expand dims for broadcasting

    bowl = R @ p0 + b

    return bowl