"""Geometric operations on the (d-1)-sphere S^{d-1} embedded in R^d."""

from __future__ import annotations
import numpy as np


def project_to_sphere(x: np.ndarray) -> np.ndarray:
    """Normalize points onto the unit sphere.

    x: (..., d)
    Returns: (..., d) with unit norm along last axis.
    """
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def project_to_tangent(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project v onto the tangent space T_x S^{d-1} = {w : w · x = 0}.

    x: (..., d) unit vectors (base points)
    v: (..., d) arbitrary vectors to project
    Returns: (..., d) tangent vectors satisfying result · x = 0.
    """
    dots = np.sum(v * x, axis=-1, keepdims=True)
    return v - dots * x


def exponential_map(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Geodesic step from x in direction v: exp_x(v) on S^{d-1}.

    exp_x(v) = cos(||v||) x + sin(||v||) (v / ||v||)

    x: (..., d) unit vectors
    v: (..., d) tangent vectors at x
    Returns: (..., d) unit vectors.
    """
    norm_v = np.linalg.norm(v, axis=-1, keepdims=True)
    safe_norm = np.where(norm_v == 0, 1.0, norm_v)
    direction = v / safe_norm
    return np.cos(norm_v) * x + np.sin(norm_v) * direction


def logarithmic_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Inverse of exp_x: tangent vector at x pointing toward y.

    log_x(y) = (theta / sin(theta)) * (y - cos(theta) x)
    where theta = arccos(x · y).

    x, y: (..., d) unit vectors
    Returns: (..., d) tangent vectors at x.
    """
    dots = np.clip(np.sum(x * y, axis=-1, keepdims=True), -1.0, 1.0)
    theta = np.arccos(dots)
    sin_theta = np.sin(theta)
    # First-order approximation near theta=0 to avoid 0/0
    coeff = np.where(np.abs(sin_theta) < 1e-10, 1.0, theta / sin_theta)
    return coeff * (y - dots * x)


def parallel_transport(x: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Parallel-transport tangent vector v at x to the tangent space at y.

    Uses the Schild's ladder / closed-form formula along the geodesic from x to y.

    x, y: (..., d) unit vectors
    v:    (..., d) tangent vector at x
    Returns: (..., d) tangent vector at y.
    """
    log_xy = logarithmic_map(x, y)
    norm_log_xy = np.linalg.norm(log_xy, axis=-1, keepdims=True)
    safe_xy = np.where(norm_log_xy == 0, 1.0, norm_log_xy)
    u_x = log_xy / safe_xy  # unit geodesic direction at x

    log_yx = logarithmic_map(y, x)
    norm_log_yx = np.linalg.norm(log_yx, axis=-1, keepdims=True)
    safe_yx = np.where(norm_log_yx == 0, 1.0, norm_log_yx)
    u_y = log_yx / safe_yx  # unit geodesic direction at y (pointing back)

    v_dot_u = np.sum(v * u_x, axis=-1, keepdims=True)
    return v - v_dot_u * (u_x + u_y)


def tangent_basis(x: np.ndarray) -> np.ndarray:
    """Orthonormal basis for T_x S^{d-1} at a single point x.

    x: (d,) unit vector
    Returns: (d, d-1) matrix — columns span the tangent space.
    Memory per particle: O(d^2); across all particles and steps: O(tNd^2).
    """
    d = x.shape[0]
    # SVD of the tangent projector (I - x x^T) to extract orthonormal columns
    projector = np.eye(d) - np.outer(x, x)
    _, _, Vt = np.linalg.svd(projector, full_matrices=True)
    # Last singular value is ~0 (along x); drop it
    return Vt[:-1].T  # (d, d-1)


def pairwise_dot_products(positions: np.ndarray) -> np.ndarray:
    """Gram matrix of dot products between N position vectors.

    positions: (N, d) unit vectors
    Returns:   (N, N) — memory O(N^2).
    """
    return positions @ positions.T


def pairwise_geodesic_distances(positions: np.ndarray) -> np.ndarray:
    """Geodesic (arc-length) distance matrix for N points on S^{d-1}.

    positions: (N, d) unit vectors
    Returns:   (N, N) in radians — memory O(N^2).
    """
    dots = np.clip(positions @ positions.T, -1.0, 1.0)
    return np.arccos(dots)


def to_spherical(x: np.ndarray) -> np.ndarray:
    """Convert ambient R^d coordinates to d-1 spherical angles.

    x: (..., d) unit vectors
    Returns: (..., d-1) angles phi_1, ..., phi_{d-1} in [0, pi] x ... x [0, 2pi].
    Intrinsic memory: O(N(d-1)) vs O(Nd) — same asymptotic, smaller constant.
    """
    d = x.shape[-1]
    angles = []
    for k in range(d - 1):
        denom = np.linalg.norm(x[..., k:], axis=-1)
        denom = np.where(denom == 0, 1.0, denom)
        cos_phi = x[..., k] / denom
        phi = np.arccos(np.clip(cos_phi, -1.0, 1.0))
        # Last angle uses atan2 to resolve the full [0, 2pi] range
        if k == d - 2:
            phi = np.where(x[..., -1] < 0, 2 * np.pi - phi, phi)
        angles.append(phi)
    return np.stack(angles, axis=-1)


def from_spherical(angles: np.ndarray) -> np.ndarray:
    """Convert d-1 spherical angles back to ambient unit vectors in R^d.

    angles: (..., d-1)
    Returns: (..., d) unit vectors.
    """
    d = angles.shape[-1] + 1
    shape = angles.shape[:-1]
    x = np.ones((*shape, d))
    for k in range(d - 1):
        x[..., k] = np.cos(angles[..., k]) * np.prod(np.sin(angles[..., :k]), axis=-1) if k > 0 else np.cos(angles[..., 0])
        x[..., k + 1:] *= np.sin(angles[..., k : k + 1])
    # Recompute cleanly with the standard recursive formula
    x = np.ones((*shape, d))
    for i in range(d):
        x[..., i] = np.prod(np.sin(angles[..., :i]), axis=-1) if i > 0 else 1.0
        if i < d - 1:
            x[..., i] *= np.cos(angles[..., i])
    return x
