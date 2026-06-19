"""Single-timestep state of N particles on S^{d-1}.

Memory complexity: O(Nd) for positions + velocities + accelerations.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .geometry import project_to_sphere, project_to_tangent, to_spherical, from_spherical


@dataclass
class HypersphereState:
    """Ambient-coordinate state of N particles on S^{d-1} at one instant.

    positions:     (N, d)  unit vectors constrained to S^{d-1}
    velocities:    (N, d)  tangent vectors satisfying vel[i] · pos[i] = 0
    accelerations: (N, d)  tangent vectors satisfying acc[i] · pos[i] = 0

    Total memory: O(Nd)  — three arrays of shape (N, d).
    """

    positions: np.ndarray       # (N, d)
    velocities: np.ndarray      # (N, d)
    accelerations: np.ndarray   # (N, d)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        self.velocities = np.asarray(self.velocities, dtype=float)
        self.accelerations = np.asarray(self.accelerations, dtype=float)
        self._check_shapes()

    def _check_shapes(self) -> None:
        N, d = self.positions.shape
        for name, arr in [("velocities", self.velocities), ("accelerations", self.accelerations)]:
            if arr.shape != (N, d):
                raise ValueError(f"{name} shape {arr.shape} must match positions shape ({N}, {d})")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def N(self) -> int:
        return self.positions.shape[0]

    @property
    def d(self) -> int:
        return self.positions.shape[1]

    @property
    def kinetic_energy(self) -> np.ndarray:
        """Per-particle kinetic energy: 0.5 ||v_i||^2. Shape: (N,)."""
        return 0.5 * np.sum(self.velocities ** 2, axis=1)

    # ------------------------------------------------------------------
    # Constraint enforcement
    # ------------------------------------------------------------------

    def enforce_constraints(self) -> HypersphereState:
        """Return a new state with positions on S^{d-1} and tangent velocities/accels."""
        pos = project_to_sphere(self.positions)
        vel = project_to_tangent(pos, self.velocities)
        acc = project_to_tangent(pos, self.accelerations)
        return HypersphereState(pos, vel, acc)

    def constraint_violations(self) -> dict[str, float]:
        """Maximum constraint violation norms — useful for integration diagnostics."""
        pos_norm_err = np.max(np.abs(np.linalg.norm(self.positions, axis=1) - 1.0))
        vel_tangency = np.max(np.abs(np.sum(self.velocities * self.positions, axis=1)))
        acc_tangency = np.max(np.abs(np.sum(self.accelerations * self.positions, axis=1)))
        return {
            "position_unit_norm": float(pos_norm_err),
            "velocity_tangency": float(vel_tangency),
            "acceleration_tangency": float(acc_tangency),
        }

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def random(cls, N: int, d: int, speed: float = 0.1, seed: int | None = None) -> HypersphereState:
        """N particles uniformly on S^{d-1} with random tangent velocities."""
        rng = np.random.default_rng(seed)
        pos = project_to_sphere(rng.standard_normal((N, d)))
        vel = project_to_tangent(pos, rng.standard_normal((N, d)) * speed)
        acc = np.zeros((N, d))
        return cls(pos, vel, acc)

    @classmethod
    def north_pole(cls, N: int, d: int) -> HypersphereState:
        """N particles stacked at the north pole (0, ..., 0, 1) with zero dynamics."""
        pos = np.zeros((N, d))
        pos[:, -1] = 1.0
        return cls(pos, np.zeros((N, d)), np.zeros((N, d)))

    @classmethod
    def from_intrinsic(cls, angles: np.ndarray, angle_velocities: np.ndarray) -> HypersphereState:
        """Construct from intrinsic spherical coordinates.

        angles:           (N, d-1) spherical angles — memory O(N(d-1))
        angle_velocities: (N, d-1) rates of change of each angle

        Converts to ambient coordinates (N, d) with zero accelerations.
        """
        pos = from_spherical(angles)
        vel = _angular_velocity_to_tangent(pos, angles, angle_velocities)
        return cls(pos, vel, np.zeros_like(pos))

    # ------------------------------------------------------------------
    # Intrinsic coordinate view
    # ------------------------------------------------------------------

    def to_intrinsic(self) -> np.ndarray:
        """Return intrinsic spherical coordinates (N, d-1). Memory O(N(d-1))."""
        return to_spherical(self.positions)


# ---------------------------------------------------------------------------
# Intrinsic-only representation — smaller constant, same O(N(d-1)) asymptotics
# ---------------------------------------------------------------------------

@dataclass
class IntrinsicHypersphereState:
    """State of N particles stored in intrinsic spherical coordinates.

    angles:     (N, d-1)  spherical angles — memory O(N(d-1))
    ang_vel:    (N, d-1)  angle velocities
    ang_acc:    (N, d-1)  angle accelerations

    Slightly smaller constant than ambient representation, same O asymptotics.
    Useful when d is large and the factor of (d-1)/d matters.
    """

    angles: np.ndarray      # (N, d-1)
    ang_vel: np.ndarray     # (N, d-1)
    ang_acc: np.ndarray     # (N, d-1)

    def __post_init__(self) -> None:
        self.angles = np.asarray(self.angles, dtype=float)
        self.ang_vel = np.asarray(self.ang_vel, dtype=float)
        self.ang_acc = np.asarray(self.ang_acc, dtype=float)

    @property
    def N(self) -> int:
        return self.angles.shape[0]

    @property
    def d(self) -> int:
        return self.angles.shape[1] + 1

    def to_ambient(self) -> HypersphereState:
        """Convert to ambient HypersphereState (N, d)."""
        pos = from_spherical(self.angles)
        vel = _angular_velocity_to_tangent(pos, self.angles, self.ang_vel)
        acc = _angular_velocity_to_tangent(pos, self.angles, self.ang_acc)
        return HypersphereState(pos, vel, acc)

    @classmethod
    def random(cls, N: int, d: int, speed: float = 0.1, seed: int | None = None) -> IntrinsicHypersphereState:
        rng = np.random.default_rng(seed)
        angles = np.zeros((N, d - 1))
        angles[:, :-1] = rng.uniform(0, np.pi, (N, d - 2))
        angles[:, -1] = rng.uniform(0, 2 * np.pi, N)
        ang_vel = rng.standard_normal((N, d - 1)) * speed
        return cls(angles, ang_vel, np.zeros((N, d - 1)))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _angular_velocity_to_tangent(
    pos: np.ndarray, angles: np.ndarray, ang_vel: np.ndarray
) -> np.ndarray:
    """Approximate Jacobian mapping: angle velocities → ambient tangent vectors.

    Uses finite differences on from_spherical for generality.
    pos:     (N, d)
    angles:  (N, d-1)
    ang_vel: (N, d-1)
    Returns: (N, d) tangent vectors.
    """
    eps = 1e-6
    N, d = pos.shape
    tangent = np.zeros((N, d))
    for k in range(d - 1):
        a_fwd = angles.copy()
        a_fwd[:, k] += eps
        dp = (from_spherical(a_fwd) - pos) / eps  # (N, d) Jacobian column
        tangent += ang_vel[:, k : k + 1] * dp
    return project_to_tangent(pos, tangent)
