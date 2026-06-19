"""Trajectory of N particles on S^{d-1} over t time steps.

Memory complexity summary
--------------------------
Base (positions + velocities + accelerations):  O(tNd)
+ pairwise geodesic distances:                  O(tNd + tN^2)
+ local tangent bases:                          O(tNd^2)
In-place (current step only):                   O(Nd)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .hypersphere_states import HypersphereState
from .geometry import (
    project_to_sphere,
    project_to_tangent,
    exponential_map,
    pairwise_geodesic_distances,
    tangent_basis,
)


# ---------------------------------------------------------------------------
# Full trajectory: O(tNd) base memory
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """Full trajectory of N particles on S^{d-1} for t time steps.

    positions:           (t, N, d)   unit vectors — always stored
    velocities:          (t, N, d)   tangent vectors — always stored
    accelerations:       (t, N, d)   tangent vectors — always stored
    times:               (t,)        time stamps
    pairwise_distances:  (t, N, N)   optional; add O(tN^2) memory
    tangent_bases:       (t, N, d, d-1)  optional; add O(tNd^2) memory
    """

    positions: np.ndarray               # (t, N, d)
    velocities: np.ndarray              # (t, N, d)
    accelerations: np.ndarray           # (t, N, d)
    times: np.ndarray                   # (t,)
    pairwise_distances: Optional[np.ndarray] = field(default=None)   # (t, N, N)
    tangent_bases: Optional[np.ndarray] = field(default=None)        # (t, N, d, d-1)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        self.velocities = np.asarray(self.velocities, dtype=float)
        self.accelerations = np.asarray(self.accelerations, dtype=float)
        self.times = np.asarray(self.times, dtype=float)

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    @property
    def t(self) -> int:
        return self.positions.shape[0]

    @property
    def N(self) -> int:
        return self.positions.shape[1]

    @property
    def d(self) -> int:
        return self.positions.shape[2]

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    @property
    def memory_bytes(self) -> dict[str, int]:
        """Estimated heap memory in bytes for each stored array (float64)."""
        f64 = 8
        tnd = self.t * self.N * self.d * f64
        result: dict[str, int] = {
            "positions": tnd,
            "velocities": tnd,
            "accelerations": tnd,
            "times": self.t * f64,
        }
        result["total"] = sum(result.values())
        if self.pairwise_distances is not None:
            pw = self.t * self.N * self.N * f64
            result["pairwise_distances"] = pw
            result["total"] += pw
        if self.tangent_bases is not None:
            tb = self.t * self.N * self.d * (self.d - 1) * f64
            result["tangent_bases"] = tb
            result["total"] += tb
        return result

    def memory_summary(self) -> str:
        """Human-readable memory summary."""
        mb = self.memory_bytes
        lines = [f"Trajectory(t={self.t}, N={self.N}, d={self.d})"]
        for k, v in mb.items():
            lines.append(f"  {k:30s}: {v / 2**20:8.2f} MiB")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def state_at(self, step: int) -> HypersphereState:
        """Extract a HypersphereState snapshot at time index `step`."""
        return HypersphereState(
            self.positions[step].copy(),
            self.velocities[step].copy(),
            self.accelerations[step].copy(),
        )

    def __getitem__(self, step: int) -> HypersphereState:
        return self.state_at(step)

    # ------------------------------------------------------------------
    # Optional geometry caches
    # ------------------------------------------------------------------

    def compute_pairwise_distances(self) -> None:
        """Compute and cache (t, N, N) geodesic distances. Adds O(tN^2) memory."""
        self.pairwise_distances = np.stack([
            pairwise_geodesic_distances(self.positions[i]) for i in range(self.t)
        ])

    def compute_tangent_bases(self) -> None:
        """Compute and cache (t, N, d, d-1) local frames. Adds O(tNd^2) memory."""
        bases = np.zeros((self.t, self.N, self.d, self.d - 1))
        for i in range(self.t):
            for n in range(self.N):
                bases[i, n] = tangent_basis(self.positions[i, n])
        self.tangent_bases = bases

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def allocate(cls, t: int, N: int, d: int, dt: float = 1.0) -> Trajectory:
        """Pre-allocate zero-filled arrays. Memory: O(tNd)."""
        return cls(
            positions=np.zeros((t, N, d)),
            velocities=np.zeros((t, N, d)),
            accelerations=np.zeros((t, N, d)),
            times=np.arange(t, dtype=float) * dt,
        )

    @classmethod
    def from_states(
        cls,
        states: list[HypersphereState],
        times: Optional[np.ndarray] = None,
    ) -> Trajectory:
        """Stack a list of HypersphereState snapshots into a Trajectory."""
        t = len(states)
        if times is None:
            times = np.arange(t, dtype=float)
        return cls(
            positions=np.stack([s.positions for s in states]),
            velocities=np.stack([s.velocities for s in states]),
            accelerations=np.stack([s.accelerations for s in states]),
            times=times,
        )

    @classmethod
    def record(cls, integrator: Integrator, t: int, dt: float) -> Trajectory:
        """Run an integrator for t steps and collect the full trajectory."""
        traj = cls.allocate(t, integrator.state.N, integrator.state.d, dt)
        for i in range(t):
            s = integrator.state
            traj.positions[i] = s.positions
            traj.velocities[i] = s.velocities
            traj.accelerations[i] = s.accelerations
            integrator.step(dt)
        return traj


# ---------------------------------------------------------------------------
# In-place integrator: O(Nd) current-step-only memory
# ---------------------------------------------------------------------------

class Integrator:
    """Leapfrog integrator on S^{d-1}. Memory: O(Nd) — no history stored.

    Stores only the current positions, velocities, and accelerations.
    The force_fn callback computes accelerations from positions; it must
    return (N, d) tangent vectors (already projected onto T_pos S^{d-1}).

    If force_fn is None, particles coast along geodesics (free motion).
    """

    def __init__(
        self,
        initial: HypersphereState,
        force_fn=None,
    ) -> None:
        s = initial.enforce_constraints()
        # Store as plain arrays, not a dataclass, to make in-place updates obvious
        self._pos = s.positions.copy()    # (N, d)
        self._vel = s.velocities.copy()   # (N, d)
        self._acc = s.accelerations.copy()  # (N, d)
        self._force_fn = force_fn
        self._time = 0.0

    @property
    def state(self) -> HypersphereState:
        return HypersphereState(self._pos.copy(), self._vel.copy(), self._acc.copy())

    @property
    def time(self) -> float:
        return self._time

    def step(self, dt: float) -> None:
        """One Störmer–Verlet (leapfrog) step on S^{d-1}. O(Nd) per call."""
        # Half-step velocity in tangent space
        v_half = project_to_tangent(self._pos, self._vel + 0.5 * dt * self._acc)

        # Geodesic position update via exponential map
        self._pos = project_to_sphere(exponential_map(self._pos, dt * v_half))

        # Recompute forces at new positions
        if self._force_fn is not None:
            self._acc = project_to_tangent(self._pos, self._force_fn(self._pos))
        else:
            self._acc = np.zeros_like(self._pos)

        # Complete velocity step
        self._vel = project_to_tangent(self._pos, v_half + 0.5 * dt * self._acc)
        self._time += dt

    @classmethod
    def from_random(cls, N: int, d: int, speed: float = 0.1, seed: int | None = None, force_fn=None) -> Integrator:
        return cls(HypersphereState.random(N, d, speed=speed, seed=seed), force_fn=force_fn)