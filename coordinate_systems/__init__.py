"""coordinate_systems — N particles on S^{d-1}.

Memory complexity reference
----------------------------
HypersphereState (single step):          O(Nd)
IntrinsicHypersphereState (single step): O(N(d-1))   smaller constant
Trajectory (full history):               O(tNd)
Trajectory + pairwise distances:         O(tNd + tN^2)
Trajectory + tangent bases:              O(tNd^2)
Integrator (in-place, no history):       O(Nd)
"""

from .geometry import (
    project_to_sphere,
    project_to_tangent,
    exponential_map,
    logarithmic_map,
    parallel_transport,
    tangent_basis,
    pairwise_dot_products,
    pairwise_geodesic_distances,
    to_spherical,
    from_spherical,
)
from .hypersphere_states import (
    HypersphereState,
    IntrinsicHypersphereState,
)
from .trajectory import (
    Trajectory,
    Integrator,
)

__all__ = [
    "project_to_sphere",
    "project_to_tangent",
    "exponential_map",
    "logarithmic_map",
    "parallel_transport",
    "tangent_basis",
    "pairwise_dot_products",
    "pairwise_geodesic_distances",
    "to_spherical",
    "from_spherical",
    "HypersphereState",
    "IntrinsicHypersphereState",
    "Trajectory",
    "Integrator",
]
