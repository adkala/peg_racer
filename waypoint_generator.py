"""
Waypoint generator for track following.
This module provides waypoint generation and track progress tracking functionality.
"""
from dataclasses import dataclass
from typing import Optional, Literal, Tuple
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import pandas as pd
import yaml
import os


WaypointKind = Literal['circle', 'oval', 'square', 'custom']


@dataclass
class WaypointSpec:
    kind: WaypointKind
    dt: float
    H: int
    speed: float
    path: Optional[jnp.ndarray] = None
    scale: float = 1.0
    waypoint_t: Optional[jnp.ndarray] = None
    waypoint_xy: Optional[jnp.ndarray] = None


# Register as PyTree for JAX
jax.tree_util.register_pytree_node(
    WaypointSpec,
    lambda s: ((s.path, s.waypoint_t, s.waypoint_xy),
               (s.kind, s.dt, s.H, s.speed, s.scale)),
    lambda data, aux: WaypointSpec(aux[0], aux[1], aux[2], aux[3], data[0], aux[4], data[1], data[2])
)


def seg_len(x1, y1, x2, y2):
    """Calculate segment length between two points"""
    return jnp.hypot(x2 - x1, y2 - y1)


def get_curvature(x1, y1, x2, y2, x3, y3):
    """Calculate curvature using Heron's formula"""
    a = seg_len(x1, y1, x2, y2)
    b = seg_len(x2, y2, x3, y3)
    c = seg_len(x1, y1, x3, y3)
    s = 0.5 * (a + b + c)

    area_term = jnp.maximum(s * (s - a) * (s - b) * (s - c), 0.0)
    num = 4.0 * jnp.sqrt(area_term)
    den = jnp.maximum(a * b * c, 1e-12)
    prod = (x2 - x1) * (y3 - y2) - (x3 - x2) * (y2 - y1)
    k = (num / den) * jnp.sign(prod)
    return k


def line_point_t(px, py, x1, y1, x2, y2):
    """Find projection parameter of point onto line segment"""
    dx, dy = x2 - x1, y2 - y1
    denom = dx * dx + dy * dy
    denom = jnp.maximum(denom, 1e-12)
    t = ((px - x1) * dx + (py - y1) * dy) / denom
    return t


def clamp01(x):
    """Clamp value to [0, 1]"""
    return jnp.minimum(1.0, jnp.maximum(0.0, x))


def custom_fn(theta: jnp.ndarray, traj: jnp.ndarray):
    """Interpolate position, velocity, and curvature from custom trajectory"""
    t_end = traj[-1, 0]
    theta_wrapped = jnp.mod(theta, t_end)

    i = jnp.clip(jnp.searchsorted(traj[:, 0], theta_wrapped, side='right') - 1, 0, traj.shape[0] - 2)

    t0 = traj[i, 0]
    t1 = traj[i + 1, 0]
    x0 = traj[i, 1]
    y0 = traj[i, 2]
    v0 = traj[i, 3]
    x1 = traj[i + 1, 1]
    y1 = traj[i + 1, 2]
    v1 = traj[i + 1, 3]

    ratio = (theta_wrapped - t0) / jnp.maximum(t1 - t0, 1e-12)
    x = x0 + ratio * (x1 - x0)
    y = y0 + ratio * (y1 - y0)
    v = v0 + ratio * (v1 - v0)

    N = traj.shape[0]
    i1 = (i + 1) % N
    i2 = (i + 2) % N

    k = get_curvature(traj[i, 1], traj[i, 2], traj[i1, 1], traj[i1, 2], traj[i2, 1], traj[i2, 2])

    return jnp.stack([x, y], axis=0), v, {'curv': k}


def _nearest_waypoint_idx(waypoint_xy: jnp.ndarray, pos2d: jnp.ndarray):
    """Find nearest waypoint index to position"""
    dists = jnp.linalg.norm(waypoint_xy - pos2d[None, :], axis=-1)
    return jnp.argmin(dists)


def _refine_along_segments(waypoint_t: jnp.ndarray, waypoint_xy: jnp.ndarray, i: jnp.ndarray, pos2d: jnp.ndarray):
    """Refine position estimate by projecting onto adjacent segments"""
    N = waypoint_xy.shape[0]
    im1 = (i - 1) % N
    ip1 = (i + 1) % N

    Pi = waypoint_xy[i]
    Pim1 = waypoint_xy[im1]
    Pip1 = waypoint_xy[ip1]

    # Previous segment
    d_prev = seg_len(Pim1[0], Pim1[1], Pi[0], Pi[1])
    t1 = line_point_t(pos2d[0], pos2d[1], Pim1[0], Pim1[1], Pi[0], Pi[1])
    side1 = jnp.sign((Pi[0] - Pim1[0]) * (pos2d[1] - Pim1[1]) - (Pi[1] - Pim1[1]) * (pos2d[0] - Pim1[0]))
    pt1 = Pim1 + t1 * (Pi - Pim1)
    dist1 = jnp.linalg.norm(pos2d - pt1) * side1
    d1 = d_prev * (clamp01(t1) - 1.0)

    # Next segment
    d_next = seg_len(Pi[0], Pi[1], Pip1[0], Pip1[1])
    t2 = line_point_t(pos2d[0], pos2d[1], Pi[0], Pi[1], Pip1[0], Pip1[1])
    side2 = jnp.sign((Pip1[0] - Pi[0]) * (pos2d[1] - Pi[1]) - (Pip1[1] - Pi[1]) * (pos2d[0] - Pi[0]))
    pt2 = Pi + t2 * (Pip1 - Pi)
    dist2 = jnp.linalg.norm(pos2d - pt2) * side2
    d2 = d_next * clamp01(t2)

    final_dist = jnp.where(jnp.abs(dist1) < jnp.abs(dist2), dist1, dist2)

    return waypoint_t[i] + d1 + d2, final_dist


def gen_custom_targets(path, t0, H, dt, mu_factor, body_speed):
    """Generate target waypoints along custom path"""
    idxs = jnp.arange(H + 1)

    def per_i(carry, i):
        t = t0 + i * dt * body_speed
        pos, speed, info = custom_fn(t, path)
        speed = speed * jnp.sqrt(mu_factor)
        pos_next, _, _ = custom_fn(t + dt * speed, path)
        vel = (pos_next - pos) / jnp.maximum(dt, 1e-12)
        speed_ref = jnp.clip(jnp.linalg.norm(vel), 0.5, 100.0)
        psi = jnp.arctan2(vel[1], vel[0])
        tgt = jnp.array([pos[0], pos[1], psi, info['curv'], speed])
        return carry, tgt

    _, out = lax.scan(per_i, None, idxs)
    return out


def init_waypoints(kind: WaypointKind, dt: float, H: int, speed: float,
                   path: Optional[jnp.ndarray] = None, scale: float = 1.0) -> WaypointSpec:
    """Initialize waypoint specification for custom path"""
    if kind == 'custom':
        assert path is not None, "For kind='custom', you must pass path (N,4)."
        t_max = path[-1, 0]
        waypoint_t = jnp.arange(0.0, t_max, 0.1 * scale)

        def f(t):
            pos, _, _ = custom_fn(t, path)
            return pos

        waypoint_xy = jax.vmap(f)(waypoint_t)
        return WaypointSpec(kind, dt, H, speed, path=path, scale=scale,
                           waypoint_t=waypoint_t, waypoint_xy=waypoint_xy)
    else:
        raise ValueError(f"Waypoint kind '{kind}' not implemented. Only 'custom' is supported.")


def generate(spec: WaypointSpec,
            obs: jnp.ndarray,
            dt: float = -1.0,
            mu_factor: float = 1.0,
            body_speed: float = 1.0):
    """
    Generate waypoints for a given observation.

    Args:
        spec: Waypoint specification
        obs: Observation array [x, y, psi, vx, vy, ...]
        dt: Time step (if < 0, use spec.dt)
        mu_factor: Friction factor
        body_speed: Body speed multiplier

    Returns:
        targets: Target waypoints (H+1, 5) [x, y, psi, curv, speed]
        kin_pos: Kinematic lookahead position
        s: Track progress (longitudinal)
        e: Track error (lateral)
    """
    dt_use = jnp.where(dt < 0, spec.dt, dt)
    pos2d = obs[:2]

    i = _nearest_waypoint_idx(spec.waypoint_xy, pos2d)
    t_closed = spec.waypoint_t[i]
    t_refined, final_dist = _refine_along_segments(spec.waypoint_t, spec.waypoint_xy, i, pos2d)

    pos0, speed0, _ = custom_fn(t_refined, spec.path)
    speed0 = speed0 * jnp.sqrt(mu_factor)
    kin_pos, _, _ = custom_fn(t_refined + 1.2 * speed0, spec.path)

    targets = gen_custom_targets(spec.path, t_refined, spec.H, dt_use, mu_factor, body_speed)
    return targets, kin_pos, t_refined, final_dist


def load_path(waypoint_type, ref_trajs_dir=None):
    """
    Load waypoint path from YAML configuration.

    Args:
        waypoint_type: Path to the YAML configuration file
        ref_trajs_dir: Base directory for reference trajectories.
                      If None, uses './ref_trajs' relative to repo root.
    """
    if ref_trajs_dir is None:
        ref_trajs_dir = os.path.join(os.path.dirname(__file__), "ref_trajs")

    yaml_content = yaml.load(open(waypoint_type, "r"), Loader=yaml.FullLoader)
    centerline_file = yaml_content["track_info"]["centerline_file"][:-4]
    ox = yaml_content["track_info"]["ox"]
    oy = yaml_content["track_info"]["oy"]

    csv_path = os.path.join(ref_trajs_dir, centerline_file + "_with_speeds.csv")
    df = pd.read_csv(csv_path)

    if waypoint_type.find("num") != -1:
        return np.array(df.iloc[:-1, :]) * yaml_content["track_info"]["scale"] + np.array([0, ox, oy, 0])
    else:
        return np.array(df.iloc[:, :]) + np.array([0, ox, oy, 0])


class WaypointGenerator:
    """Waypoint generator for track following"""

    def __init__(self,
                 params_yaml_path: Optional[str] = None,
                 ref_trajs_dir: Optional[str] = None,
                 dt: float = 0.1,
                 H: int = 9,
                 speed: float = 1.0,
                 scale: float = 6.5):
        """
        Initialize waypoint generator.

        Args:
            params_yaml_path: Path to params YAML file (defaults to ./data/params-num.yaml)
            ref_trajs_dir: Directory containing reference trajectories (defaults to ./ref_trajs)
            dt: Time step for waypoint generation
            H: Horizon (number of waypoints)
            speed: Speed scale factor
            scale: Track scale factor
        """
        if params_yaml_path is None:
            params_yaml_path = os.path.join(os.path.dirname(__file__), "data", "params-num.yaml")

        path = load_path(params_yaml_path, ref_trajs_dir)
        self.spec = init_waypoints(
            kind="custom", dt=dt, H=H, speed=speed, path=jnp.array(path), scale=scale
        )
        self.track_L = float(path[-1, 0])

    def __call__(self, obs5: jnp.ndarray, vx: float):
        """
        Generate waypoints for given observation.

        Args:
            obs5: Observation [x, y, psi, vx, vy]
            vx: Current velocity

        Returns:
            targets: Target waypoints
            kin_pos: Kinematic lookahead position
            s: Track progress
            e: Track error
        """
        targets, kin_pos, s, e = generate(
            self.spec, obs5, dt=0.1, mu_factor=1.0, body_speed=vx
        )
        return targets, kin_pos, s, e
