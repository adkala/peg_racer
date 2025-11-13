"""
Single-agent racing environment for training a car to complete a track.
Uses MPC-inspired reward function for training with perl_jax.
"""
from dataclasses import dataclass
from typing import NamedTuple, Tuple
import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from jax_waypoint import init_waypoints, generate
from mpc_reward import (
    compute_mpc_reward,
    RewardState,
    MPCRewardConfig,
    create_default_reward_config,
)

GRAVITY = 9.81


class CarState(NamedTuple):
    x: jnp.ndarray
    y: jnp.ndarray
    psi: jnp.ndarray
    vx: jnp.ndarray
    vy: jnp.ndarray
    omega: jnp.ndarray


class EnvState(NamedTuple):
    car: CarState
    delay_buf: jnp.ndarray
    t: jnp.int32  # type: ignore
    last_s: jnp.ndarray  # last progress along track
    track_L: float
    reward_state: RewardState  # MPC reward state


@dataclass
class DynamicParams:
    num_envs: int
    LF: float = 0.11
    LR: float = 0.23
    MASS: float = 4.65
    DT: float = 0.05
    K_RFY: float = 20.0
    K_FFY: float = 20.0
    Iz: float = 0.1
    Ta: float = 5.5
    Tb: float = -1.0
    Sa: float = 0.36
    Sb: float = 0.03
    mu: float = 3.0
    Cf: float = 1.0
    Cr: float = 1.0
    Bf: float = 60.0
    Br: float = 60.0
    hcom: float = 0.0
    fr: float = 0.1
    delay: int = 4


def dbm_dxdt(
    x, y, psi, vx, vy, omega,
    target_vel, target_steer,
    Ta, Tb, Sa, Sb, LF, LR, MASS,
    K_RFY, K_FFY, Iz, mu, Cf, Cr,
    Bf, Br, hcom, fr,
):
    steer = target_steer * Sa + Sb
    prev_vel = jnp.hypot(vx, vy)
    throttle = target_vel * Ta - target_vel * Tb * prev_vel

    next_x = vx * jnp.cos(psi) - vy * jnp.sin(psi)
    next_y = vx * jnp.sin(psi) + vy * jnp.cos(psi)
    next_psi = omega

    alpha_f = steer - jnp.arctan((LF * omega + vy) / jnp.maximum(vx, 0.5))
    alpha_r = jnp.arctan((LR * omega - vy) / jnp.maximum(vx, 0.5))

    F_rx = throttle - fr * MASS * GRAVITY * jnp.sign(vx)

    F_fz = 0.5 * MASS * GRAVITY * LR / (LF + LR) - 0.5 * hcom / (LF + LR) * F_rx
    F_rz = 0.5 * MASS * GRAVITY * LF / (LF + LR) + 0.5 * hcom / (LF + LR) * F_rx

    F_fy = 2 * mu * F_fz * jnp.sin(Cf * jnp.arctan(Bf * alpha_f))
    F_ry = 2 * mu * F_rz * jnp.sin(Cr * jnp.arctan(Br * alpha_r))

    ax = (F_rx - F_fy * jnp.sin(steer) + vy * omega * MASS) / MASS
    ay = (F_ry + F_fy * jnp.cos(steer) - vx * omega * MASS) / MASS
    adot = (F_fy * LF * jnp.cos(steer) - F_ry * LR) / Iz
    return next_x, next_y, next_psi, ax, ay, adot


def rk4_step(params: DynamicParams, state, target_vel, target_steer):
    """RK4 integration for a single car."""
    DT = params.DT
    K1 = dbm_dxdt(
        *state, target_vel, target_steer,
        params.Ta, params.Tb, params.Sa, params.Sb,
        params.LF, params.LR, params.MASS,
        params.K_RFY, params.K_FFY, params.Iz,
        params.mu, params.Cf, params.Cr,
        params.Bf, params.Br, params.hcom, params.fr
    )

    S2 = tuple(state[i] + 0.5 * DT * K1[i] for i in range(6))
    K2 = dbm_dxdt(
        *S2, target_vel, target_steer,
        params.Ta, params.Tb, params.Sa, params.Sb,
        params.LF, params.LR, params.MASS,
        params.K_RFY, params.K_FFY, params.Iz,
        params.mu, params.Cf, params.Cr,
        params.Bf, params.Br, params.hcom, params.fr
    )

    S3 = tuple(state[i] + 0.5 * DT * K2[i] for i in range(6))
    K3 = dbm_dxdt(
        *S3, target_vel, target_steer,
        params.Ta, params.Tb, params.Sa, params.Sb,
        params.LF, params.LR, params.MASS,
        params.K_RFY, params.K_FFY, params.Iz,
        params.mu, params.Cf, params.Cr,
        params.Bf, params.Br, params.hcom, params.fr
    )

    S4 = tuple(state[i] + DT * K3[i] for i in range(6))
    K4 = dbm_dxdt(
        *S4, target_vel, target_steer,
        params.Ta, params.Tb, params.Sa, params.Sb,
        params.LF, params.LR, params.MASS,
        params.K_RFY, params.K_FFY, params.Iz,
        params.mu, params.Cf, params.Cr,
        params.Bf, params.Br, params.hcom, params.fr
    )

    return tuple(
        state[i] + DT / 6.0 * (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i])
        for i in range(6)
    )


def wrap_diff(a, b, L):
    """Compute wrapped difference for cyclic track."""
    d = a - b
    d = jnp.where(d < -L / 2.0, d + L, d)
    d = jnp.where(d > L / 2.0, d - L, d)
    return d


def build_env_functions(
    params: DynamicParams,
    EP_LEN: int,
    track_L: float,
    delay: int,
    wp_generate,
    reward_config: MPCRewardConfig
):
    """Build reset and step functions for single-agent environment with MPC reward."""

    def jax_reset(key: jax.Array) -> Tuple[EnvState, jnp.ndarray]:
        # Spawn at a fixed starting position
        car = CarState(
            x=jnp.array(0.0),
            y=jnp.array(0.0),
            psi=jnp.array(-jnp.pi / 2 - 0.5),
            vx=jnp.array(0.0),
            vy=jnp.array(0.0),
            omega=jnp.array(0.0),
        )
        delay_buf = jnp.zeros((delay, 2), dtype=jnp.float32)

        # Initialize reward state with zero action
        reward_state = RewardState(last_action=jnp.zeros(2, dtype=jnp.float32))

        # Get initial waypoint features
        obs5 = jnp.array([car.x, car.y, car.psi, car.vx, car.vy])
        tgt, _, s, e = wp_generate(obs5, car.vx)
        theta = tgt[0, 2]
        theta_diff = jnp.arctan2(
            jnp.sin(theta - car.psi), jnp.cos(theta - car.psi)
        )
        curv = tgt[0, 3]
        curv_lh = tgt[-1, 3]

        # Observation: [s, e, theta_diff, vx, vy, omega, curv, curv_lh]
        obs = jnp.array(
            [s, e, theta_diff, car.vx, car.vy, car.omega, curv, curv_lh],
            dtype=jnp.float32,
        )

        state = EnvState(
            car=car,
            delay_buf=delay_buf,
            t=jnp.array(0, jnp.int32),
            last_s=s,
            track_L=jnp.asarray(track_L, jnp.float32),
            reward_state=reward_state,
        )

        return state, obs

    def jax_step(state: EnvState, action: jnp.ndarray):
        # Clip and process action
        a = jnp.clip(action, -1.0, 1.0)
        a0 = jnp.array([jnp.clip(a[0], 0.0, 1.0), jnp.clip(a[1], -1.0, 1.0)])

        # Update delay buffer
        buf1 = jnp.concatenate([a0[None, :], state.delay_buf[:-1, :]], axis=0)
        cmd = buf1[-1, :]
        target_vel = cmd[0]
        target_steer = cmd[1]

        # Step dynamics
        S = state.car
        next_tuple = rk4_step(
            params, (S.x, S.y, S.psi, S.vx, S.vy, S.omega), target_vel, target_steer
        )

        car2 = CarState(
            x=next_tuple[0],
            y=next_tuple[1],
            psi=next_tuple[2],
            vx=next_tuple[3],
            vy=next_tuple[4],
            omega=next_tuple[5],
        )

        # Get waypoint features
        obs5 = jnp.array([car2.x, car2.y, car2.psi, car2.vx, car2.vy])
        tgt, _, s, e = wp_generate(obs5, car2.vx)
        theta = tgt[0, 2]
        theta_diff = jnp.arctan2(
            jnp.sin(theta - car2.psi), jnp.cos(theta - car2.psi)
        )
        curv = tgt[0, 3]
        curv_lh = tgt[-1, 3]

        # Current observation (before step)
        current_obs = jnp.array(
            [state.last_s, 0.0, 0.0, S.vx, S.vy, S.omega, 0.0, 0.0],
            dtype=jnp.float32,
        )

        # Next observation (after step)
        next_obs = jnp.array(
            [s, e, theta_diff, car2.vx, car2.vy, car2.omega, curv, curv_lh],
            dtype=jnp.float32,
        )

        # Compute MPC-inspired reward
        reward, new_reward_state = compute_mpc_reward(
            current_obs,
            a0,
            next_obs,
            state.reward_state,
            reward_config,
            track_L
        )

        t2 = state.t + jnp.int32(1)
        done = t2 >= jnp.int32(EP_LEN)
        truncated = done

        state2 = EnvState(
            car=car2,
            delay_buf=buf1,
            t=t2,
            last_s=s,
            track_L=track_L,
            reward_state=new_reward_state,
        )

        return state2, next_obs, reward, done, truncated

    return jax_reset, jax_step


def load_path(waypoint_type, ref_trajs_dir=None):
    """Load waypoint path from YAML configuration."""
    if ref_trajs_dir is None:
        ref_trajs_dir = os.path.join(os.path.dirname(__file__), "data", "ref_trajs")

    yaml_content = yaml.load(open(waypoint_type, "r"), Loader=yaml.FullLoader)
    centerline_file = yaml_content["track_info"]["centerline_file"][:-4]
    ox = yaml_content["track_info"]["ox"]
    oy = yaml_content["track_info"]["oy"]

    csv_path = os.path.join(ref_trajs_dir, centerline_file + "_with_speeds.csv")
    df = pd.read_csv(csv_path)

    if waypoint_type.find("num") != -1:
        return np.array(df.iloc[:-1, :]) * yaml_content["track_info"][
            "scale"
        ] + np.array([0, ox, oy, 0])
    else:
        return np.array(df.iloc[:, :]) + np.array([0, ox, oy, 0])


EP_LEN = 500


def build_single_agent_env(
    num_envs,
    params_yaml_path=None,
    ref_trajs_dir=None,
    reward_config: MPCRewardConfig = None
):
    """
    Build JAX-compiled reset and step functions for single-agent racing.

    Args:
        num_envs: Number of parallel environments
        params_yaml_path: Path to the params YAML file
        ref_trajs_dir: Base directory for reference trajectories
        reward_config: MPC reward configuration (uses default if None)

    Returns:
        reset_jit: JIT-compiled reset function
        step_jit: JIT-compiled step function
        obs_dim: Observation dimension
        act_dim: Action dimension
    """
    params = DynamicParams(
        num_envs=num_envs, DT=0.1, Sa=0.34, Sb=0.0, Ta=20.0, Tb=0.0, mu=0.5, delay=4
    )

    if params_yaml_path is None:
        params_yaml_path = os.path.join(
            os.path.dirname(__file__), "data", "params-num.yaml"
        )

    if reward_config is None:
        reward_config = create_default_reward_config()

    path = load_path(params_yaml_path, ref_trajs_dir)
    spec = init_waypoints(
        kind="custom", dt=0.1, H=9, speed=1.0, path=jnp.array(path), scale=6.5
    )
    track_L = float(path[-1, 0])

    def wp_generate(obs5, vx):
        targets, kin_pos, s, e = generate(
            spec, obs5, dt=0.1, mu_factor=1.0, body_speed=vx
        )

        if targets.shape[1] == 4:
            zeros = jnp.zeros((targets.shape[0], 1))
            targets = jnp.concatenate([targets[:, :3], zeros, targets[:, 3:4]], axis=1)
        return targets, kin_pos, s, e

    reset_fn, step_fn = build_env_functions(
        params, EP_LEN, float(track_L), params.delay, wp_generate, reward_config
    )

    reset_jit = jax.jit(reset_fn)
    step_jit = jax.jit(step_fn, donate_argnums=(0,))

    # Observation: [s, e, theta_diff, vx, vy, omega, curv, curv_lh] = 8 dims
    # Action: [throttle, steering] = 2 dims
    return reset_jit, step_jit, 8, 2
