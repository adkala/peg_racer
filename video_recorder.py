"""
Video recording utilities for racing environment.
Supports both single-agent and multi-agent environments.
"""
import pygame
import numpy as np
import pandas as pd
import yaml
import os
from pathlib import Path
import jax.numpy as jnp


class SingleAgentVisualizer:
    """Lightweight visualizer for single-agent racing (headless mode supported)."""

    def __init__(self, width=800, height=800, headless=False):
        """
        Initialize the visualizer.

        Args:
            width: Window width in pixels
            height: Window height in pixels
            headless: If True, don't create display window (for video recording)
        """
        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Single Agent Racing")
        else:
            pygame.init()
            self.screen = pygame.Surface((width, height))

        self.width = width
        self.height = height
        self.headless = headless

        # Colors
        self.BG_COLOR = (20, 20, 30)
        self.TRACK_COLOR = (100, 100, 100)
        self.TRACK_BORDER_COLOR = (200, 200, 50)
        self.WAYPOINT_COLOR = (50, 150, 255)
        self.CAR_COLOR = (255, 50, 50)  # Red
        self.TEXT_COLOR = (255, 255, 255)

        # Track data
        self.track_waypoints = None
        self.track_width = None
        self.track_left_boundary = None
        self.track_right_boundary = None
        self.track_scale = 1.0
        self.track_offset = np.array([0.0, 0.0])

        # Font
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)

        # Load track data
        self.load_track_data()

    def load_track_data(self):
        """Load track waypoints from data files."""
        try:
            with open('data/params-num.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)

            centerline_file = params['track_info']['centerline_file'][:-4]
            scale = params['track_info']['scale']
            ox = params['track_info']['ox']
            oy = params['track_info']['oy']
            self.track_width = params['track_info']['track_width'] * scale

            csv_path = f'data/ref_trajs/{centerline_file}_with_speeds.csv'
            df = pd.read_csv(csv_path, comment='#')

            waypoints = np.array(df.iloc[:, 1:3]) * scale + np.array([ox, oy])
            self.track_waypoints = waypoints

            # Compute track boundaries
            self._compute_track_boundaries()

            # Compute viewport
            self._compute_viewport()

        except Exception as e:
            print(f"Warning: Could not load track data: {e}")
            angles = np.linspace(0, 2*np.pi, 100)
            radius = 50
            self.track_waypoints = np.column_stack([
                radius * np.cos(angles),
                radius * np.sin(angles)
            ])
            self.track_width = 10.0
            self._compute_track_boundaries()
            self._compute_viewport()

    def _compute_track_boundaries(self):
        """Compute left and right track boundaries from centerline."""
        if self.track_waypoints is None or len(self.track_waypoints) < 2:
            return

        # Compute tangent vectors using adjacent waypoints
        # For closed track, use wrap-around
        n_points = len(self.track_waypoints)
        prev_points = np.roll(self.track_waypoints, 1, axis=0)
        next_points = np.roll(self.track_waypoints, -1, axis=0)

        # Tangent direction (from previous to next point)
        tangents = next_points - prev_points
        tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangent_norms = np.maximum(tangent_norms, 1e-6)  # Avoid division by zero
        tangents_normalized = tangents / tangent_norms

        # Normal vectors (perpendicular to tangent, rotated 90 degrees)
        # Rotate by 90 degrees: (x, y) -> (-y, x) for left, (y, -x) for right
        normals_left = np.column_stack([-tangents_normalized[:, 1],
                                         tangents_normalized[:, 0]])
        normals_right = -normals_left

        # Compute boundaries
        half_width = self.track_width / 2.0
        self.track_left_boundary = self.track_waypoints + normals_left * half_width
        self.track_right_boundary = self.track_waypoints + normals_right * half_width

    def _compute_viewport(self):
        """Compute scale and offset to fit track in window."""
        if self.track_waypoints is None:
            return

        min_x, min_y = self.track_waypoints.min(axis=0)
        max_x, max_y = self.track_waypoints.max(axis=0)

        track_width = max_x - min_x
        track_height = max_y - min_y

        margin = 0.1
        track_width *= (1 + 2 * margin)
        track_height *= (1 + 2 * margin)

        scale_x = (self.width * 0.9) / track_width
        scale_y = (self.height * 0.9) / track_height
        self.track_scale = min(scale_x, scale_y)

        track_center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
        window_center = np.array([self.width / 2, self.height / 2])
        self.track_offset = window_center - track_center * self.track_scale

    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        screen_x = x * self.track_scale + self.track_offset[0]
        screen_y = -y * self.track_scale + self.track_offset[1]
        return int(screen_x), int(screen_y)

    def draw_track(self):
        """Draw the track using waypoints and boundaries."""
        if self.track_waypoints is None:
            return

        # Draw track boundaries
        if self.track_left_boundary is not None:
            left_points = [self.world_to_screen(x, y) for x, y in self.track_left_boundary]
            if len(left_points) > 1:
                pygame.draw.lines(self.screen, self.TRACK_BORDER_COLOR, True, left_points, 3)

        if self.track_right_boundary is not None:
            right_points = [self.world_to_screen(x, y) for x, y in self.track_right_boundary]
            if len(right_points) > 1:
                pygame.draw.lines(self.screen, self.TRACK_BORDER_COLOR, True, right_points, 3)

        # Draw track centerline (thinner, dimmer)
        points = [self.world_to_screen(x, y) for x, y in self.track_waypoints]
        if len(points) > 1:
            pygame.draw.lines(self.screen, (100, 100, 100), True, points, 1)

        # Draw waypoint dots (optional, smaller now)
        for x, y in self.track_waypoints[::50]:
            screen_pos = self.world_to_screen(x, y)
            pygame.draw.circle(self.screen, self.WAYPOINT_COLOR, screen_pos, 1)

    def draw_car(self, x, y, psi):
        """Draw the car."""
        screen_x, screen_y = self.world_to_screen(x, y)

        car_length = 25
        car_width = 15

        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        corners_local = np.array([
            [car_length/2, car_width/2],
            [-car_length/2, car_width/2],
            [-car_length/2, -car_width/2],
            [car_length/2, -car_width/2],
        ])

        rotation = np.array([[cos_psi, -sin_psi],
                             [sin_psi, cos_psi]])
        corners_world = corners_local @ rotation.T
        corners_screen = [(screen_x + c[0], screen_y - c[1])
                          for c in corners_world]

        pygame.draw.polygon(self.screen, self.CAR_COLOR, corners_screen)
        pygame.draw.polygon(self.screen, (255, 255, 255), corners_screen, 2)

        # Draw heading indicator
        front_x = screen_x + cos_psi * car_length * 0.7
        front_y = screen_y - sin_psi * car_length * 0.7
        pygame.draw.line(self.screen, (255, 255, 0),
                         (screen_x, screen_y),
                         (int(front_x), int(front_y)), 4)

    def draw_info(self, step, reward, speed, cross_track_error):
        """Draw information panel."""
        y_offset = 15
        line_height = 35

        texts = [
            f"Step: {step}",
            f"Reward: {reward:+.3f}",
            f"Speed: {speed:.2f} m/s",
            f"Cross-track: {cross_track_error:.2f} m",
        ]

        for text_str in texts:
            text = self.font.render(text_str, True, self.TEXT_COLOR)
            # Draw text background
            text_rect = text.get_rect()
            text_rect.topleft = (15, y_offset)
            bg_rect = text_rect.inflate(10, 5)
            pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
            self.screen.blit(text, (15, y_offset))
            y_offset += line_height

    def render(self, state, step, reward, obs):
        """
        Render the current state.

        Args:
            state: Environment state with car position
            step: Current step number
            reward: Current reward
            obs: Observation array [s, e, theta_diff, vx, vy, omega, curv, curv_lh]
        """
        self.screen.fill(self.BG_COLOR)
        self.draw_track()

        # Extract car state
        car = state.car
        x = float(car.x)
        y = float(car.y)
        psi = float(car.psi)
        vx = float(car.vx)
        vy = float(car.vy)
        speed = np.sqrt(vx**2 + vy**2)

        # Extract cross-track error from observation
        cross_track_error = float(obs[1])

        self.draw_car(x, y, psi)
        self.draw_info(step, float(reward), speed, cross_track_error)

        if not self.headless:
            pygame.display.flip()

        # Return frame as numpy array for video recording
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))  # pygame uses (width, height, 3)
        return frame

    def close(self):
        """Close the visualizer."""
        pygame.quit()


def record_episode(
    env_reset,
    env_step,
    policy_fn,
    rng,
    max_steps=500,
    save_path=None,
    fps=20,
):
    """
    Record a single episode.

    Args:
        env_reset: Environment reset function
        env_step: Environment step function
        policy_fn: Function that takes (params, obs) and returns action
        rng: JAX random key
        max_steps: Maximum steps to record
        save_path: If provided, save video to this path
        fps: Frames per second for video

    Returns:
        frames: List of RGB frames
        total_reward: Total episode reward
        episode_length: Number of steps
    """
    visualizer = SingleAgentVisualizer(width=800, height=800, headless=True)

    # Reset environment
    state, obs = env_reset(rng)

    frames = []
    total_reward = 0.0
    step = 0

    for step in range(max_steps):
        # Get action from policy
        action = policy_fn(obs)

        # Render frame
        reward_to_display = total_reward / (step + 1) if step > 0 else 0.0
        frame = visualizer.render(state, step, reward_to_display, obs)
        frames.append(frame)

        # Step environment
        state, obs, reward, done, truncated = env_step(state, action)
        total_reward += float(reward)

        if done or truncated:
            break

    visualizer.close()

    # Save video if path provided
    if save_path is not None:
        save_video(frames, save_path, fps=fps)

    return frames, total_reward, step + 1


def save_video(frames, save_path, fps=20):
    """
    Save frames as video using imageio.

    Args:
        frames: List of RGB frames (numpy arrays)
        save_path: Path to save video
        fps: Frames per second
    """
    try:
        import imageio

        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Save video
        imageio.mimsave(save_path, frames, fps=fps)
        print(f"  Video saved to {save_path}")

    except ImportError:
        print("  Warning: imageio not installed. Cannot save video.")
        print("  Install with: pip install imageio imageio-ffmpeg")
    except Exception as e:
        print(f"  Warning: Could not save video: {e}")


def frames_to_wandb_video(frames, fps=20):
    """
    Convert frames to wandb.Video format.

    Args:
        frames: List of RGB frames (numpy arrays)
        fps: Frames per second

    Returns:
        wandb.Video object or None if wandb not available
    """
    try:
        import wandb

        # Stack frames into (T, H, W, C) format
        video_array = np.stack(frames, axis=0)

        # wandb expects (T, C, H, W) format
        video_array = np.transpose(video_array, (0, 3, 1, 2))

        return wandb.Video(video_array, fps=fps, format="mp4")

    except ImportError:
        print("  Warning: wandb not installed. Cannot create wandb video.")
        return None
    except Exception as e:
        print(f"  Warning: Could not create wandb video: {e}")
        return None
