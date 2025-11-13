"""
Pygame-based visualizer for car_jax racing environment.
Displays cars, track, and waypoints in real-time with mp4 recording support.
"""

import pygame
import numpy as np
import jax.numpy as jnp
import pandas as pd
import yaml
import os
from typing import Optional, List
import subprocess
import tempfile
import shutil


class RacingVisualizer:
    """
    Lightweight pygame visualizer for the racing environment.
    Supports real-time rendering and mp4 video recording.
    """

    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        fps: int = 20,
        track_yaml_path: Optional[str] = None,
        ref_trajs_dir: Optional[str] = None,
        record_video: bool = False,
        video_path: str = "racing_output.mp4"
    ):
        """
        Initialize the visualizer.

        Args:
            width: Window width in pixels
            height: Window height in pixels
            fps: Target frames per second
            track_yaml_path: Path to track YAML config (defaults to ./data/params-num.yaml)
            ref_trajs_dir: Directory with reference trajectories (defaults to ./ref_trajs)
            record_video: Whether to record video to mp4
            video_path: Output path for mp4 video
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("car_jax Racing Visualizer")
        self.clock = pygame.time.Clock()
        self.fps = fps

        # Video recording
        self.record_video = record_video
        self.video_path = video_path
        self.frames = [] if record_video else None
        self.temp_dir = None

        # Colors
        self.BG_COLOR = (20, 20, 30)
        self.TRACK_COLOR = (100, 100, 100)
        self.TRACK_BORDER_COLOR = (200, 200, 50)
        self.WAYPOINT_COLOR = (50, 150, 255)
        self.CAR_COLORS = [
            (255, 50, 50),    # Red
            (50, 255, 50),    # Green
            (50, 150, 255),   # Blue
        ]
        self.TEXT_COLOR = (255, 255, 255)

        # Track data
        self.track_waypoints = None
        self.track_left_boundary = None
        self.track_right_boundary = None
        self.track_width = 1.0
        self.track_scale = 1.0
        self.track_offset = np.array([0.0, 0.0])

        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Load track data
        if track_yaml_path is None:
            track_yaml_path = os.path.join(os.path.dirname(__file__), "data", "params-num.yaml")
        if ref_trajs_dir is None:
            ref_trajs_dir = os.path.join(os.path.dirname(__file__), "ref_trajs")
        self.load_track_data(track_yaml_path, ref_trajs_dir)

    def load_track_data(self, yaml_path: str, ref_trajs_dir: str):
        """Load track waypoints from data files."""
        try:
            # Load YAML config
            with open(yaml_path, 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)

            centerline_file = params['track_info']['centerline_file']
            if centerline_file.endswith('.csv'):
                centerline_file = centerline_file[:-4]

            scale = params['track_info'].get('scale', 1.0)
            ox = params['track_info'].get('ox', 0)
            oy = params['track_info'].get('oy', 0)
            self.track_width = params['track_info'].get('track_width', 4.0) * scale

            # Load CSV trajectory
            csv_path = os.path.join(ref_trajs_dir, f'{centerline_file}_with_speeds.csv')
            df = pd.read_csv(csv_path)

            # Extract x, y positions and apply scale/offset
            waypoints = np.array(df.iloc[:, 1:3]) * scale + np.array([ox, oy])
            self.track_waypoints = waypoints

            # Compute track boundaries
            self._compute_track_boundaries()

            # Compute appropriate scale and offset for visualization
            self._compute_viewport()

        except Exception as e:
            print(f"Warning: Could not load track data: {e}")
            # Use default waypoints (circle)
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
        n_points = len(self.track_waypoints)
        prev_points = np.roll(self.track_waypoints, 1, axis=0)
        next_points = np.roll(self.track_waypoints, -1, axis=0)

        # Tangent direction (from previous to next point)
        tangents = next_points - prev_points
        tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangent_norms = np.maximum(tangent_norms, 1e-6)
        tangents_normalized = tangents / tangent_norms

        # Normal vectors (perpendicular to tangent, rotated 90 degrees)
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

        # Get track bounds
        min_x, min_y = self.track_waypoints.min(axis=0)
        max_x, max_y = self.track_waypoints.max(axis=0)

        track_width = max_x - min_x
        track_height = max_y - min_y

        # Add margin
        margin = 0.1
        track_width *= (1 + 2 * margin)
        track_height *= (1 + 2 * margin)

        # Compute scale to fit in window
        scale_x = (self.width * 0.8) / track_width
        scale_y = (self.height * 0.8) / track_height
        self.track_scale = min(scale_x, scale_y)

        # Center track in window
        track_center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
        window_center = np.array([self.width / 2, self.height / 2])
        self.track_offset = window_center - track_center * self.track_scale

    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        screen_x = x * self.track_scale + self.track_offset[0]
        screen_y = -y * self.track_scale + self.track_offset[1]  # Flip Y axis
        return int(screen_x), int(screen_y)

    def draw_track(self):
        """Draw the track using waypoints."""
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

        # Draw waypoint dots (every 50th waypoint)
        for x, y in self.track_waypoints[::50]:
            screen_pos = self.world_to_screen(x, y)
            pygame.draw.circle(self.screen, self.WAYPOINT_COLOR, screen_pos, 1)

    def draw_car(self, x, y, psi, car_id=0, label=""):
        """
        Draw a car.

        Args:
            x, y: Position in world coordinates
            psi: Heading angle in radians
            car_id: Car index (0, 1, 2) for color
            label: Optional label to display near car
        """
        screen_x, screen_y = self.world_to_screen(x, y)

        # Car dimensions (in pixels)
        car_length = 20
        car_width = 12

        # Compute car corners
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        # Car rectangle corners (in car frame)
        corners_local = np.array([
            [car_length/2, car_width/2],
            [-car_length/2, car_width/2],
            [-car_length/2, -car_width/2],
            [car_length/2, -car_width/2],
        ])

        # Rotate and translate to world coordinates
        rotation = np.array([[cos_psi, -sin_psi],
                             [sin_psi, cos_psi]])
        corners_world = corners_local @ rotation.T
        corners_screen = [(screen_x + c[0], screen_y - c[1])
                          for c in corners_world]

        # Draw car body
        color = self.CAR_COLORS[car_id % len(self.CAR_COLORS)]
        pygame.draw.polygon(self.screen, color, corners_screen)
        pygame.draw.polygon(self.screen, (255, 255, 255), corners_screen, 2)

        # Draw heading indicator (front of car)
        front_x = screen_x + cos_psi * car_length * 0.6
        front_y = screen_y - sin_psi * car_length * 0.6
        pygame.draw.line(self.screen, (255, 255, 0),
                         (screen_x, screen_y),
                         (int(front_x), int(front_y)), 3)

        # Draw label if provided
        if label:
            text = self.small_font.render(label, True, self.TEXT_COLOR)
            self.screen.blit(text, (screen_x + 15, screen_y - 15))

    def draw_info(self, step, rewards, speeds):
        """
        Draw information panel.

        Args:
            step: Current step number
            rewards: List of rewards for each car
            speeds: List of speeds for each car
        """
        y_offset = 10
        line_height = 25

        # Draw step counter
        text = self.font.render(f"Step: {step}", True, self.TEXT_COLOR)
        self.screen.blit(text, (10, y_offset))
        y_offset += line_height

        # Draw car info
        for i, (reward, speed) in enumerate(zip(rewards, speeds)):
            color = self.CAR_COLORS[i]
            text = self.font.render(
                f"Car {i+1}: R={reward:+.3f} V={speed:.2f}m/s",
                True, color
            )
            self.screen.blit(text, (10, y_offset))
            y_offset += line_height

        # Draw controls info at bottom
        controls_text = [
            "Controls:",
            "ESC - Quit",
            "SPACE - Pause/Resume",
        ]
        if self.record_video:
            controls_text.append(f"Recording: {self.video_path}")

        y_offset = self.height - len(controls_text) * 20 - 10
        for line in controls_text:
            text = self.small_font.render(line, True, (180, 180, 180))
            self.screen.blit(text, (10, y_offset))
            y_offset += 20

    def render(self, state, step, rewards, paused=False):
        """
        Render the current state.

        Args:
            state: CarState from car_jax with fields (x, y, psi, vx, vy, omega)
            step: Current step number
            rewards: Rewards for each car (numpy array or list)
            paused: Whether simulation is paused
        """
        # Clear screen
        self.screen.fill(self.BG_COLOR)

        # Draw track
        self.draw_track()

        # Convert JAX arrays to numpy if needed
        if hasattr(state, 'x'):
            # Extract car states from CarState
            x_arr = np.array(state.x) if isinstance(state.x, jnp.ndarray) else state.x
            y_arr = np.array(state.y) if isinstance(state.y, jnp.ndarray) else state.y
            psi_arr = np.array(state.psi) if isinstance(state.psi, jnp.ndarray) else state.psi
            vx_arr = np.array(state.vx) if isinstance(state.vx, jnp.ndarray) else state.vx
            vy_arr = np.array(state.vy) if isinstance(state.vy, jnp.ndarray) else state.vy

            # Handle both single car and multi-car cases
            if np.ndim(x_arr) == 0:
                n_cars = 1
                x_arr = np.array([x_arr])
                y_arr = np.array([y_arr])
                psi_arr = np.array([psi_arr])
                vx_arr = np.array([vx_arr])
                vy_arr = np.array([vy_arr])
            else:
                n_cars = len(x_arr)
        else:
            raise ValueError("State must have x, y, psi, vx, vy attributes")

        # Draw each car
        speeds = []
        for i in range(n_cars):
            x = float(x_arr[i])
            y = float(y_arr[i])
            psi = float(psi_arr[i])
            vx = float(vx_arr[i])
            vy = float(vy_arr[i])
            speed = np.sqrt(vx**2 + vy**2)
            speeds.append(speed)

            label = f"Car {i+1}"
            self.draw_car(x, y, psi, i, label)

        # Convert rewards to list if needed
        if isinstance(rewards, (jnp.ndarray, np.ndarray)):
            rewards = [float(r) for r in rewards]

        # Draw info panel
        self.draw_info(step, rewards, speeds)

        # Draw pause indicator
        if paused:
            text = self.font.render("PAUSED", True, (255, 255, 0))
            text_rect = text.get_rect(center=(self.width // 2, 30))
            self.screen.blit(text, text_rect)

        # Update display
        pygame.display.flip()

        # Capture frame for video if recording
        if self.record_video:
            self._capture_frame()

        self.clock.tick(self.fps)

    def _capture_frame(self):
        """Capture current screen as a frame for video."""
        # Get the current surface as a numpy array
        frame_surface = pygame.surfarray.array3d(self.screen)
        # Pygame uses (width, height, 3), but we need (height, width, 3)
        frame = np.transpose(frame_surface, (1, 0, 2))
        self.frames.append(frame.copy())

    def save_video(self):
        """Save captured frames as mp4 video using ffmpeg."""
        if not self.record_video or not self.frames:
            print("No frames to save")
            return

        print(f"Saving video with {len(self.frames)} frames...")

        # Create temporary directory for frames
        self.temp_dir = tempfile.mkdtemp()

        try:
            # Save frames as PNG files
            for i, frame in enumerate(self.frames):
                from PIL import Image
                img = Image.fromarray(frame.astype('uint8'), 'RGB')
                img.save(os.path.join(self.temp_dir, f'frame_{i:06d}.png'))

            # Use ffmpeg to create video
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-framerate', str(self.fps),
                '-i', os.path.join(self.temp_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-preset', 'medium',
                '-crf', '23',
                self.video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✓ Video saved to: {self.video_path}")
            else:
                print(f"✗ Error saving video: {result.stderr}")

        except Exception as e:
            print(f"✗ Error saving video: {e}")
        finally:
            # Cleanup temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

    def check_quit(self):
        """Check if user wants to quit or pause."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return 'quit'
                if event.key == pygame.K_SPACE:
                    return 'pause'
        return None

    def close(self):
        """Close the visualizer and save video if recording."""
        if self.record_video:
            self.save_video()
        pygame.quit()
