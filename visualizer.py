"""
Pygame-based visualizer for jit_neppo racing environment.
Displays cars, track, and waypoints in real-time.
"""

import pygame
import numpy as np
import pandas as pd
import yaml


class RacingVisualizer:
    """
    Lightweight pygame visualizer for the racing environment.
    """

    def __init__(self, width=1200, height=800, fps=20):
        """
        Initialize the visualizer.

        Args:
            width: Window width in pixels
            height: Window height in pixels
            fps: Target frames per second
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("JIT NEPPO Racing Visualizer")
        self.clock = pygame.time.Clock()
        self.fps = fps

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
        self.track_scale = 1.0
        self.track_offset = np.array([0.0, 0.0])

        # Font
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Load track data
        self.load_track_data()

    def load_track_data(self):
        """Load track waypoints from data files."""
        try:
            # Load YAML config
            with open('data/params-num.yaml', 'r') as f:
                params = yaml.load(f, Loader=yaml.FullLoader)

            centerline_file = params['track_info']['centerline_file'][:-4]
            scale = params['track_info']['scale']
            ox = params['track_info']['ox']
            oy = params['track_info']['oy']

            # Load CSV trajectory
            csv_path = f'data/ref_trajs/{centerline_file}_with_speeds.csv'
            df = pd.read_csv(csv_path, comment='#')

            # Extract x, y positions and apply scale/offset
            waypoints = np.array(df.iloc[:, 1:3]) * scale + np.array([ox, oy])
            self.track_waypoints = waypoints

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
            self._compute_viewport()

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

        # Draw track centerline
        points = [self.world_to_screen(x, y) for x, y in self.track_waypoints]
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.TRACK_BORDER_COLOR, True, points, 2)

        # Draw waypoint dots
        for x, y in self.track_waypoints[::20]:  # Draw every 20th waypoint
            screen_pos = self.world_to_screen(x, y)
            pygame.draw.circle(self.screen, self.WAYPOINT_COLOR, screen_pos, 2)

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
        y_offset = self.height - len(controls_text) * 20 - 10
        for line in controls_text:
            text = self.small_font.render(line, True, (180, 180, 180))
            self.screen.blit(text, (10, y_offset))
            y_offset += 20

    def render(self, state, step, rewards, paused=False):
        """
        Render the current state.

        Args:
            state: Environment state with car positions
            step: Current step number
            rewards: Rewards for each car
            paused: Whether simulation is paused
        """
        # Clear screen
        self.screen.fill(self.BG_COLOR)

        # Draw track
        self.draw_track()

        # Extract car states
        cars = state.cars
        n_cars = cars.x.shape[0]

        # Draw each car
        speeds = []
        for i in range(n_cars):
            x = float(cars.x[i])
            y = float(cars.y[i])
            psi = float(cars.psi[i])
            vx = float(cars.vx[i])
            vy = float(cars.vy[i])
            speed = np.sqrt(vx**2 + vy**2)
            speeds.append(speed)

            label = f"Car {i+1}"
            self.draw_car(x, y, psi, i, label)

        # Draw info panel
        self.draw_info(step, rewards, speeds)

        # Draw pause indicator
        if paused:
            text = self.font.render("PAUSED", True, (255, 255, 0))
            text_rect = text.get_rect(center=(self.width // 2, 30))
            self.screen.blit(text, text_rect)

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

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
        """Close the visualizer."""
        pygame.quit()
