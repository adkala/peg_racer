# Video Visualizer Update - Track Boundaries

## Summary

Updated the `SingleAgentVisualizer` in `video_recorder.py` to match the pygame-based visualizer in `visualizer.py` by adding track boundary visualization.

## Changes Made

### 1. Added Track Boundary Properties

```python
# Track data
self.track_waypoints = None
self.track_width = None              # NEW
self.track_left_boundary = None      # NEW
self.track_right_boundary = None     # NEW
self.track_scale = 1.0
self.track_offset = np.array([0.0, 0.0])
```

### 2. Load Track Width from YAML

```python
def load_track_data(self):
    # Load track width parameter
    self.track_width = params['track_info']['track_width'] * scale

    # Compute track boundaries
    self._compute_track_boundaries()
```

### 3. Added Track Boundary Computation

New method `_compute_track_boundaries()` that:
- Computes tangent vectors at each waypoint using adjacent points
- Calculates normal vectors (perpendicular to tangents)
- Generates left and right boundaries by offsetting from centerline
- Uses wrap-around for closed track (indices wrap at boundaries)

```python
def _compute_track_boundaries(self):
    # Compute tangent vectors using adjacent waypoints
    prev_points = np.roll(self.track_waypoints, 1, axis=0)
    next_points = np.roll(self.track_waypoints, -1, axis=0)

    # Tangent direction (from previous to next point)
    tangents = next_points - prev_points
    tangents_normalized = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)

    # Normal vectors (perpendicular, rotated 90 degrees)
    normals_left = np.column_stack([-tangents_normalized[:, 1],
                                     tangents_normalized[:, 0]])
    normals_right = -normals_left

    # Compute boundaries
    half_width = self.track_width / 2.0
    self.track_left_boundary = self.track_waypoints + normals_left * half_width
    self.track_right_boundary = self.track_waypoints + normals_right * half_width
```

### 4. Updated Track Drawing

Modified `draw_track()` to render:

1. **Track Boundaries** (yellow, 3px wide):
   - Left boundary line
   - Right boundary line

2. **Centerline** (gray, 1px wide):
   - Dimmer and thinner than before
   - Shows the racing line

3. **Waypoint Markers** (blue, 1px dots):
   - Every 50th waypoint (reduced frequency)
   - Smaller dots to reduce clutter

```python
def draw_track(self):
    # Draw track boundaries (yellow, thick)
    if self.track_left_boundary is not None:
        left_points = [self.world_to_screen(x, y) for x, y in self.track_left_boundary]
        pygame.draw.lines(self.screen, self.TRACK_BORDER_COLOR, True, left_points, 3)

    if self.track_right_boundary is not None:
        right_points = [self.world_to_screen(x, y) for x, y in self.track_right_boundary]
        pygame.draw.lines(self.screen, self.TRACK_BORDER_COLOR, True, right_points, 3)

    # Draw centerline (gray, thin)
    points = [self.world_to_screen(x, y) for x, y in self.track_waypoints]
    pygame.draw.lines(self.screen, (100, 100, 100), True, points, 1)

    # Draw waypoint dots (blue, small, sparse)
    for x, y in self.track_waypoints[::50]:
        screen_pos = self.world_to_screen(x, y)
        pygame.draw.circle(self.screen, self.WAYPOINT_COLOR, screen_pos, 1)
```

## Visual Comparison

### Before (Old Visualization)
- Single yellow centerline (3px thick)
- Waypoint markers every 20 points (3px dots)
- No track boundaries visible
- Harder to see valid racing area

### After (New Visualization)
- **Yellow track boundaries** (left and right, 3px thick)
- Gray centerline (1px thin)
- Waypoint markers every 50 points (1px dots)
- Clear visual indication of track limits
- Easier to see if car is going off-track

## Benefits

1. **Better Understanding**: Clearly shows the valid racing area
2. **Off-Track Detection**: Easy to see when car crosses boundaries
3. **Policy Debugging**: Helps identify if policy respects track limits
4. **Visual Appeal**: More professional-looking videos
5. **Consistency**: Matches the main pygame visualizer

## Files Updated

- `video_recorder.py`: Updated `SingleAgentVisualizer` class
- `create_video.py`: No changes needed (uses updated visualizer)
- `train_single_agent.py`: No changes needed (uses updated visualizer)

## Testing

Tested with:
1. Standalone video creation: `python3 create_video.py`
2. Training integration: Videos generated during training
3. Both simple controller and trained policies

All videos now show proper track boundaries.

## Backward Compatibility

✅ Fully backward compatible
- Existing scripts work without modification
- Videos are still MP4 format at same resolution
- Performance impact is negligible
- Falls back gracefully if track width not available

## Usage

No changes required to existing code. Simply run:

```bash
# Create standalone video
python3 create_video.py --output demo.mp4

# Train with video recording
python3 train_single_agent.py --num-iterations 100 --video-frequency 25
```

Videos will automatically include track boundaries.
