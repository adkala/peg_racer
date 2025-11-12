# Video Creation Guide

This guide explains how to create videos of the racing environment for visualization and debugging.

## Quick Start

### Create a Video with Simple Controller

```bash
python3 create_video.py --output my_video.mp4
```

This creates a video using a simple proportional controller that:
- Adjusts throttle based on desired speed and track curvature
- Steers to minimize cross-track error and heading error

### Create a Video with Trained Policy

```bash
python3 create_video.py \
  --output trained_policy.mp4 \
  --policy single_agent_policy.pkl \
  --max-steps 500
```

## Command Line Arguments

- `--output`: Output video path (default: `racing_demo.mp4`)
- `--policy`: Path to trained policy pickle file (optional)
  - If not provided or file doesn't exist, uses simple controller
- `--max-steps`: Maximum steps to record (default: 500)
- `--fps`: Frames per second (default: 20)
- `--seed`: Random seed for reproducibility (default: 42)

## Examples

### Quick Demo Video
```bash
python3 create_video.py --output demo.mp4 --max-steps 200
```

### High-Quality Video
```bash
python3 create_video.py \
  --output high_quality.mp4 \
  --max-steps 1000 \
  --fps 30
```

### Compare Multiple Policies
```bash
# Baseline controller
python3 create_video.py --output baseline.mp4

# Iteration 100
python3 create_video.py \
  --policy checkpoints/iteration_100.pkl \
  --output iter_100.mp4

# Iteration 500
python3 create_video.py \
  --policy checkpoints/iteration_500.pkl \
  --output iter_500.mp4

# Final policy
python3 create_video.py \
  --policy single_agent_policy.pkl \
  --output final_policy.mp4
```

### Reproducible Videos (Same Seed)
```bash
python3 create_video.py --output run1.mp4 --seed 42
python3 create_video.py --output run2.mp4 --seed 42
# Both videos will be identical
```

### Different Random Scenarios
```bash
python3 create_video.py --output scenario1.mp4 --seed 1
python3 create_video.py --output scenario2.mp4 --seed 2
python3 create_video.py --output scenario3.mp4 --seed 3
```

## Video Contents

Each video frame shows:

1. **Track Layout**: Racing line with waypoint markers
2. **Car Position**: Red car with heading indicator (yellow line)
3. **Information Panel**:
   - Current step number
   - Episode reward (cumulative average)
   - Car speed (m/s)
   - Cross-track error (meters)

## Simple Controller

The built-in simple controller uses:

```python
# Target speed based on curvature
target_speed = 2.0 - 1.5 * abs(curvature)

# Throttle control
throttle = 0.5 + 0.3 * (target_speed - current_speed)

# Steering control
steering = -1.5 * cross_track_error - 0.8 * heading_error
```

This provides a reasonable baseline for comparison with trained policies.

## Using Trained Policies

To create videos with a trained policy, you need a pickle file containing:
- `params`: Trained network parameters
- `obs_dim`: Observation dimension
- `action_dim`: Action dimension

The script will automatically:
1. Load the policy from the pickle file
2. Create the neural network with saved parameters
3. Run deterministic actions (no exploration noise)
4. Generate a video of the policy's behavior

## Output

After running, you'll see:

```
Episode completed!
Total reward: -4466.17
Episode length: 500 steps
Frames recorded: 500
Video saved to: racing_demo.mp4
```

Key metrics:
- **Total reward**: Episode return (higher is better)
- **Episode length**: Number of steps before termination
- **Frames recorded**: Number of video frames (equals episode length)

## Video File Format

- **Format**: MP4 (H.264)
- **Resolution**: 800x800 pixels
- **Frame Rate**: 20 FPS (default, configurable)
- **Duration**: Approximately 25 seconds for 500 frames at 20 FPS

## Troubleshooting

### "imageio not installed"
```bash
pip install imageio imageio-ffmpeg
```

### Video won't play
Try converting to a more compatible format:
```bash
ffmpeg -i racing_demo.mp4 -vcodec h264 -acodec aac output.mp4
```

### Video is too large
Reduce the number of frames:
```bash
python3 create_video.py --max-steps 200
```

Or reduce FPS:
```bash
python3 create_video.py --fps 10
```

### Out of memory
Recording long videos may use significant RAM. Try reducing `--max-steps`:
```bash
python3 create_video.py --max-steps 250
```

## Integration with Training

During training, videos are automatically created at regular intervals. See `TRAINING_README.md` for details.

To create videos manually after training:

```bash
# Train the model
python3 train_single_agent.py --num-iterations 500

# Create video with trained policy
python3 create_video.py \
  --policy single_agent_policy.pkl \
  --output trained_demo.mp4
```

## Tips

1. **Compare Before/After**: Create videos at different training stages to visualize learning
2. **Use Different Seeds**: Test policy robustness across different scenarios
3. **Adjust FPS**: Lower FPS (10-15) for smaller files, higher (30) for smoother playback
4. **Max Steps**: 500 steps typically completes one full lap of the track

## Example Workflow

```bash
# 1. Create baseline video with simple controller
python3 create_video.py --output 00_baseline.mp4

# 2. Train for 100 iterations
python3 train_single_agent.py --num-iterations 100 --save-path policy_100.pkl

# 3. Create video of partially trained policy
python3 create_video.py --policy policy_100.pkl --output 01_iter100.mp4

# 4. Continue training
python3 train_single_agent.py --num-iterations 500 --save-path policy_500.pkl

# 5. Create final policy video
python3 create_video.py --policy policy_500.pkl --output 02_final.mp4

# 6. Compare all three videos
open 00_baseline.mp4 01_iter100.mp4 02_final.mp4
```
