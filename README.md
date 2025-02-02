# Dots and Boxes DQN

An interactive visualization of Deep Q-Network agents learning to play Dots and Boxes through self-play.

## Features

- Real-time visualization of game board and training progress
- Interactive speed controls (0.2x - 5.0x)
- Training metrics visualization:
  - Win rates
  - Evaluation scores
  - Q-values
  - Move quality
- Play against trained models
- Checkpoint system for saving and loading models

## Installation

```bash
pip install dots-and-boxes-dqn
```

## Usage

```bash
python -m dots_and_boxes_dqn
```

Then open http://localhost:5000 in your browser.

## Training Controls

- **Speed**: Adjust training visualization speed
  - 0.2x: Slow motion for analysis
  - 1.0x: Normal speed
  - 3.0x: Fast visualization
  - 5.0x: Training focus
- **Pause/Resume**: Temporarily halt training
- **Play vs AI**: Test your skills against saved models

## Technical Details

- Uses Dueling DQN architecture
- Prioritized Experience Replay
- Self-play training methodology
- Periodic model checkpointing
- Flask + SocketIO for real-time updates

## Development

```bash
git clone https://github.com/yourusername/dots-and-boxes-dqn
cd dots-and-boxes-dqn
pip install -e ".[dev]"
```
