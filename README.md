# Pac-Man

A Pac-Man game built with Python and Pygame, with both a learned model mode and a built-in heuristic solver.

## Setup

```bash
python -m venv .env
.\.env\Scripts\activate
pip install -r requirements.txt
```

## Play

```bash
python main.py
```

## Controls

- `Arrow keys` / `WASD`: move Pac-Man manually
- `Enter` / `Space`: start or continue
- `M`: toggle the trained model controller
- `H`: toggle the heuristic solver controller
- `Esc`: quit

## Heuristic Solver

The `H` mode uses a built-in rule-based controller implemented in `pacman/agent.py` and `pacman/board_analysis.py`.

At each tile center, the solver evaluates the four movement directions and scores every legal move using the current board state. The scoring logic favors:

- staying away from dangerous ghosts
- moving toward dots and power pellets
- chasing frightened ghosts when they are safely reachable
- choosing moves with longer open corridors
- keeping momentum by slightly preferring the current direction over reversing

For each candidate move, the solver computes:

- whether the move is blocked
- the next tile type
- shortest-path distance to dangerous ghosts
- shortest-path distance to frightened ghosts
- shortest-path distance to the nearest dot
- shortest-path distance to the nearest power pellet
- straight corridor length in that direction

It then assigns a handcrafted score and picks the legal move with the highest value. In short, the policy is:

1. Survive first.
2. Take a power pellet if pressure is rising.
3. Chase frightened ghosts when it is safe.
4. Otherwise clear dots efficiently.

## Data Collection

Labeled gameplay data is collected into `data/training_data.csv`.

The collection workflow uses the built-in heuristic solver algorithm in `H` mode, and solver decisions are logged as labeled moves for the classification task.

## Training

Train the classifier with:

```bash
python train_model.py
```

The training script fits a `StandardScaler` before the classifier, saves the trained model under `models/`, and saves the fitted scaler under `scaler/`.

## License

MIT
