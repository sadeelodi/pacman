"""Feature extraction helpers for ML-based Pac-Man agents."""

from collections import deque

from .constants import (
    COLS,
    ROWS,
    DIR_UP,
    DIR_LEFT,
    DIR_DOWN,
    DIR_RIGHT,
    DIRECTION_VEC,
    T_DOT,
    T_PELLET,
    GhostMode,
)


FEATURE_DIRS = (DIR_UP, DIR_LEFT, DIR_DOWN, DIR_RIGHT)
DIR_TO_INT = {
    "none": 0,
    DIR_UP: 1,
    DIR_LEFT: 2,
    DIR_DOWN: 3,
    DIR_RIGHT: 4,
}


def encode_direction(direction: str) -> int:
    """Convert a direction string into a stable integer."""
    return DIR_TO_INT.get(direction, 0)


def _iter_neighbors(maze, col, row):
    for direction in FEATURE_DIRS:
        if maze.can_move(col, row, direction, is_ghost=False):
            dx, dy = DIRECTION_VEC[direction]
            yield col + dx, row + dy


def shortest_path_distance(maze, start, target_fn):
    """Breadth-first search distance from start to the nearest target tile."""
    queue = deque([(start[0], start[1], 0)])
    visited = {start}

    while queue:
        col, row, dist = queue.popleft()
        if target_fn(col, row):
            return dist

        for next_col, next_row in _iter_neighbors(maze, col, row):
            if not (0 <= next_col < COLS and 0 <= next_row < ROWS):
                continue
            node = (next_col, next_row)
            if node in visited:
                continue
            visited.add(node)
            queue.append((next_col, next_row, dist + 1))

    return ROWS * COLS


def straight_path_length(maze, col, row, direction):
    """Number of free tiles ahead until the next wall."""
    distance = 0
    current_col, current_row = col, row

    while maze.can_move(current_col, current_row, direction, is_ghost=False):
        dx, dy = DIRECTION_VEC[direction]
        current_col += dx
        current_row += dy
        if not (0 <= current_col < COLS and 0 <= current_row < ROWS):
            break
        distance += 1

    return distance


def directional_ghost_pressure(scene, direction):
    """Distance to the nearest dangerous ghost if we move in this direction."""
    maze = scene.maze
    start_col, start_row = scene.pacman.get_cell()
    if not maze.can_move(start_col, start_row, direction, is_ghost=False):
        return ROWS * COLS

    dx, dy = DIRECTION_VEC[direction]
    target = (start_col + dx, start_row + dy)
    dangerous_positions = {
        ghost.get_cell()
        for ghost in scene.ghosts
        if ghost.mode not in (GhostMode.FRIGHTENED, GhostMode.EATEN, GhostMode.INDOOR)
    }

    if not dangerous_positions:
        return ROWS * COLS

    return shortest_path_distance(
        maze,
        target,
        lambda col, row: (col, row) in dangerous_positions,
    )


def extract_features(scene) -> dict[str, int]:
    """Build a numeric feature dictionary from the current gameplay state."""
    maze = scene.maze
    pacman = scene.pacman
    pac_col, pac_row = pacman.get_cell()

    features: dict[str, int] = {
        "pacman_col": pac_col,
        "pacman_row": pac_row,
        "pacman_direction": encode_direction(pacman.direction),
        "queued_direction": encode_direction(pacman.queued_dir),
        "dots_remaining": maze.total_dots,
        "dots_eaten": scene.dots_eaten,
        "score": scene.score,
        "level": scene.level,
        "lives": scene.lives,
    }

    for direction in FEATURE_DIRS:
        features[f"can_{direction}"] = int(
            maze.can_move(pac_col, pac_row, direction, is_ghost=False)
        )
        features[f"corridor_len_{direction}"] = straight_path_length(
            maze, pac_col, pac_row, direction
        )
        features[f"ghost_pressure_{direction}"] = directional_ghost_pressure(
            scene, direction
        )

    features["nearest_dot_dist"] = shortest_path_distance(
        maze,
        (pac_col, pac_row),
        lambda col, row: maze.grid[row][col] == T_DOT,
    )
    features["nearest_pellet_dist"] = shortest_path_distance(
        maze,
        (pac_col, pac_row),
        lambda col, row: maze.grid[row][col] == T_PELLET,
    )

    active_ghost_dists = []
    frightened_ghost_dists = []

    for ghost in scene.ghosts:
        ghost_col, ghost_row = ghost.get_cell()
        dx = ghost_col - pac_col
        dy = ghost_row - pac_row
        manhattan = abs(dx) + abs(dy)

        features[f"{ghost.name}_dx"] = dx
        features[f"{ghost.name}_dy"] = dy
        features[f"{ghost.name}_distance"] = manhattan
        features[f"{ghost.name}_mode"] = ghost.mode.value
        features[f"{ghost.name}_frightened"] = int(ghost.mode == GhostMode.FRIGHTENED)
        features[f"{ghost.name}_dangerous"] = int(
            ghost.mode not in (GhostMode.FRIGHTENED, GhostMode.EATEN, GhostMode.INDOOR)
        )

        if ghost.mode == GhostMode.FRIGHTENED:
            frightened_ghost_dists.append(manhattan)
        elif ghost.mode not in (GhostMode.EATEN, GhostMode.INDOOR):
            active_ghost_dists.append(manhattan)

    features["nearest_active_ghost_dist"] = (
        min(active_ghost_dists) if active_ghost_dists else ROWS * COLS
    )
    features["nearest_frightened_ghost_dist"] = (
        min(frightened_ghost_dists) if frightened_ghost_dists else ROWS * COLS
    )

    return features
