"""Heuristic board evaluation used by the solver agent."""

from .constants import (
    ALL_DIRS,
    COLS,
    DIRECTION_VEC,
    GhostMode,
    OPPOSITE_DIR,
    ROWS,
    T_DOT,
    T_PELLET,
    TUNNEL_ROW,
)
from .features import shortest_path_distance, straight_path_length


MAX_DISTANCE = ROWS * COLS
MOVE_ARROWS = {
    "up": "U",
    "left": "L",
    "down": "D",
    "right": "R",
}


def _normalize_cell(col, row):
    if row == TUNNEL_ROW:
        if col < 0:
            col += COLS
        elif col >= COLS:
            col -= COLS
    return col, row


def _distance_to_positions(maze, start, positions):
    if not positions:
        return MAX_DISTANCE
    return shortest_path_distance(
        maze,
        start,
        lambda col, row: (col, row) in positions,
    )


def _target_tile(maze, col, row):
    if 0 <= row < ROWS and 0 <= col < COLS:
        return maze.grid[row][col]
    return None


def _classify_move(move):
    if move["blocked"]:
        return "BLOCKED"
    if move["danger_distance"] <= 1:
        return "TRAP"
    if move["frightened_distance"] <= 2 and move["frightened_distance"] < move["danger_distance"]:
        return "CHASE"
    if move["pellet_distance"] <= 2 and move["danger_distance"] <= 4:
        return "POWER"
    if move["danger_distance"] <= 3:
        return "RISK"
    return "SAFE"


def _score_move(move, current_direction):
    if move["blocked"]:
        return -10_000.0

    score = 0.0

    if move["target_tile"] == T_PELLET:
        score += 14.0
    elif move["target_tile"] == T_DOT:
        score += 5.0

    score += max(0, 6 - min(move["dot_distance"], 6)) * 1.4
    score += min(move["corridor_length"], 4) * 0.8

    if move["danger_distance"] <= 6:
        score -= (7 - move["danger_distance"]) * 4.0

    if move["pellet_distance"] < MAX_DISTANCE and move["danger_distance"] <= 5:
        score += max(0, 5 - min(move["pellet_distance"], 5)) * 2.2

    if move["frightened_distance"] < MAX_DISTANCE:
        score += max(0, 5 - min(move["frightened_distance"], 5)) * 3.2

    if move["direction"] == current_direction:
        score += 1.6
    elif current_direction and move["direction"] == OPPOSITE_DIR.get(current_direction):
        score -= 1.1

    return score


def analyze_scene(scene):
    maze = scene.maze
    current_direction = scene.pacman.direction
    pac_col, pac_row = scene.pacman.get_cell()
    start = (pac_col, pac_row)

    dangerous_positions = {
        ghost.get_cell()
        for ghost in scene.ghosts
        if ghost.mode not in (GhostMode.FRIGHTENED, GhostMode.EATEN, GhostMode.INDOOR)
    }
    frightened_positions = {
        ghost.get_cell()
        for ghost in scene.ghosts
        if ghost.mode == GhostMode.FRIGHTENED
    }

    moves = []
    for direction in ALL_DIRS:
        can_move = maze.can_move(pac_col, pac_row, direction, is_ghost=False)
        dx, dy = DIRECTION_VEC[direction]
        next_col, next_row = _normalize_cell(pac_col + dx, pac_row + dy)

        move = {
            "direction": direction,
            "arrow": MOVE_ARROWS[direction],
            "blocked": not can_move,
            "next_cell": (next_col, next_row),
        }

        if can_move:
            target_tile = _target_tile(maze, next_col, next_row)
            move["target_tile"] = target_tile
            move["corridor_length"] = straight_path_length(maze, pac_col, pac_row, direction)
            move["danger_distance"] = _distance_to_positions(
                maze,
                (next_col, next_row),
                dangerous_positions,
            )
            move["frightened_distance"] = _distance_to_positions(
                maze,
                (next_col, next_row),
                frightened_positions,
            )
            move["dot_distance"] = shortest_path_distance(
                maze,
                (next_col, next_row),
                lambda col, row: maze.grid[row][col] == T_DOT,
            )
            move["pellet_distance"] = shortest_path_distance(
                maze,
                (next_col, next_row),
                lambda col, row: maze.grid[row][col] == T_PELLET,
            )
        else:
            move["target_tile"] = None
            move["corridor_length"] = 0
            move["danger_distance"] = MAX_DISTANCE
            move["frightened_distance"] = MAX_DISTANCE
            move["dot_distance"] = MAX_DISTANCE
            move["pellet_distance"] = MAX_DISTANCE

        move["status"] = _classify_move(move)
        move["score"] = _score_move(move, current_direction)
        moves.append(move)

    legal_moves = [move for move in moves if not move["blocked"]]
    safe_moves = [move for move in legal_moves if move["danger_distance"] >= 4]
    best_move = max(legal_moves, key=lambda move: move["score"], default=None)

    nearest_active = _distance_to_positions(maze, start, dangerous_positions)
    nearest_frightened = _distance_to_positions(maze, start, frightened_positions)
    nearest_dot = shortest_path_distance(
        maze,
        start,
        lambda col, row: maze.grid[row][col] == T_DOT,
    )
    nearest_pellet = shortest_path_distance(
        maze,
        start,
        lambda col, row: maze.grid[row][col] == T_PELLET,
    )

    if not legal_moves or not best_move or best_move["danger_distance"] <= 1:
        board_state = "LOSING"
        reason = "Ghost pressure is immediate. Survive before chasing points."
    elif best_move["status"] == "POWER":
        board_state = "TACTICAL"
        reason = "A power pellet is the best swing move on this board."
    elif nearest_active <= 3:
        board_state = "TACTICAL"
        reason = "The next label matters a lot. Favor open escape lanes."
    elif frightened_positions and nearest_frightened <= 4:
        board_state = "WINNING"
        reason = "A frightened ghost is close enough to turn into score."
    elif maze.total_dots <= 20:
        board_state = "WINNING"
        reason = "The board is stable and the level is close to finished."
    else:
        board_state = "WINNING"
        reason = "The board is stable and safe dot routes are available."

    if board_state == "LOSING":
        learning_focus = [
            "maximize distance from dangerous ghosts",
            "avoid dead ends and short corridors",
        ]
    elif best_move and best_move["status"] == "POWER":
        learning_focus = [
            "route toward power pellets under pressure",
            "keep at least one safe follow-up lane",
        ]
    elif best_move and best_move["status"] == "CHASE":
        learning_focus = [
            "convert frightened ghosts into points",
            "stay on the scoring lane while it is safe",
        ]
    else:
        learning_focus = [
            "collect dots without giving up safety",
            "prefer moves with more room to recover",
        ]

    return {
        "board_state": board_state,
        "reason": reason,
        "learning_focus": learning_focus,
        "recommended_move": best_move["direction"] if best_move else None,
        "recommended_status": best_move["status"] if best_move else "BLOCKED",
        "recommended_score": best_move["score"] if best_move else None,
        "moves": moves,
        "legal_moves": len(legal_moves),
        "safe_moves": len(safe_moves),
        "nearest_active_ghost_dist": nearest_active,
        "nearest_frightened_ghost_dist": nearest_frightened,
        "nearest_dot_dist": nearest_dot,
        "nearest_pellet_dist": nearest_pellet,
        "dots_remaining": maze.total_dots,
    }


def choose_board_move(scene):
    """Return the strongest heuristic move for the current scene."""
    board_view = analyze_scene(scene)
    return board_view["recommended_move"], board_view
