"""
test_alphaze.py — Interactive tester for the trained AlphaZe** DarkHex model

Modes
─────
  1. Human vs AlphaZe**   — you play in the terminal, the model responds
  2. AlphaZe** vs Random  — run N games and report win rate
  3. AlphaZe** vs AlphaZe** (self-play) — watch the model play itself

Usage
─────
  python3 test_alphaze.py                        # interactive menu
  python3 test_alphaze.py --mode eval --games 50 # quick benchmark

Board layout (3×3)
──────────────────
  Cells are numbered 0-8, row-major:
    0 | 1 | 2
    3 | 4 | 5
    6 | 7 | 8
  P0 (X) connects TOP ↔ BOTTOM
  P1 (O) connects LEFT ↔ RIGHT
"""

import os
import sys
import random
import argparse
import numpy as np

# ── JAX / backend ─────────────────────────────────────────────────────────────
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# ── OpenSpiel path ─────────────────────────────────────────────────────────────
OPEN_SPIEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../3rdparty/open_spiel"
)
if OPEN_SPIEL_PATH not in sys.path:
    sys.path.insert(0, OPEN_SPIEL_PATH)

import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.algorithms.alpha_zero.evaluator import AlphaZeroEvaluator


import tensorflow as tf

# ── Same Keras 3 patch as training ────────────────────────────────────────────
def _patched_batch_norm(training, updates, name):
    def batch_norm_layer(x):
        bn = tf.keras.layers.BatchNormalization(name=name, trainable=True)
        return bn(x)
    return batch_norm_layer

model_lib.batch_norm = _patched_batch_norm


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG  — must match trainalphaze.py exactly
# ═══════════════════════════════════════════════════════════════════════════════
BOARD_SIZE     = 3
NUM_CELLS      = BOARD_SIZE * BOARD_SIZE
NUM_OBS_PLANES = 9
NUM_ACTIONS    = NUM_CELLS

MODEL_TYPE     = "resnet"
OBS_SHAPE      = [BOARD_SIZE, BOARD_SIZE, NUM_OBS_PLANES]

CHECKPOINT_DIR  = "./hex_alphaze_checkpoint_2"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "model.ckpt-16")

N_WORLDS         = 5
MCTS_SIMULATIONS = 200
UCT_C            = 1.5


# ═══════════════════════════════════════════════════════════════════════════════
# GAMES
# ═══════════════════════════════════════════════════════════════════════════════
dark_hex_game = pyspiel.load_game(
    f"dark_hex(num_rows={BOARD_SIZE},num_cols={BOARD_SIZE})"
)
hex_game = pyspiel.load_game(
    f"hex(num_rows={BOARD_SIZE},num_cols={BOARD_SIZE})"
)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path=CHECKPOINT_PATH):
    print(f"Loading model from: {checkpoint_path} …")
    m = model_lib.Model.build_model(
        MODEL_TYPE,
        OBS_SHAPE,
        NUM_ACTIONS,
        nn_width=64,
        nn_depth=4,
        weight_decay=1e-4,
        learning_rate=1e-3,
        path=CHECKPOINT_DIR,
    )
    m.load_checkpoint(checkpoint_path)
    print("Model loaded.")
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS  (identical to training script)
# ═══════════════════════════════════════════════════════════════════════════════

def get_obs(state):
    return np.array(state.observation_tensor(), dtype=np.float32).flatten()


def legal_mask(state):
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for a in state.legal_actions():
        if a < NUM_ACTIONS:
            mask[a] = 1.0
    return mask


def decode_board(dark_hex_state):
    obs = np.array(
        dark_hex_state.observation_tensor(), dtype=np.float32
    ).reshape(BOARD_SIZE, BOARD_SIZE, NUM_OBS_PLANES)
    p0_mask    = obs[:, :, 0:3].sum(axis=2) > 0.5
    p1_mask    = obs[:, :, 4:7].sum(axis=2) > 0.5
    empty_mask = (~p0_mask) & (~p1_mask)
    cell  = lambda r, c: r * BOARD_SIZE + c
    p0    = [cell(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if p0_mask[r, c]]
    p1    = [cell(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if p1_mask[r, c]]
    empty = [cell(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if empty_mask[r, c]]
    return p0, p1, empty


def sample_hex_world(dark_hex_state):
    cur           = dark_hex_state.current_player()
    p0, p1, empty = decode_board(dark_hex_state)
    n_p1_total    = len(p0) if cur == 0 else max(len(p0) - 1, 0)
    n_hidden      = min(max(0, n_p1_total - len(p1)), len(empty))
    hidden_opp    = random.sample(empty, n_hidden) if n_hidden > 0 else []
    full_p0 = list(p0);  full_p1 = list(p1) + hidden_opp
    random.shuffle(full_p0);  random.shuffle(full_p1)
    world   = hex_game.new_initial_state()
    max_len = max(len(full_p0), len(full_p1), 1)
    for i in range(max_len):
        if world.is_terminal(): break
        if world.current_player() == 0 and i < len(full_p0):
            a = full_p0[i]
            if a in world.legal_actions(): world.apply_action(a)
        if world.is_terminal(): break
        if world.current_player() == 1 and i < len(full_p1):
            a = full_p1[i]
            if a in world.legal_actions(): world.apply_action(a)
    return world


def get_model_move(dark_hex_state, model, n_worlds=N_WORLDS, greedy=True):
    """
    Run PC-PIMC and return the chosen action plus the full policy vector.
    greedy=True  → always pick the highest-probability legal action (for eval)
    greedy=False → sample proportionally (for self-play variety)
    """
    az_eval  = AlphaZeroEvaluator(hex_game, model)
    dk_mask  = legal_mask(dark_hex_state)
    policies = []

    for _ in range(n_worlds):
        world = sample_hex_world(dark_hex_state)
        if world.is_terminal():
            continue
        w_mask = legal_mask(world)
        bot    = mcts.MCTSBot(
            hex_game,
            uct_c=UCT_C,
            max_simulations=MCTS_SIMULATIONS,
            evaluator=az_eval,
            solve=False,
            random_state=np.random.RandomState(random.randint(0, 2**31 - 1)),
        )
        root   = bot.mcts_search(world)
        counts = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for child in root.children:
            if child.action < NUM_ACTIONS:
                counts[child.action] = float(child.explore_count)
        counts *= w_mask
        total   = counts.sum()
        if total > 1e-8:
            policies.append(counts / total)

    if not policies:
        legal = dark_hex_state.legal_actions()
        return random.choice(legal), np.ones(NUM_ACTIONS) / NUM_ACTIONS

    mean_pi = np.mean(policies, axis=0) * dk_mask
    total   = mean_pi.sum()
    if total < 1e-8:
        legal = dark_hex_state.legal_actions()
        return random.choice(legal), dk_mask / dk_mask.sum()
    mean_pi /= total

    if greedy:
        action = int(np.argmax(mean_pi))
    else:
        action = int(np.random.choice(NUM_ACTIONS, p=mean_pi))

    return action, mean_pi


def raw_value(state, model):
    obs = get_obs(state)[np.newaxis]
    mask = legal_mask(state)[np.newaxis]
    _, value = model.inference(obs, mask)
    
    # Convert to a flat numpy array
    val_array = np.array(value).flatten()
    
    # DEBUG: Remove this once you see the output
    if val_array.size > 1:
        print(f"DEBUG: Value head returned {val_array.size} numbers: {val_array}")
        
    # Take the first element (usually the win probability for Player 0)
    return float(val_array[0])


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

# Symbols from each player's perspective
P0_SYM = "X"   # P0 connects top–bottom
P1_SYM = "O"   # P1 connects left–right
EMPTY  = "."
HIDDEN = "?"   # opponent stone not yet revealed

def render_board(state, perspective: int):
    """
    Print the board as seen by `perspective` (0 or 1).
    Own stones are shown; opponent stones that haven't been revealed are '?'.
    """
    obs = np.array(state.observation_tensor(), dtype=np.float32
                   ).reshape(BOARD_SIZE, BOARD_SIZE, NUM_OBS_PLANES)

    # From Table 2: planes 0-2 = P0 stones, planes 4-6 = P1 stones
    p0_mask = obs[:, :, 0:3].sum(axis=2) > 0.5
    p1_mask = obs[:, :, 4:7].sum(axis=2) > 0.5

    own_sym = P0_SYM if perspective == 0 else P1_SYM
    opp_sym = P1_SYM if perspective == 0 else P0_SYM

    own_mask = p0_mask if perspective == 0 else p1_mask
    opp_mask = p1_mask if perspective == 0 else p0_mask

    turn_str = f"P{perspective}'s view  ({'X=top↔bot' if perspective==0 else 'O=left↔right'})"
    print(f"\n  {turn_str}")
    print("  ┌" + "───┬" * (BOARD_SIZE - 1) + "───┐")
    for r in range(BOARD_SIZE):
        row_cells = []
        for c in range(BOARD_SIZE):
            if own_mask[r, c]:
                row_cells.append(f" {own_sym} ")
            elif opp_mask[r, c]:
                row_cells.append(f" {opp_sym} ")   # revealed opponent stone
            else:
                row_cells.append(f" {EMPTY} ")
        print("  │" + "│".join(row_cells) + "│")
        if r < BOARD_SIZE - 1:
            print("  ├" + "───┼" * (BOARD_SIZE - 1) + "───┤")
    print("  └" + "───┴" * (BOARD_SIZE - 1) + "───┘")

    # Cell index reference (shown once)
    print("  Cell indices:")
    for r in range(BOARD_SIZE):
        row = "  "
        for c in range(BOARD_SIZE):
            row += f"[{r*BOARD_SIZE+c}]"
        print(row)


def render_policy(policy, legal_actions):
    """Print the model's policy as a heatmap on the board."""
    print("  Model policy (visit %):")
    for r in range(BOARD_SIZE):
        row = "  "
        for c in range(BOARD_SIZE):
            cell = r * BOARD_SIZE + c
            if cell in legal_actions:
                row += f"[{policy[cell]*100:4.1f}]"
            else:
                row += "[ -- ]"
        print(row)


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 1 — HUMAN vs AlphaZe**
# ═══════════════════════════════════════════════════════════════════════════════

def play_human_vs_model(model, human_player=0):
    """
    human_player: 0 → you are P0 (X, top↔bottom)
                  1 → you are P1 (O, left↔right)
    """
    model_player = 1 - human_player
    state = dark_hex_game.new_initial_state()

    print(f"\n{'═'*50}")
    print(f"  Human (P{human_player}) vs AlphaZe** (P{model_player})")
    print(f"  You are {'X (connect top↔bottom)' if human_player==0 else 'O (connect left↔right)'}")
    print(f"{'═'*50}")

    while not state.is_terminal():
        cur = state.current_player()
        render_board(state, cur)

        if cur == human_player:
            # ── Human turn ────────────────────────────────────────────────
            legal = state.legal_actions()
            print(f"\n  Your turn (P{human_player}). Legal cells: {legal}")
            while True:
                try:
                    choice = int(input("  Enter cell number: ").strip())
                    if choice in legal:
                        state.apply_action(choice)
                        break
                    else:
                        print(f"  ✗ {choice} is not a legal move. Try again.")
                except (ValueError, EOFError):
                    print("  ✗ Please enter a number.")
        else:
            # ── Model turn ────────────────────────────────────────────────
            print(f"\n  AlphaZe** (P{model_player}) is thinking …")
            action, policy = get_model_move(state, model, greedy=True)
            render_policy(policy, state.legal_actions())
            val = raw_value(state, model)
            print(f"  → Plays cell {action}   (net value estimate: {val:+.3f})")
            state.apply_action(action)

    # ── Result ────────────────────────────────────────────────────────────
    returns = state.returns()
    print(f"\n{'═'*50}")
    if returns[human_player] > 0:
        print("  🎉  You win!")
    elif returns[model_player] > 0:
        print("  🤖  AlphaZe** wins!")
    else:
        print("  Draw.")
    print(f"{'═'*50}\n")
    return returns


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 2 — AlphaZe** vs Random agent (benchmark)
# ═══════════════════════════════════════════════════════════════════════════════

def random_move(state):
    return random.choice(state.legal_actions())


def eval_vs_random(model, n_games=50, model_plays_as=0):
    """
    Run n_games of AlphaZe**(model_plays_as) vs Random(1-model_plays_as).
    Returns win rate of the model.
    """
    wins = 0
    for g in range(n_games):
        state = dark_hex_game.new_initial_state()
        while not state.is_terminal():
            cur = state.current_player()
            if cur == model_plays_as:
                action, _ = get_model_move(state, model, greedy=True)
            else:
                action = random_move(state)
            state.apply_action(action)
        if state.returns()[model_plays_as] > 0:
            wins += 1
        pct = wins / (g + 1) * 100
        print(f"  Game {g+1:3d}/{n_games}  wins={wins}  win%={pct:.1f}", end="\r")

    print()
    win_rate = wins / n_games
    print(f"\n  AlphaZe** (P{model_plays_as}) vs Random: "
          f"{wins}/{n_games} wins  ({win_rate*100:.1f}%)")
    return win_rate


def eval_both_sides(model, n_games=50):
    """Evaluate as both P0 and P1 to check for colour bias."""
    print(f"\n{'═'*50}")
    print(f"  Evaluation: {n_games} games each side")
    print(f"{'═'*50}")
    print("\n  Playing as P0 (X, top↔bottom) …")
    wr0 = eval_vs_random(model, n_games=n_games, model_plays_as=0)
    print("\n  Playing as P1 (O, left↔right) …")
    wr1 = eval_vs_random(model, n_games=n_games, model_plays_as=1)
    avg = (wr0 + wr1) / 2
    print(f"\n  Average win rate: {avg*100:.1f}%")
    return wr0, wr1


# ═══════════════════════════════════════════════════════════════════════════════
# MODE 3 — AlphaZe** self-play (watch a game)
# ═══════════════════════════════════════════════════════════════════════════════

def watch_self_play(model, pause=True):
    """Play one full game model vs model, rendering every move."""
    state = dark_hex_game.new_initial_state()
    move_num = 0

    print(f"\n{'═'*50}")
    print("  AlphaZe** self-play")
    print(f"{'═'*50}")

    while not state.is_terminal():
        cur = state.current_player()
        render_board(state, cur)
        action, policy = get_model_move(state, model, greedy=False)
        val = raw_value(state, model)
        render_policy(policy, state.legal_actions())
        print(f"\n  Move {move_num+1}: P{cur} plays cell {action}  "
              f"(value estimate: {val:+.3f})")
        state.apply_action(action)
        move_num += 1
        if pause:
            input("  [Enter to continue]")

    returns = state.returns()
    winner  = 0 if returns[0] > 0 else 1
    print(f"\n  Game over after {move_num} moves. P{winner} wins!")
    return returns


# ═══════════════════════════════════════════════════════════════════════════════
# MENU
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_menu(model):
    print("\n" + "═"*50)
    print("  AlphaZe** DarkHex Tester")
    print("═"*50)
    print("  1. Play against AlphaZe** (you are P0 / X)")
    print("  2. Play against AlphaZe** (you are P1 / O)")
    print("  3. Benchmark: AlphaZe** vs Random (50 games each side)")
    print("  4. Watch AlphaZe** self-play")
    print("  5. Quick value probe (enter a board manually)")
    print("  q. Quit")
    print("═"*50)

    while True:
        choice = input("\nChoice: ").strip().lower()
        if choice == "1":
            play_human_vs_model(model, human_player=0)
        elif choice == "2":
            play_human_vs_model(model, human_player=1)
        elif choice == "3":
            n = input("  Games per side [50]: ").strip()
            n = int(n) if n.isdigit() else 50
            eval_both_sides(model, n_games=n)
        elif choice == "4":
            watch_self_play(model, pause=True)
        elif choice == "5":
            value_probe(model)
        elif choice == "q":
            print("Bye!")
            break
        else:
            print("  Unknown option.")


def value_probe(model):
    """
    Show the model's raw policy and value for the initial empty board,
    then let you apply moves and inspect after each one.
    """
    state = dark_hex_game.new_initial_state()
    print("\n  Value probe — starting from empty board.")
    print("  Enter cell numbers one at a time, or 'r' to reset, 'q' to quit.\n")

    while True:
        render_board(state, state.current_player())
        obs  = get_obs(state)[np.newaxis]
        mask = legal_mask(state)[np.newaxis]
        policy_raw, value_raw = model.inference(obs, mask)
        p = policy_raw[0]
        v = float(value_raw[0])
        print(f"  Network value : {v:+.4f}  (positive = good for P0)")
        render_policy(p, state.legal_actions())

        if state.is_terminal():
            print(f"  Terminal — returns: {state.returns()}")
            break

        cmd = input("\n  Next cell (or 'r'=reset, 'q'=quit): ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "r":
            state = dark_hex_game.new_initial_state()
        else:
            try:
                a = int(cmd)
                if a in state.legal_actions():
                    state.apply_action(a)
                else:
                    print(f"  ✗ {a} is not legal here.")
            except ValueError:
                print("  ✗ Enter a number, 'r', or 'q'.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Test the trained AlphaZe** model")
    p.add_argument("--checkpoint", default=CHECKPOINT_PATH,
                   help="Path to checkpoint (default: %(default)s)")
    p.add_argument("--mode", choices=["menu", "eval", "selfplay", "human"],
                   default="menu",
                   help="Run mode (default: interactive menu)")
    p.add_argument("--games", type=int, default=50,
                   help="Games per side for eval mode (default: 50)")
    p.add_argument("--human-side", type=int, choices=[0, 1], default=0,
                   help="Which side you play in human mode (default: 0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = load_model(args.checkpoint)

    if args.mode == "menu":
        interactive_menu(model)
    elif args.mode == "eval":
        eval_both_sides(model, n_games=args.games)
    elif args.mode == "selfplay":
        watch_self_play(model, pause=True)
    elif args.mode == "human":
        play_human_vs_model(model, human_player=args.human_side)