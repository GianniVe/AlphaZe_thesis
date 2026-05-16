"""
compare_algorithms.py
─────────────────────
Head-to-head comparison between the best PC-PIMC and SO-ISMCTS champions.

Each algorithm's best checkpoint is loaded and the two models play:
  - 100 games where PC-PIMC is P0  (connects top↔bottom)
  - 100 games where SO-ISMCTS is P0
  = 200 games total

Both models use PC-PIMC / SO-ISMCTS respectively for move selection
(same algorithm they were trained with, same simulation budget).

Results reported:
  - Win rate per side
  - Overall win rate
  - Per-game log with move count
  - Statistical significance note (binomial test)

Usage
─────
  python3 compare_algorithms.py                        # auto-detect best checkpoints
  python3 compare_algorithms.py --games 100            # games per side (default 100)
  python3 compare_algorithms.py \\
      --ckpt-pc  ./darkhex_alphaze_checkpoints_PC_PIMC/model.ckpt-29 \\
      --ckpt-iso ./darkhex_alphaze_checkpoints_SO_ISMCTS/model.ckpt-29
"""

import os
import sys
import glob
import random
import argparse
import logging
import numpy as np
from scipy import stats

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.algorithms.alpha_zero.evaluator import AlphaZeroEvaluator
from open_spiel.python.algorithms.alpha_zero.model import TrainInput

import tensorflow as tf

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Keras 3 patch (same as training script) ───────────────────────────────────
def _patched_batch_norm(training, updates, name):
    def batch_norm_layer(x):
        bn = tf.keras.layers.BatchNormalization(name=name, trainable=True)
        return bn(x)
    return batch_norm_layer

model_lib.batch_norm = _patched_batch_norm


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — must match trainalphaze.py exactly
# ═══════════════════════════════════════════════════════════════════════════════
BOARD_SIZE     = 5
NUM_CELLS      = BOARD_SIZE * BOARD_SIZE
NUM_OBS_PLANES = 9
MODEL_TYPE     = "resnet"
OBS_SHAPE      = [BOARD_SIZE, BOARD_SIZE, NUM_OBS_PLANES]

SIMULATION_BUDGET = 4000
N_WORLDS          = 5
UCT_C             = 1.5
SELECTION_TEMP    = 0.0

PC_PIMC_CHECKPOINT_DIR  = "/scratch/darkhex_alphaze_checkpoints_PC_PIMC"
SO_ISMCTS_CHECKPOINT_DIR = "/scratch/darkhex_alphaze_checkpoints_SO_ISMCTS"


# ═══════════════════════════════════════════════════════════════════════════════
# GAMES
# ═══════════════════════════════════════════════════════════════════════════════
dark_hex_game = pyspiel.load_game(
    f"dark_hex(num_rows={BOARD_SIZE},num_cols={BOARD_SIZE},gameversion=adh)"
)
hex_game = pyspiel.load_game(
    f"hex(num_rows={BOARD_SIZE},num_cols={BOARD_SIZE})"
)

NUM_ACTIONS = hex_game.num_distinct_actions()


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint(checkpoint_dir):
    pattern     = os.path.join(checkpoint_dir, "model.ckpt-*.index")
    index_files = glob.glob(pattern)
    if not index_files:
        return None, -1
    epochs = []
    for f in index_files:
        try:
            n = int(os.path.basename(f)
                    .replace("model.ckpt-", "").replace(".index", ""))
            epochs.append(n)
        except ValueError:
            continue
    if not epochs:
        return None, -1
    best = max(epochs)
    return os.path.join(checkpoint_dir, f"model.ckpt-{best}"), best


def load_model(checkpoint_path, checkpoint_dir):
    m = model_lib.Model.build_model(
        MODEL_TYPE,
        OBS_SHAPE,
        NUM_ACTIONS,
        nn_width=128,
        nn_depth=6,
        weight_decay=1e-4,
        learning_rate=0.01,
        path=checkpoint_dir,
    )
    m.load_checkpoint(checkpoint_path)
    log.info(f"  Loaded: {checkpoint_path}")
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_obs(state):
    return np.array(state.observation_tensor(), dtype=np.float32)

def legal_mask(state):
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for a in state.legal_actions():
        if a < NUM_ACTIONS:
            mask[a] = 1.0
    return mask

def legal_mask_from_world(world):
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for a in world.legal_actions():
        if a < NUM_ACTIONS:
            mask[a] = 1.0
    return mask

def sample_action_greedy(policy, mask):
    probs = policy * mask
    if probs.sum() < 1e-8:
        return int(np.random.choice(np.where(mask > 0)[0]))
    return int(np.argmax(probs))

def visits_to_policy(root, mask):
    counts = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for child in root.children:
        if child.action < NUM_ACTIONS:
            counts[child.action] = float(child.explore_count)
    counts *= mask
    total = counts.sum()
    if total < 1e-8:
        n = mask.sum()
        return mask / max(n, 1.0)
    return counts / total


# ═══════════════════════════════════════════════════════════════════════════════
# WORLD SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

def decode_board(dark_hex_state):
    obs = np.array(
        dark_hex_state.observation_tensor(), dtype=np.float32
    ).reshape(BOARD_SIZE, BOARD_SIZE, NUM_OBS_PLANES)
    p0_mask    = obs[:, :, 0:3].sum(axis=2) > 0.5
    p1_mask    = obs[:, :, 4:7].sum(axis=2) > 0.5
    empty_mask = (~p0_mask) & (~p1_mask)
    cell  = lambda r, c: r * BOARD_SIZE + c
    p0    = [cell(r, c) for r in range(BOARD_SIZE)
             for c in range(BOARD_SIZE) if p0_mask[r, c]]
    p1    = [cell(r, c) for r in range(BOARD_SIZE)
             for c in range(BOARD_SIZE) if p1_mask[r, c]]
    empty = [cell(r, c) for r in range(BOARD_SIZE)
             for c in range(BOARD_SIZE) if empty_mask[r, c]]
    return p0, p1, empty


def sample_hex_world(dark_hex_state):
    cur           = dark_hex_state.current_player()
    p0, p1, empty = decode_board(dark_hex_state)
    n_p1_total    = len(p0) if cur == 0 else max(len(p0) - 1, 0)
    n_hidden      = min(max(0, n_p1_total - len(p1)), len(empty))
    hidden_opp    = random.sample(empty, n_hidden) if n_hidden > 0 else []
    full_p0 = list(p0)
    full_p1 = list(p1) + hidden_opp
    random.shuffle(full_p0)
    random.shuffle(full_p1)
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


# ═══════════════════════════════════════════════════════════════════════════════
# PC-PIMC MOVE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

def get_pc_pimc_move(dark_hex_state, model):
    az_eval  = AlphaZeroEvaluator(hex_game, model)
    dk_mask  = legal_mask(dark_hex_state)
    policies = []

    for _ in range(N_WORLDS):
        world = sample_hex_world(dark_hex_state)
        if world.is_terminal():
            continue
        w_mask = legal_mask_from_world(world)
        bot    = mcts.MCTSBot(
            hex_game,
            uct_c=UCT_C,
            max_simulations=SIMULATION_BUDGET // N_WORLDS,
            evaluator=az_eval,
            solve=False,
            random_state=np.random.RandomState(random.randint(0, 2**31 - 1)),
        )
        root = bot.mcts_search(world)
        policies.append(visits_to_policy(root, w_mask))

    if not policies:
        return random.choice(dark_hex_state.legal_actions())

    mean_pi = np.mean(policies, axis=0) * dk_mask
    total   = mean_pi.sum()
    if total < 1e-8:
        return random.choice(dark_hex_state.legal_actions())
    mean_pi /= total

    return sample_action_greedy(mean_pi, dk_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# SO-ISMCTS MOVE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

class ISMCTSNode:
    __slots__ = ("action", "parent", "children", "owner",
                 "n_visits", "n_avail", "total_reward", "action_avail")

    def __init__(self, action=None, parent=None, owner=0):
        self.action       = action
        self.parent       = parent
        self.owner        = owner
        self.children     = {}
        self.action_avail = {}
        self.n_visits     = 0
        self.n_avail      = 0
        self.total_reward = 0.0

    def ucb_score(self, c=UCT_C):
        if self.n_visits == 0 or self.n_avail == 0:
            return float("inf")
        return (self.total_reward / self.n_visits
                + c * np.sqrt(np.log(self.n_avail) / self.n_visits))


def _evaluate_leaf(world_state, az_evaluator):
    if world_state.is_terminal():
        return float(world_state.returns()[0])
    _, value = az_evaluator.evaluate(world_state)
    return float(value)


def get_ismcts_move(dark_hex_state, model):
    root    = ISMCTSNode(owner=dark_hex_state.current_player())
    dk_mask = legal_mask(dark_hex_state)
    az_eval = AlphaZeroEvaluator(hex_game, model)

    for _ in range(SIMULATION_BUDGET):
        world = sample_hex_world(dark_hex_state)
        if world.is_terminal():
            continue

        node      = root
        path      = [root]
        sim_world = world

        # Selection
        while not sim_world.is_terminal():
            legal_now      = [a for a in sim_world.legal_actions()
                              if a < NUM_ACTIONS]
            existing_legal = [a for a in legal_now if a in node.children]
            unvisited      = [a for a in legal_now if a not in node.children]

            if unvisited:
                break
            if not existing_legal:
                break

            for a in legal_now:
                node.action_avail[a] = node.action_avail.get(a, 0) + 1
                if a in node.children:
                    node.children[a].n_avail = node.action_avail[a]

            best      = max(existing_legal,
                            key=lambda a: node.children[a].ucb_score(UCT_C))
            node      = node.children[best]
            path.append(node)
            sim_world = sim_world.child(best)

        # Expansion
        if not sim_world.is_terminal():
            legal_now = [a for a in sim_world.legal_actions()
                         if a < NUM_ACTIONS]
            unvisited = [a for a in legal_now if a not in node.children]
            if unvisited:
                new_action = random.choice(unvisited)
                child      = ISMCTSNode(action=new_action, parent=node,
                                        owner=sim_world.current_player())
                child.n_avail = node.action_avail.get(new_action, 1)
                node.children[new_action] = child
                path.append(child)
                sim_world = sim_world.child(new_action)

        # Simulation
        reward = _evaluate_leaf(sim_world, az_eval)

        # Backpropagation
        for visited in path:
            visited.n_visits     += 1
            node_reward           = reward if visited.owner == 0 else -reward
            visited.total_reward += node_reward

    # Extract policy
    counts = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for a, child in root.children.items():
        if a < NUM_ACTIONS and dk_mask[a] > 0:
            counts[a] = float(child.n_visits)

    total = counts.sum()
    if total < 1e-8:
        return random.choice(dark_hex_state.legal_actions())

    pi = counts / total
    return sample_action_greedy(pi, dk_mask)


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE GAME
# ═══════════════════════════════════════════════════════════════════════════════

def play_game(model_p0, algo_p0, model_p1, algo_p1, game_num, verbose=False):
    state     = dark_hex_game.new_initial_state()
    move_count = 0

    while not state.is_terminal():
        cur   = state.current_player()
        m     = model_p0 if cur == 0 else model_p1
        algo  = algo_p0  if cur == 0 else algo_p1

        if algo == "PC_PIMC":
            action = get_pc_pimc_move(state, m)
        else:
            action = get_ismcts_move(state, m)

        if action not in state.legal_actions():
            action = random.choice(state.legal_actions())

        state.apply_action(action)
        move_count += 1

        if move_count > NUM_CELLS * NUM_CELLS:
            log.warning(f"  Game {game_num}: step limit hit — forcing terminal.")
            break

    result  = state.returns() if state.is_terminal() else [0, 0]
    winner  = 0 if result[0] > 0 else 1

    if verbose:
        log.info(f"  Game {game_num:3d}: P{winner} wins in {move_count} moves")

    return winner, move_count


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def run_comparison(pc_model, iso_model, games_per_side=100):
    print("\n" + "═"*60)
    print("  AlphaZe** Algorithm Comparison")
    print("  PC-PIMC  vs  SO-ISMCTS")
    print(f"  {games_per_side} games per side  ({games_per_side*2} total)")
    print("═"*60)

    pc_wins_as_p0   = 0   # PC-PIMC wins when playing as P0
    pc_wins_as_p1   = 0   # PC-PIMC wins when playing as P1
    iso_wins_as_p0  = 0   # SO-ISMCTS wins when playing as P0
    iso_wins_as_p1  = 0   # SO-ISMCTS wins when playing as P1
    move_counts     = []

    # ── Block 1: PC-PIMC as P0, SO-ISMCTS as P1 ──────────────────────────
    print(f"\n  Block 1/2: PC-PIMC (P0) vs SO-ISMCTS (P1)")
    print(f"  {'Game':>5}  {'Winner':>10}  {'Moves':>6}  {'PC-PIMC wins':>13}")
    print("  " + "─"*42)

    for g in range(1, games_per_side + 1):
        winner, moves = play_game(
            pc_model,  "PC_PIMC",
            iso_model, "SO_ISMCTS",
            game_num=g
        )
        move_counts.append(moves)
        if winner == 0:   # P0 = PC-PIMC won
            pc_wins_as_p0 += 1
        else:             # P1 = SO-ISMCTS won
            iso_wins_as_p1 += 1

        if g % 10 == 0:
            pct = pc_wins_as_p0 / g * 100
            print(f"  {g:>5}  {'PC-PIMC' if winner==0 else 'SO-ISMCTS':>10}"
                  f"  {moves:>6}  {pc_wins_as_p0:>5}/{g:<5} ({pct:4.1f}%)")

    # ── Block 2: SO-ISMCTS as P0, PC-PIMC as P1 ──────────────────────────
    print(f"\n  Block 2/2: SO-ISMCTS (P0) vs PC-PIMC (P1)")
    print(f"  {'Game':>5}  {'Winner':>10}  {'Moves':>6}  {'SO-ISMCTS wins':>15}")
    print("  " + "─"*44)

    for g in range(1, games_per_side + 1):
        winner, moves = play_game(
            iso_model, "SO_ISMCTS",
            pc_model,  "PC_PIMC",
            game_num=g
        )
        move_counts.append(moves)
        if winner == 0:   # P0 = SO-ISMCTS won
            iso_wins_as_p0 += 1
        else:             # P1 = PC-PIMC won
            pc_wins_as_p1 += 1

        if g % 10 == 0:
            pct = iso_wins_as_p0 / g * 100
            print(f"  {g:>5}  {'SO-ISMCTS' if winner==0 else 'PC-PIMC':>10}"
                  f"  {moves:>6}  {iso_wins_as_p0:>5}/{g:<5} ({pct:4.1f}%)")

    # ── Results ───────────────────────────────────────────────────────────
    total_games    = games_per_side * 2
    pc_total_wins  = pc_wins_as_p0 + pc_wins_as_p1
    iso_total_wins = iso_wins_as_p0 + iso_wins_as_p1

    pc_win_rate  = pc_total_wins  / total_games
    iso_win_rate = iso_total_wins / total_games

    avg_moves = np.mean(move_counts)
    std_moves = np.std(move_counts)

    best_wins = max(pc_total_wins, iso_total_wins)
    binom_result = stats.binomtest(best_wins, total_games, p=0.5,
                                   alternative="greater")
    p_value = binom_result.pvalue

    print("\n" + "═"*60)
    print("  RESULTS")
    print("═"*60)
    print(f"\n  {'Algorithm':<15} {'As P0':>8} {'As P1':>8} {'Total':>8} {'Win%':>7}")
    print(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*7}")
    print(f"  {'PC-PIMC':<15} "
          f"{pc_wins_as_p0:>5}/{games_per_side:<2} "
          f"{pc_wins_as_p1:>5}/{games_per_side:<2} "
          f"{pc_total_wins:>5}/{total_games:<3} "
          f"{pc_win_rate*100:>6.1f}%")
    print(f"  {'SO-ISMCTS':<15} "
          f"{iso_wins_as_p0:>5}/{games_per_side:<2} "
          f"{iso_wins_as_p1:>5}/{games_per_side:<2} "
          f"{iso_total_wins:>5}/{total_games:<3} "
          f"{iso_win_rate*100:>6.1f}%")

    print(f"\n  Average game length : {avg_moves:.1f} moves  (σ = {std_moves:.1f})")

    winner_name = "PC-PIMC" if pc_total_wins > iso_total_wins else \
                  "SO-ISMCTS" if iso_total_wins > pc_total_wins else "TIED"

    print(f"\n  Overall winner      : {winner_name}")
    print(f"  p-value (binomial)  : {p_value:.4f}")

    if p_value < 0.01:
        sig = "strongly statistically significant (p < 0.01)"
    elif p_value < 0.05:
        sig = "statistically significant (p < 0.05)"
    elif p_value < 0.10:
        sig = "marginally significant (p < 0.10)"
    else:
        sig = "NOT statistically significant (p ≥ 0.10) — results may be due to chance"

    print(f"  Significance        : {sig}")

    # First-mover advantage check
    # In Hex, P0 has a theoretical first-mover advantage.
    # If one algorithm dominates only as P0, that's a bias signal.
    p0_wins_total = pc_wins_as_p0 + iso_wins_as_p0
    p1_wins_total = pc_wins_as_p1 + iso_wins_as_p1
    print(f"\n  First-mover check:")
    print(f"    P0 total wins across all games : {p0_wins_total}/{total_games} "
          f"({p0_wins_total/total_games*100:.1f}%)")
    print(f"    P1 total wins across all games : {p1_wins_total}/{total_games} "
          f"({p1_wins_total/total_games*100:.1f}%)")
    if p0_wins_total / total_games > 0.65:
        print("    ⚠ Strong first-mover bias — P0 wins most games regardless of algorithm.")
        print("      Consider this when interpreting the head-to-head result.")
    else:
        print("    ✓ No strong first-mover bias detected.")

    print("\n" + "═"*60 + "\n")

    return {
        "pc_wins_as_p0"  : pc_wins_as_p0,
        "pc_wins_as_p1"  : pc_wins_as_p1,
        "iso_wins_as_p0" : iso_wins_as_p0,
        "iso_wins_as_p1" : iso_wins_as_p1,
        "pc_win_rate"    : pc_win_rate,
        "iso_win_rate"   : iso_win_rate,
        "p_value"        : p_value,
        "winner"         : winner_name,
        "avg_moves"      : avg_moves,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Compare PC-PIMC vs SO-ISMCTS champions head to head"
    )
    p.add_argument("--ckpt-pc",  default=None,
                   help="PC-PIMC checkpoint path (default: auto-detect latest)")
    p.add_argument("--ckpt-iso", default=None,
                   help="SO-ISMCTS checkpoint path (default: auto-detect latest)")
    p.add_argument("--games", type=int, default=100,
                   help="Games per side (default: 100, total: 200)")
    p.add_argument("--budget", type=int, default=SIMULATION_BUDGET,
                   help=f"Simulation budget per move (default: {SIMULATION_BUDGET})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    SIMULATION_BUDGET = args.budget

    # ── Load PC-PIMC model ────────────────────────────────────────────────
    pc_ckpt = args.ckpt_pc
    if pc_ckpt is None:
        pc_ckpt, pc_epoch = find_latest_checkpoint(PC_PIMC_CHECKPOINT_DIR)
        if pc_ckpt is None:
            print(f"ERROR: No PC-PIMC checkpoint found in {PC_PIMC_CHECKPOINT_DIR}")
            print("       Run trainalphaze.py with MOVE_GEN='PC_PIMC' first.")
            sys.exit(1)
        print(f"PC-PIMC  : auto-detected checkpoint epoch {pc_epoch}")
    else:
        print(f"PC-PIMC  : using {pc_ckpt}")

    print("Loading PC-PIMC model …")
    pc_model = load_model(pc_ckpt, PC_PIMC_CHECKPOINT_DIR)

    # ── Load SO-ISMCTS model ──────────────────────────────────────────────
    iso_ckpt = args.ckpt_iso
    if iso_ckpt is None:
        iso_ckpt, iso_epoch = find_latest_checkpoint(SO_ISMCTS_CHECKPOINT_DIR)
        if iso_ckpt is None:
            print(f"ERROR: No SO-ISMCTS checkpoint found in {SO_ISMCTS_CHECKPOINT_DIR}")
            print("       Run trainalphaze.py with MOVE_GEN='SO_ISMCTS' first.")
            sys.exit(1)
        print(f"SO-ISMCTS: auto-detected checkpoint epoch {iso_epoch}")
    else:
        print(f"SO-ISMCTS: using {iso_ckpt}")

    print("Loading SO-ISMCTS model …")
    iso_model = load_model(iso_ckpt, SO_ISMCTS_CHECKPOINT_DIR)

    # ── Run comparison ────────────────────────────────────────────────────
    results = run_comparison(pc_model, iso_model, games_per_side=args.games)