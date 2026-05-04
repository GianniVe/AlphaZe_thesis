import os
import sys
import glob
import random
import logging
import numpy as np

# ── JAX / backend ────────────────────────────────────────────────────────────
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import pyspiel
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.algorithms.alpha_zero.model import TrainInput
from open_spiel.python.algorithms.alpha_zero.evaluator import AlphaZeroEvaluator


logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

import tensorflow as tf

def _patched_batch_norm(training, updates, name):
    def batch_norm_layer(x):
        bn = tf.keras.layers.BatchNormalization(name=name, trainable=True)
        return bn(x)
    return batch_norm_layer

model_lib.batch_norm = _patched_batch_norm


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
BOARD_SIZE     = 3
NUM_CELLS      = BOARD_SIZE * BOARD_SIZE
NUM_OBS_PLANES = 9

NUM_EPOCHS        = 50
TSL_EPOCHS        = 5
SAMPLES_PER_EPOCH = 100
N_WORLDS          = 10
MCTS_SIMULATIONS  = 800
UCT_C             = 1.5
SELECTION_TEMP    = 1.0

MODEL_TYPE     = "resnet"
CHECKPOINT_DIR = "./darkhex_alphaze_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

FORCE_FRESH_START = False


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT RESUME HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, "model.ckpt-*.index")
    index_files = glob.glob(pattern)

    if not index_files:
        return None, -1

    # Extract epoch numbers and find the maximum
    epochs = []
    for f in index_files:
        basename = os.path.basename(f)
        try:
            epoch_num = int(basename.replace("model.ckpt-", "").replace(".index", ""))
            epochs.append(epoch_num)
        except ValueError:
            continue

    if not epochs:
        return None, -1

    latest_epoch = max(epochs)
    ckpt_path    = os.path.join(checkpoint_dir, f"model.ckpt-{latest_epoch}")
    return ckpt_path, latest_epoch


# ═══════════════════════════════════════════════════════════════════════════════
# GAMES
# ═══════════════════════════════════════════════════════════════════════════════
dark_hex_game = pyspiel.load_game(
    f"dark_hex(num_rows={BOARD_SIZE},num_cols={BOARD_SIZE})"
)
hex_game = pyspiel.load_game(
    f"hex(num_rows={BOARD_SIZE},num_cols={BOARD_SIZE})"
)

assert dark_hex_game.num_distinct_actions() == hex_game.num_distinct_actions()
NUM_ACTIONS    = hex_game.num_distinct_actions()
_FLAT_OBS_SIZE = int(np.prod(dark_hex_game.observation_tensor_shape()))

if MODEL_TYPE in ("resnet", "conv2d"):
    OBS_SHAPE = [BOARD_SIZE, BOARD_SIZE, NUM_OBS_PLANES]
else:
    OBS_SHAPE = [_FLAT_OBS_SIZE]

log.info(f"Architecture : {MODEL_TYPE}")
log.info(f"OBS_SHAPE    : {OBS_SHAPE}")
log.info(f"NUM_ACTIONS  : {NUM_ACTIONS}")


def get_obs(state):
    return np.array(state.observation_tensor(), dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def legal_mask(state):
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for a in state.legal_actions():
        if a < NUM_ACTIONS:
            mask[a] = 1.0
    return mask


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


def sample_action(policy, mask, temp=SELECTION_TEMP):
    probs = policy * mask
    if probs.sum() < 1e-8:
        return int(np.random.choice(np.where(mask > 0)[0]))
    if temp < 1e-3:
        return int(np.argmax(probs))
    log_p = np.log(probs + 1e-30) / temp
    log_p -= log_p.max()
    probs_t = np.exp(log_p) * mask
    probs_t /= probs_t.sum()
    return int(np.random.choice(NUM_ACTIONS, p=probs_t))


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
# PC-PIMC
# ═══════════════════════════════════════════════════════════════════════════════

def get_pc_pimc_move(dark_hex_state, model, n_worlds=N_WORLDS):
    az_eval  = AlphaZeroEvaluator(hex_game, model)
    dk_mask  = legal_mask(dark_hex_state)
    policies = []
    for _ in range(n_worlds):
        world = sample_hex_world(dark_hex_state)
        if world.is_terminal(): continue
        w_mask = legal_mask(world)
        bot    = mcts.MCTSBot(
            hex_game, uct_c=UCT_C, max_simulations=MCTS_SIMULATIONS,
            evaluator=az_eval, solve=False,
            random_state=np.random.RandomState(random.randint(0, 2**31 - 1)),
        )
        root = bot.mcts_search(world)
        policies.append(visits_to_policy(root, w_mask))
    if not policies:
        n = dk_mask.sum()
        fb = dk_mask / max(n, 1.0)
        return sample_action(fb, dk_mask), fb
    mean_pi = np.mean(policies, axis=0) * dk_mask
    total   = mean_pi.sum()
    if total < 1e-8:
        n = dk_mask.sum()
        mean_pi = dk_mask / max(n, 1.0)
    else:
        mean_pi /= total
    return sample_action(mean_pi, dk_mask, temp=SELECTION_TEMP), mean_pi


# ═══════════════════════════════════════════════════════════════════════════════
# TRUESIGHT LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

def reconstruct_true_world(dark_hex_state):
    try:
        history = dark_hex_state.history()
    except Exception:
        return None
    world = hex_game.new_initial_state()
    for a in history:
        if world.is_terminal(): break
        if a in world.legal_actions(): world.apply_action(a)
    return world


def get_tsl_move(dark_hex_state, model):
    true_world = reconstruct_true_world(dark_hex_state)
    if true_world is None or true_world.is_terminal():
        return get_pc_pimc_move(dark_hex_state, model, n_worlds=1)
    dk_mask = legal_mask(dark_hex_state)
    w_mask  = legal_mask(true_world)
    az_eval = AlphaZeroEvaluator(hex_game, model)
    bot  = mcts.MCTSBot(hex_game, uct_c=UCT_C, max_simulations=MCTS_SIMULATIONS,
                        evaluator=az_eval, solve=False)
    root = bot.mcts_search(true_world)
    pi   = visits_to_policy(root, w_mask) * dk_mask
    total = pi.sum()
    if total < 1e-8:
        n = dk_mask.sum()
        pi = dk_mask / max(n, 1.0)
    else:
        pi /= total
    return sample_action(pi, dk_mask, temp=SELECTION_TEMP), pi


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

model = model_lib.Model.build_model(
    MODEL_TYPE,
    OBS_SHAPE,
    NUM_ACTIONS,
    nn_width=64,
    nn_depth=4,
    weight_decay=1e-4,
    learning_rate=5e-4,
    path=CHECKPOINT_DIR,
)
log.info(f"Model built ({MODEL_TYPE}).")

# ── Resume logic ──────────────────────────────────────────────────────────────
if FORCE_FRESH_START:
    START_EPOCH = 0
    log.info("FORCE_FRESH_START=True — starting from epoch 0.")
else:
    latest_ckpt, last_epoch = find_latest_checkpoint(CHECKPOINT_DIR)

    if latest_ckpt is not None:
        log.info(f"Found checkpoint: {latest_ckpt}  (epoch {last_epoch})")
        model.load_checkpoint(latest_ckpt)
        START_EPOCH = last_epoch + 1
        log.info(f"Weights restored. Resuming from epoch {START_EPOCH}.")
    else:
        START_EPOCH = 0
        log.info("No checkpoint found — starting from epoch 0.")

if START_EPOCH >= NUM_EPOCHS:
    log.info(f"Already completed all {NUM_EPOCHS} epochs. Nothing to do.")
    log.info("Increase NUM_EPOCHS to continue training.")
    sys.exit(0)

log.info(f"Training epochs {START_EPOCH} → {NUM_EPOCHS - 1}.")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

for epoch in range(START_EPOCH, NUM_EPOCHS):
    # TSL is used for the first TSL_EPOCHS epochs regardless of resume point
    use_tsl = (epoch < TSL_EPOCHS)
    mode    = "TSL" if use_tsl else "PC-PIMC"
    log.info(f"\n{'='*60}")
    log.info(f"Epoch {epoch:3d}/{NUM_EPOCHS}  [{mode}]")
    log.info(f"{'='*60}")

    train_inputs = []

    for game_idx in range(SAMPLES_PER_EPOCH):
        state   = dark_hex_game.new_initial_state()
        history = []
        step    = 0

        while not state.is_terminal():
            obs  = get_obs(state)
            mask = legal_mask(state)

            action, pi = get_tsl_move(state, model) if use_tsl \
                    else get_pc_pimc_move(state, model)

            assert np.isfinite(pi).all(),             "Non-finite policy"
            assert abs(pi.sum() - 1.0) < 1e-4,       f"Policy sum={pi.sum():.6f}"
            assert (pi * (1.0 - mask)).sum() < 1e-6, "Policy mass on illegal action"

            history.append({
                "obs": obs, "mask": mask,
                "pi": pi,   "player": state.current_player(),
            })

            if action not in state.legal_actions():
                action = random.choice(state.legal_actions())
            state.apply_action(action)
            step += 1
            if step > NUM_CELLS * 3:
                log.warning("Step limit — truncating.")
                break

        z = float(state.returns()[0]) if state.is_terminal() else 0.0

        for e in history:
            v = z if e["player"] == 0 else -z
            train_inputs.append(TrainInput(
                observation=e["obs"],
                legals_mask=e["mask"],
                policy=e["pi"],
                value=v,
            ))

        if (game_idx + 1) % 5 == 0:
            log.info(f"  game {game_idx+1:3d}/{SAMPLES_PER_EPOCH} | "
                     f"z={z:+.1f} | dataset: {len(train_inputs)}")

    if not train_inputs:
        log.warning("Empty dataset — skipping.")
        continue

    pi_arr  = np.array([t.policy for t in train_inputs])
    entropy = (-pi_arr * np.log(pi_arr + 1e-30)).sum(axis=1).mean()
    log.info(f"  Policy mean entropy : {entropy:.4f}  "
             f"(max uniform = {np.log(NUM_ACTIONS):.2f})")

    log.info(f"  Training on {len(train_inputs)} triplets …")
    losses = model.update(train_inputs)
    log.info(f"  Loss: {losses}")

    try:
        ckpt = os.path.join(CHECKPOINT_DIR, "model.ckpt")
        model._saver.save(model._session, ckpt, global_step=epoch)
        log.info(f"  ✓ Saved: {ckpt}-{epoch}")
    except Exception as e:
        log.warning(f"  _saver failed ({e}), trying model.save() …")
        try:
            model.save(CHECKPOINT_DIR)
            log.info("  ✓ Saved via model.save()")
        except Exception as e2:
            log.error(f"  Could not save: {e2}")

log.info("\nTraining complete.")