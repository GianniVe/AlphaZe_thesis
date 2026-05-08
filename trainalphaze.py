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
# ▶▶▶  ALGORITHM SELECTION  ◀◀◀
# ═══════════════════════════════════════════════════════════════════════════════
MOVE_GEN = "PC_PIMC"       # "PC_PIMC"  |  "SO_ISMCTS"
# ═══════════════════════════════════════════════════════════════════════════════

assert MOVE_GEN in ("PC_PIMC", "SO_ISMCTS"), \
    f"MOVE_GEN must be 'PC_PIMC' or 'SO_ISMCTS', got '{MOVE_GEN}'"


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
BOARD_SIZE     = 3
NUM_CELLS      = BOARD_SIZE * BOARD_SIZE
NUM_OBS_PLANES = 9

NUM_EPOCHS        = 50
TSL_DECAY_EPOCHS  = 20
SAMPLES_PER_EPOCH = 100
SIMULATION_BUDGET = 800
N_WORLDS          = 10
UCT_C             = 1.5
SELECTION_TEMP    = 1.0

# ── Replay buffer ─────────────────────────────────────────────────────────────
BUFFER_MAX_SIZE         = 500_000
TRAIN_SAMPLES_PER_EPOCH = 50_000

# ── Batch size ──────────────────────────────────
BATCH_SIZE = 512

# ── Overfitting / early stopping ──────────────────────────────────────────────
VALIDATION_FRACTION = 0.10
OVERFIT_RATIO       = 1.25
OVERFIT_PATIENCE    = 5

# ── Generational self-play ────────────────────────────────────────────────────
EVAL_GAMES    = 20
WIN_THRESHOLD = 0.55

MODEL_TYPE = "resnet"

CHECKPOINT_DIR = f"./darkhex_alphaze_checkpoints_{MOVE_GEN}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

FORCE_FRESH_START = False


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT RESUME HELPER  (your original, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, "model.ckpt-*.index")
    index_files = glob.glob(pattern)

    if not index_files:
        return None, -1

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


def save_checkpoint(model, epoch):
    try:
        ckpt = os.path.join(CHECKPOINT_DIR, "model.ckpt")
        model._saver.save(model._session, ckpt, global_step=epoch)
        log.info(f"  ✓ Saved: {ckpt}-{epoch}")
        return True
    except Exception as e:
        log.warning(f"  _saver failed ({e}), trying model.save() …")
        try:
            model.save(CHECKPOINT_DIR)
            log.info("  ✓ Saved via model.save()")
            return True
        except Exception as e2:
            log.error(f"  Could not save: {e2}")
            return False


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

log.info(f"Algorithm    : {MOVE_GEN}")
log.info(f"Architecture : {MODEL_TYPE}")
log.info(f"OBS_SHAPE    : {OBS_SHAPE}")
log.info(f"NUM_ACTIONS  : {NUM_ACTIONS}")
log.info(f"CHECKPOINT   : {CHECKPOINT_DIR}")


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
            hex_game, uct_c=UCT_C, max_simulations=SIMULATION_BUDGET // N_WORLDS,
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
# SO-ISMCTS
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


def get_ismcts_move(dark_hex_state, model,
                    simulations=SIMULATION_BUDGET, c=UCT_C):
    root    = ISMCTSNode(owner=dark_hex_state.current_player())
    dk_mask = legal_mask(dark_hex_state)
    az_eval = AlphaZeroEvaluator(hex_game, model)

    for _ in range(simulations):
        world = sample_hex_world(dark_hex_state)
        if world.is_terminal():
            continue

        node      = root
        path      = [root]
        sim_world = world

        # ── Selection ─────────────────────────────────────────────────────
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

            best      = max(existing_legal, key=lambda a: node.children[a].ucb_score(c))
            node      = node.children[best]
            path.append(node)
            sim_world = sim_world.child(best)

        # ── Expansion ─────────────────────────────────────────────────────
        if not sim_world.is_terminal():
            legal_now = [a for a in sim_world.legal_actions() if a < NUM_ACTIONS]
            unvisited = [a for a in legal_now if a not in node.children]
            if unvisited:
                new_action = random.choice(unvisited)
                child      = ISMCTSNode(action=new_action, parent=node, owner=sim_world.current_player())
                child.n_avail = node.action_avail.get(new_action, 1)
                node.children[new_action] = child
                path.append(child)
                sim_world = sim_world.child(new_action)

        reward = _evaluate_leaf(sim_world, az_eval)

        # ── Backpropagation ───────────────────────────────────────────────
        for visited in path:
            visited.n_visits += 1
            node_reward = reward if visited.owner == 0 else -reward
            visited.total_reward += node_reward

    # ── Extract policy ────────────────────────────────────────────────────
    counts = np.zeros(NUM_ACTIONS, dtype=np.float32)
    for a, child in root.children.items():
        if a < NUM_ACTIONS and dk_mask[a] > 0:
            counts[a] = float(child.n_visits)

    total = counts.sum()
    if total < 1e-8:
        n  = dk_mask.sum()
        pi = dk_mask / max(n, 1.0)
        return random.choice(dark_hex_state.legal_actions()), pi

    pi     = counts / total
    action = sample_action(pi, dk_mask, temp=SELECTION_TEMP)
    return action, pi


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
        if MOVE_GEN == "PC_PIMC":
            return get_pc_pimc_move(dark_hex_state, model, n_worlds=1)
        else:
            return get_ismcts_move(dark_hex_state, model, simulations=100)
    dk_mask = legal_mask(dark_hex_state)
    w_mask  = legal_mask(true_world)
    az_eval = AlphaZeroEvaluator(hex_game, model)
    bot  = mcts.MCTSBot(hex_game, uct_c=UCT_C, max_simulations=SIMULATION_BUDGET,
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
# MOVE DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

def get_move(dark_hex_state, model, use_tsl=False):
    if use_tsl:
        return get_tsl_move(dark_hex_state, model)
    if MOVE_GEN == "PC_PIMC":
        return get_pc_pimc_move(dark_hex_state, model)
    else:
        return get_ismcts_move(dark_hex_state, model)


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATIONAL SELF-PLAY
# ═══════════════════════════════════════════════════════════════════════════════

def play_one_eval_game(model_p0, model_p1):
    state = dark_hex_game.new_initial_state()
    while not state.is_terminal():
        cur   = state.current_player()
        m     = model_p0 if cur == 0 else model_p1
        obs   = get_obs(state)[np.newaxis]
        mask  = legal_mask(state)[np.newaxis]
        try:
            policy, _ = m.inference(obs, mask)
            masked    = policy[0] * legal_mask(state)
            action    = int(np.argmax(masked))
        except Exception:
            action = random.choice(state.legal_actions())
        if action not in state.legal_actions():
            action = random.choice(state.legal_actions())
        state.apply_action(action)
    return int(state.returns()[0])


def evaluate_vs_champion(new_model, champion_model, n_games=EVAL_GAMES):
    new_wins = 0
    for g in range(n_games):
        if g % 2 == 0:
            result = play_one_eval_game(new_model, champion_model)
            if result > 0:
                new_wins += 1
        else:
            result = play_one_eval_game(champion_model, new_model)
            if result < 0:   # new_model was P1 and won
                new_wins += 1
    wr = new_wins / n_games
    log.info(f"  New model win rate vs champion: {wr*100:.1f}%  "
             f"(need ≥ {WIN_THRESHOLD*100:.0f}%)")
    return wr


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION CROSS-ENTROPY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_val_ce(model, val_inputs):
    if not val_inputs:
        return None
    obs_arr = np.array([t.observation for t in val_inputs], dtype=np.float32)
    msk_arr = np.array([t.legals_mask for t in val_inputs], dtype=np.float32)
    tgt_arr = np.array([t.policy      for t in val_inputs], dtype=np.float32)
    try:
        policy_batch, _ = model.inference(obs_arr, msk_arr)
        ce = -(tgt_arr * np.log(policy_batch + 1e-30)).sum(axis=1).mean()
        return float(ce)
    except Exception as e:
        log.warning(f"  Val CE failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(checkpoint_path=None):
    m = model_lib.Model.build_model(
        MODEL_TYPE,
        OBS_SHAPE,
        NUM_ACTIONS,
        nn_width=64,
        nn_depth=4,
        weight_decay=1e-4,
        learning_rate=5e-4,
        path=CHECKPOINT_DIR,
    )
    if checkpoint_path is not None:
        m.load_checkpoint(checkpoint_path)
        log.info(f"  Loaded: {checkpoint_path}")
    return m


# ── Resume logic ─────────────────────────────
if FORCE_FRESH_START:
    champion_model = build_model()
    START_EPOCH    = 0
    log.info("FORCE_FRESH_START=True — starting from epoch 0.")
else:
    latest_ckpt, last_epoch = find_latest_checkpoint(CHECKPOINT_DIR)

    if latest_ckpt is not None:
        log.info(f"Found checkpoint: {latest_ckpt}  (epoch {last_epoch})")
        champion_model = build_model(latest_ckpt)
        START_EPOCH    = last_epoch + 1
        log.info(f"Weights restored. Resuming from epoch {START_EPOCH}.")
    else:
        champion_model = build_model()
        START_EPOCH    = 0
        log.info("No checkpoint found — starting from epoch 0.")

if START_EPOCH >= NUM_EPOCHS:
    log.info(f"Already completed all {NUM_EPOCHS} epochs. Nothing to do.")
    log.info("Increase NUM_EPOCHS to continue training.")
    sys.exit(0)

log.info(f"Training epochs {START_EPOCH} → {NUM_EPOCHS - 1}.")
log.info(f"Replay buffer max       : {BUFFER_MAX_SIZE:,}")
log.info(f"Batch size              : {BATCH_SIZE}")
log.info(f"Generational eval games : {EVAL_GAMES}  "
         f"(win threshold {WIN_THRESHOLD*100:.0f}%)")
log.info(f"TSL decay over          : {TSL_DECAY_EPOCHS} epochs")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

replay_buffer = []
overfit_count = 0

for epoch in range(START_EPOCH, NUM_EPOCHS):
    tsl_fraction = max(0.0, 1.0 - epoch / TSL_DECAY_EPOCHS)
    mode_str     = (f"TSL {tsl_fraction*100:.0f}% / "
                    f"{MOVE_GEN} {(1-tsl_fraction)*100:.0f}%")

    log.info(f"\n{'='*60}")
    log.info(f"Epoch {epoch:3d}/{NUM_EPOCHS}  [{mode_str}]")
    log.info(f"  Replay buffer: {len(replay_buffer):,} entries")
    log.info(f"{'='*60}")

    latest_ckpt, _ = find_latest_checkpoint(CHECKPOINT_DIR)
    new_model       = build_model(latest_ckpt)

    new_inputs = []

    for game_idx in range(SAMPLES_PER_EPOCH):
        state   = dark_hex_game.new_initial_state()
        history = []
        step    = 0
        while not state.is_terminal():
            use_tsl = (random.random() < tsl_fraction)
            obs  = get_obs(state)
            mask = legal_mask(state)

            action, pi = get_move(state, new_model, use_tsl=use_tsl)

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
            new_inputs.append(TrainInput(
                observation=e["obs"],
                legals_mask=e["mask"],
                policy=e["pi"],
                value=v,
            ))

        if (game_idx + 1) % 5 == 0:
            log.info(f"  game {game_idx+1:3d}/{SAMPLES_PER_EPOCH} | "
                     f"z={z:+.1f} | new this epoch: {len(new_inputs)}")

    # ── Extend replay buffer, evict oldest ───────────────────────────────
    replay_buffer.extend(new_inputs)
    if len(replay_buffer) > BUFFER_MAX_SIZE:
        replay_buffer = replay_buffer[-BUFFER_MAX_SIZE:]

    if not replay_buffer:
        log.warning("Empty replay buffer — skipping.")
        continue

    # ── Train / validation split ──────────────────────────────────────────
    n_val      = max(1, int(len(replay_buffer) * VALIDATION_FRACTION))
    val_idx    = set(random.sample(range(len(replay_buffer)), n_val))
    val_inputs = [replay_buffer[i] for i in val_idx]
    pool       = [replay_buffer[i] for i in range(len(replay_buffer))
                  if i not in val_idx]

    n_train      = min(len(pool), TRAIN_SAMPLES_PER_EPOCH)
    train_inputs = random.sample(pool, n_train)
    random.shuffle(train_inputs)

    # ── Diagnostics ───────────────────────────────────────────────────────
    pi_arr  = np.array([t.policy for t in train_inputs])
    entropy = (-pi_arr * np.log(pi_arr + 1e-30)).sum(axis=1).mean()
    log.info(f"  Policy mean entropy : {entropy:.4f}  "
             f"(uniform = {np.log(NUM_ACTIONS):.2f}  — watch for collapse if << 1.0)")

    # ── Mini-batch gradient updates ───────────────────────────────────────
    log.info(f"  Training {n_train:,} samples  (BATCH_SIZE={BATCH_SIZE}) …")
    batch_losses = []
    for i in range(0, len(train_inputs), BATCH_SIZE):
        batch = train_inputs[i : i + BATCH_SIZE]
        if not batch:
            break
        loss = new_model.update(batch)
        loss_val = loss.total if hasattr(loss, "total") else float(loss)
        batch_losses.append(loss_val)

    mean_train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
    log.info(f"  Train loss (mean over batches) : {mean_train_loss:.4f}")

    # ── Validation cross-entropy ──────────────────────────────────────────
    val_ce = compute_val_ce(new_model, val_inputs)
    if val_ce is not None:
        log.info(f"  Val cross-entropy              : {val_ce:.4f}")
        if val_ce > mean_train_loss * OVERFIT_RATIO:
            overfit_count += 1
            log.warning(f"  Overfit signal "
                        f"({overfit_count}/{OVERFIT_PATIENCE}): "
                        f"val {val_ce:.4f} > "
                        f"train {mean_train_loss:.4f} × {OVERFIT_RATIO}")
            if overfit_count >= OVERFIT_PATIENCE:
                log.warning("  Early stopping triggered — keeping champion.")
                break
        else:
            if overfit_count > 0:
                log.info(f"  Overfit signal cleared (was {overfit_count}).")
            overfit_count = 0

    # ── Generational self-play: does new model beat champion? ────────────
    log.info(f"  Evaluating new model vs champion ({EVAL_GAMES} games) …")
    win_rate = evaluate_vs_champion(new_model, champion_model)

    if win_rate >= WIN_THRESHOLD:
        log.info(f"  New model promoted  "
                 f"(win rate {win_rate*100:.1f}% ≥ {WIN_THRESHOLD*100:.0f}%)")
        champion_model = new_model
        save_checkpoint(champion_model, epoch)
    else:
        log.info(f"  Champion retained  "
                 f"(new model only {win_rate*100:.1f}%)")

log.info("\nTraining complete.")
best_ckpt, best_epoch = find_latest_checkpoint(CHECKPOINT_DIR)
log.info(f"Best checkpoint: {best_ckpt}  (epoch {best_epoch})")