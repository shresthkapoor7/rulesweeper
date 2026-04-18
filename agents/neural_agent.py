from __future__ import annotations
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

from agents import Agent
from config import GameConfig
from game import MinesweeperGame, GameState

MAX_BOARD = 30
NUM_CHANNELS = 4
CONTEXT_DIM = 9
NUM_ACTIONS = 2  # reveal, flag

Transition = namedtuple(
    "Transition",
    ["board", "context", "action", "reward", "next_board", "next_context", "next_mask", "done"],
)


# ── State encoding ──────────────────────────────────────────────────

def encode_state(game: MinesweeperGame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode the game state for the neural network.

    Returns:
        board_tensor:  (4, MAX_BOARD, MAX_BOARD) float32
        context:       (CONTEXT_DIM,) float32
        action_mask:   (2, MAX_BOARD, MAX_BOARD) bool — True = valid action
    """
    cfg = game.get_config()
    rows, cols = cfg.rows, cfg.cols
    metrics = game.get_player_metrics()

    # Board channels
    board = torch.zeros(NUM_CHANNELS, MAX_BOARD, MAX_BOARD)
    for r in range(rows):
        for c in range(cols):
            cell = game.get_cell(r, c)
            if cell.is_revealed:
                board[0, r, c] = 1.0
                board[1, r, c] = cell.adjacent_mines / 8.0
            if cell.is_flagged:
                board[2, r, c] = 1.0
            board[3, r, c] = 1.0  # valid mask

    # Context vector
    safe_cells = max(rows * cols - cfg.mine_count, 1)
    context = torch.tensor([
        rows / MAX_BOARD,
        cols / MAX_BOARD,
        cfg.mine_count / max(rows * cols, 1),
        cfg.starting_health / 10.0,
        cfg.mine_damage / 5.0,
        float(cfg.safe_first_click),
        float(cfg.reveal_strategy == "cascade"),
        metrics["health_remaining"] / max(cfg.starting_health, 1),
        metrics["cells_revealed"] / safe_cells,
    ], dtype=torch.float32)

    # Action mask: (2, MAX_BOARD, MAX_BOARD), True = valid
    action_mask = torch.zeros(NUM_ACTIONS, MAX_BOARD, MAX_BOARD, dtype=torch.bool)
    flags_placed = sum(
        1 for r in range(rows) for c in range(cols)
        if game.get_cell(r, c).is_flagged
    )
    can_flag = cfg.flag_limit is None or flags_placed < cfg.flag_limit
    for r in range(rows):
        for c in range(cols):
            cell = game.get_cell(r, c)
            if not cell.is_revealed and not cell.is_flagged:
                action_mask[0, r, c] = True   # reveal
                if can_flag:
                    action_mask[1, r, c] = True  # flag

    return board, context, action_mask


def decode_action(flat_idx: int) -> tuple[str, int, int]:
    """Convert flat action index to (action_str, row, col)."""
    action_type = flat_idx // (MAX_BOARD * MAX_BOARD)
    spatial = flat_idx % (MAX_BOARD * MAX_BOARD)
    row = spatial // MAX_BOARD
    col = spatial % MAX_BOARD
    return ("reveal" if action_type == 0 else "flag", row, col)


# ── CNN Model ───────────────────────────────────────────────────────

class MinesweeperDQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_front = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.conv_back = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
        )
        # FiLM: project context to per-channel scale and shift for 128-ch features
        self.film_proj = nn.Linear(CONTEXT_DIM, 128 * 2)
        # Per-cell Q-values directly from conv features
        self.q_head = nn.Conv2d(64, NUM_ACTIONS, 1)

    def forward(self, board: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            board:   (batch, 4, 30, 30)
            context: (batch, 9)
        Returns:
            q_values: (batch, 2, 30, 30)
        """
        x = self.conv_front(board)                          # (batch, 128, 30, 30)
        # FiLM conditioning
        film = self.film_proj(context)                      # (batch, 256)
        gamma = film[:, :128].unsqueeze(-1).unsqueeze(-1) + 1.0  # residual init
        beta = film[:, 128:].unsqueeze(-1).unsqueeze(-1)
        x = gamma * x + beta
        x = self.conv_back(x)                               # (batch, 64, 30, 30)
        return self.q_head(x)                               # (batch, 2, 30, 30)


# ── Replay Buffer ───────────────────────────────────────────────────

class ReplayBuffer:
    """Uniform experience replay buffer."""

    def __init__(self, capacity: int = 100_000) -> None:
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def add(self, t: Transition) -> None:
        self._buf.append(t)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


# ── Reward computation ──────────────────────────────────────────────

def compute_reward(result: dict, cfg: GameConfig) -> float:
    """Compute scalar reward from a game action result."""
    state = result["state"]
    if state == "won":
        return 1.0
    if state == "lost":
        return -1.0
    if result["hit_mine"]:
        return -0.3
    n = len(result["newly_revealed"])
    safe_cells = cfg.rows * cfg.cols - cfg.mine_count
    return n / max(safe_cells, 1)


# ── NeuralAgent (inference) ─────────────────────────────────────────

class NeuralAgent(Agent):
    """Plays minesweeper using a pretrained DQN model."""

    def __init__(self, seed: int | None = None, model_path: str = "checkpoints/best.pt") -> None:
        self._rng = random.Random(seed)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = MinesweeperDQN().to(self._device)
        checkpoint = torch.load(model_path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(checkpoint["online_net"])
        self._model.eval()

    def choose_action(self, game: MinesweeperGame) -> tuple[str, int, int]:
        board_t, ctx_t, mask = encode_state(game)
        board_t = board_t.unsqueeze(0).to(self._device)
        ctx_t = ctx_t.unsqueeze(0).to(self._device)

        with torch.no_grad():
            q = self._model(board_t, ctx_t).squeeze(0)  # (2, 30, 30)

        q[~mask] = float("-inf")
        flat_idx = q.reshape(-1).argmax().item()
        return decode_action(flat_idx)


# ── Training logic ──────────────────────────────────────────────────

class Trainer:
    """DQN trainer for MinesweeperDQN."""

    def __init__(
        self,
        lr: float = 1e-4,
        gamma: float = 0.95,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 100_000,
        min_buffer: int = 1_000,
        tau: float = 0.005,
        device: str | None = None,
    ) -> None:
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MinesweeperDQN().to(self.device)
        self.target_net = MinesweeperDQN().to(self.device)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.min_buffer = min_buffer
        self.tau = tau

        self.step_count = 0
        self.episode_count = 0

    @property
    def epsilon(self) -> float:
        frac = min(self.step_count / max(self.eps_decay_steps, 1), 1.0)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def select_action(
        self, board_t: torch.Tensor, ctx_t: torch.Tensor, mask: torch.Tensor
    ) -> int:
        """Epsilon-greedy action selection. Returns flat action index."""
        if random.random() < self.epsilon:
            valid = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            return valid[random.randint(0, len(valid) - 1)].item()

        self.model.eval()
        with torch.no_grad():
            q = self.model(
                board_t.unsqueeze(0).to(self.device),
                ctx_t.unsqueeze(0).to(self.device),
            ).squeeze(0)
        self.model.train()

        q[~mask] = float("-inf")
        return q.reshape(-1).argmax().item()

    def train_step(self) -> float | None:
        """Sample a batch from the buffer and do one gradient step. Returns loss."""
        if len(self.buffer) < self.min_buffer:
            return None

        batch = self.buffer.sample(self.batch_size)
        boards = torch.stack([t.board for t in batch]).to(self.device)
        ctxs = torch.stack([t.context for t in batch]).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_boards = torch.stack([t.next_board for t in batch]).to(self.device)
        next_ctxs = torch.stack([t.next_context for t in batch]).to(self.device)
        next_masks = torch.stack([t.next_mask for t in batch]).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)

        # Current Q-values for chosen actions
        q_all = self.model(boards, ctxs)                 # (batch, 2, 30, 30)
        q_flat = q_all.reshape(self.batch_size, -1)       # (batch, 2*30*30)
        q_values = q_flat.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values — Double DQN: online net selects action, target net evaluates
        with torch.no_grad():
            # Action selection: online network
            next_q_online = self.model(next_boards, next_ctxs)
            next_q_online[~next_masks] = float("-inf")
            next_actions = next_q_online.reshape(self.batch_size, -1).argmax(dim=1)

            # Action evaluation: target network
            next_q_target = self.target_net(next_boards, next_ctxs)
            next_q_target[~next_masks] = float("-inf")
            next_q_flat = next_q_target.reshape(self.batch_size, -1)
            next_max = next_q_flat.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Replace any non-finite value with 0 so the Bellman target stays well-defined.
            next_max = torch.where(torch.isfinite(next_max), next_max, torch.zeros_like(next_max))
            targets = rewards + self.gamma * next_max * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)
        if not torch.isfinite(loss):
            return None  # skip corrupted batch rather than propagating NaN weights

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._soft_update_target()

        return loss.item()

    def _soft_update_target(self) -> None:
        for p, tp in zip(self.model.parameters(), self.target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def run_episode(self, config: GameConfig, seed: int | None = None) -> dict:
        """Play one full game, collecting transitions. Returns episode stats."""
        game = MinesweeperGame(config, seed=seed)
        total_reward = 0.0
        losses: list[float] = []

        board_t, ctx_t, mask = encode_state(game)

        while game.get_state() in (GameState.PENDING, GameState.ACTIVE):
            if not mask.any():
                break
            flat_action = self.select_action(board_t, ctx_t, mask)
            action_str, row, col = decode_action(flat_action)

            if action_str == "reveal":
                result = game.reveal(row, col)
            else:
                result = game.flag(row, col)

            reward = compute_reward(result, config)
            total_reward += reward

            next_board_t, next_ctx_t, next_mask = encode_state(game)
            done = game.get_state() in (GameState.WON, GameState.LOST)

            self.buffer.add(Transition(
                board=board_t, context=ctx_t, action=flat_action,
                reward=reward, next_board=next_board_t,
                next_context=next_ctx_t, next_mask=next_mask, done=done,
            ))

            loss = self.train_step()
            if loss is not None:
                losses.append(loss)

            board_t, ctx_t, mask = next_board_t, next_ctx_t, next_mask
            self.step_count += 1

        self.episode_count += 1
        metrics = game.get_player_metrics()
        return {
            "state": game.get_state().value,
            "total_reward": total_reward,
            "avg_loss": sum(losses) / len(losses) if losses else 0.0,
            "steps": self.step_count,
            "epsilon": self.epsilon,
            **metrics,
        }

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "online_net": self.model.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "episode_count": self.episode_count,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = ckpt["step_count"]
        self.episode_count = ckpt["episode_count"]
