"""CLI training script for the neural minesweeper agent."""
from __future__ import annotations
import argparse
import os

import torch

from config import GameConfig
from game import MinesweeperGame, GameState
from agents.neural_agent import Trainer, encode_state, decode_action


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the neural minesweeper agent")
    p.add_argument("--episodes", type=int, default=50_000, help="Total training episodes")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    p.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    p.add_argument("--buffer-size", type=int, default=100_000, help="Replay buffer capacity")
    p.add_argument("--eps-decay", type=int, default=100_000, help="Epsilon decay steps")
    p.add_argument("--tau", type=float, default=0.005, help="Polyak averaging coefficient for target network")
    p.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    p.add_argument("--checkpoint-freq", type=int, default=1000, help="Save every N episodes")
    p.add_argument("--eval-freq", type=int, default=500, help="Evaluate every N episodes")
    p.add_argument("--eval-games", type=int, default=20, help="Games per evaluation")
    p.add_argument("--resume", default=None, help="Resume from checkpoint path")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu or cuda (auto-detect if omitted)")
    # Board config for training
    p.add_argument("--rows", type=int, default=9, help="Board rows for training")
    p.add_argument("--cols", type=int, default=9, help="Board cols for training")
    p.add_argument("--mines", type=int, default=10, help="Mine count for training")
    return p.parse_args()


def evaluate(trainer: Trainer, config: GameConfig, n_games: int) -> dict:
    """Run n_games with greedy policy and return aggregated stats."""
    trainer.model.eval()
    wins = 0
    total_revealed = 0
    safe_cells = config.rows * config.cols - config.mine_count

    for i in range(n_games):
        game = MinesweeperGame(config, seed=i)
        board_t, ctx_t, mask = encode_state(game)

        while game.get_state() in (GameState.PENDING, GameState.ACTIVE):
            if not mask.any():
                break
            with torch.no_grad():
                q = trainer.model(
                    board_t.unsqueeze(0).to(trainer.device),
                    ctx_t.unsqueeze(0).to(trainer.device),
                ).squeeze(0)
            q[~mask] = float("-inf")
            flat_idx = q.reshape(-1).argmax().item()
            action_str, row, col = decode_action(flat_idx)
            if action_str == "reveal":
                game.reveal(row, col)
            else:
                game.flag(row, col)
            board_t, ctx_t, mask = encode_state(game)

        if game.get_state() == GameState.WON:
            wins += 1
        total_revealed += game.get_player_metrics()["cells_revealed"]

    trainer.model.train()
    return {
        "win_rate": wins / n_games,
        "avg_progress": (total_revealed / n_games) / max(safe_cells, 1),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    config = GameConfig(rows=args.rows, cols=args.cols, mine_count=args.mines, flag_limit=0)

    trainer = Trainer(
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        eps_decay_steps=args.eps_decay,
        tau=args.tau,
        device=args.device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume} (episode {trainer.episode_count})")

    best_win_rate = -1.0

    for ep in range(1, args.episodes + 1):
        seed = (args.seed + ep) if args.seed is not None else None
        stats = trainer.run_episode(config, seed=seed)

        if ep % 50 == 0:
            print(
                f"Ep {ep:>6d} | "
                f"reward {stats['total_reward']:>7.1f} | "
                f"loss {stats['avg_loss']:.4f} | "
                f"eps {stats['epsilon']:.3f} | "
                f"state {stats['state']}"
            )

        if ep % args.eval_freq == 0:
            ev = evaluate(trainer, config, args.eval_games)
            print(f"  EVAL   | win_rate {ev['win_rate']*100:.1f}% | progress {ev['avg_progress']*100:.1f}%")
            if ev["win_rate"] > best_win_rate:
                best_win_rate = ev["win_rate"]
                best_path = os.path.join(args.checkpoint_dir, "best.pt")
                trainer.save_checkpoint(best_path)
                print(f"  NEW BEST → {best_path}")

        if ep % args.checkpoint_freq == 0:
            path = os.path.join(args.checkpoint_dir, f"ep_{ep}.pt")
            trainer.save_checkpoint(path)

    # Final save
    final_path = os.path.join(args.checkpoint_dir, "final.pt")
    trainer.save_checkpoint(final_path)
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
