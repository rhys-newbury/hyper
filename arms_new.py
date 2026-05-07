import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# ============================================================
#  SAVE DIRECTORIES
# ============================================================

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = Path("results") / RUN_ID
MODEL_DIR = Path("saved_models")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
#  CONFIG
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # General
    # --------------------------------------------------------
    parser.add_argument(
        "--emotion",
        type=str,
        default="happy",
        choices=["happy", "sad", "angry", "calm"],
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--use-lstm", action="store_true")

    # --------------------------------------------------------
    # PPO
    # --------------------------------------------------------
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------
    parser.add_argument("--lstm-hidden-size", type=int, default=64)
    parser.add_argument("--n-lstm-layers", type=int, default=1)

    # --------------------------------------------------------
    # Reward weights
    # --------------------------------------------------------
    parser.add_argument("--shoulder-band-width", type=float, default=0.12)
    parser.add_argument("--elbow-band-width", type=float, default=0.60)

    parser.add_argument("--shoulder-band-weight", type=float, default=0.4)
    parser.add_argument("--elbow-band-weight", type=float, default=0.6)

    parser.add_argument("--band-reward-weight", type=float, default=1.5)
    parser.add_argument("--motion-reward-weight", type=float, default=0.5)
    parser.add_argument("--action-penalty-weight", type=float, default=0.01)

    return parser


def dump_config(cfg: argparse.Namespace) -> None:
    path = RESULTS_DIR / "config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(vars(cfg).copy(), f, indent=2, sort_keys=True)
    print(f"Saved config: {path}")


# ============================================================
#  ENVIRONMENT
# ============================================================
class ExpressiveWavingArmEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        cfg: argparse.Namespace,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.cfg = cfg
        self.emotion = cfg.emotion
        self.render_mode = render_mode

        self.dt = 0.03  # simulation time step
        self.max_steps = 400  # 400*0.03 = 12s

        # Link lengths
        self.l1 = 120
        self.l2 = 100
        self.joint_mins = np.array([0.15, 0.40], dtype=np.float32)
        self.joint_maxs = np.array([0.55, 1.80], dtype=np.float32)

        self.omega_phase = 2 * np.pi * 0.8

        self.shoulder_center = 0.35
        self.elbow_center = 1.10
        self.shoulder_nom_amp = 0.08
        self.elbow_nom_amp = 0.45
        self.theta_center = np.array(
            [self.shoulder_center, self.elbow_center], dtype=np.float32
        )

        # Observation: [theta1, theta2, theta_dot1, theta_dot2]
        self.observation_space = spaces.Box(
            low=np.array([0.15, 0.40, -8.0, -8.0], dtype=np.float32),
            high=np.array([0.55, 1.80, 8.0, 8.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.screen = None
        self.clock = None
        self.origin = np.array([350, 280], dtype=np.float32)

        self.steps = 0
        self.phase = 0.0
        self.theta = np.zeros(2, dtype=np.float32)
        self.prev_theta = np.zeros(2, dtype=np.float32)
        self.theta_dot = np.zeros(2, dtype=np.float32)
        self.prev_theta_dot = np.zeros(2, dtype=np.float32)
        self.prev_action = np.zeros(2, dtype=np.float32)

        self.episode_rewards = []
        self.phase_errors = []
        self.amplitudes = []
        self.speeds = []
        self.jerks = []
        self.actions = []

        self.theta_history = []
        self.nominal_history = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.steps = 0
        self.phase = 0.0

        self.theta = self._nominal_theta(self.phase).copy()
        self.prev_theta = self.theta.copy()
        self.theta_dot = self._nominal_theta_dot(self.phase).copy()
        self.prev_theta_dot = self.theta_dot.copy()
        self.prev_action = np.zeros(2, dtype=np.float32)

        self.episode_rewards = []
        self.phase_errors = []
        self.amplitudes = []
        self.speeds = []
        self.jerks = []
        self.actions = []

        self.theta_history = []
        self.nominal_history = []

        return self._get_obs(), {}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        self.prev_theta = self.theta.copy()
        self.prev_theta_dot = self.theta_dot.copy()

        action_scale = 3.0

        # 1. Transition updates
        self.theta_dot += action * action_scale * self.dt
        self.theta += self.theta_dot * self.dt
        self.theta = np.clip(self.theta, self.joint_mins, self.joint_maxs)
        self.phase += self.omega_phase * self.dt
        self.phase = self.phase % (2.0 * np.pi)

        reward, terms = self._compute_reward(action)

        self.episode_rewards.append(reward)
        self.phase_errors.append(terms["phase_error"])
        self.amplitudes.append(terms["amplitude"])
        self.speeds.append(terms["speed"])
        self.jerks.append(terms["jerk"])
        self.actions.append(float(np.linalg.norm(action)))

        self.theta_history.append(self.theta.copy())
        self.nominal_history.append(self._nominal_theta(self.phase).copy())

        self.prev_action = action.copy()

        self.steps += 1
        terminated = self.steps >= self.max_steps
        truncated = False

        return self._get_obs(), reward, terminated, truncated, terms

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self.theta[0],
                self.theta[1],
                self.theta_dot[0],
                self.theta_dot[1],
            ],
            dtype=np.float32,
        )

    def _nominal_theta(self, phase: float) -> np.ndarray:
        # Nominal wave
        #   self.shoulder_center = 0.35
        #   self.elbow_center = 1.10
        #   self.shoulder_nom_amp = 0.08
        #   self.elbow_nom_amp = 0.45
        theta1 = self.shoulder_center + self.shoulder_nom_amp * np.sin(phase)
        theta2 = self.elbow_center + self.elbow_nom_amp * np.sin(2 * phase)
        return np.array([theta1, theta2], dtype=np.float32)

    def _nominal_theta_dot(self, phase: float) -> np.ndarray:
        theta1_dot = self.shoulder_nom_amp * np.cos(phase) * self.omega_phase
        theta2_dot = 2.0 * self.elbow_nom_amp * np.cos(2.0 * phase) * self.omega_phase
        return np.array([theta1_dot, theta2_dot], dtype=np.float32)

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        theta = self.theta
        theta_dot = self.theta_dot

        theta_nom = self._nominal_theta(self.phase)
        theta_nom_dot = self._nominal_theta_dot(self.phase)

        x1 = self.cfg.shoulder_band_width
        x2 = self.cfg.elbow_band_width

        # Per-joint errors
        shoulder_error = abs(theta[0] - theta_nom[0])
        elbow_error = abs(theta[1] - theta_nom[1])

        shoulder_norm_error = shoulder_error / self.cfg.shoulder_band_width
        elbow_norm_error = elbow_error / self.cfg.elbow_band_width
        
        # Smooth Gaussian falloff — no cliff, no unbounded negatives
        r_shoulder_band = 1.0 - shoulder_norm_error**2
        r_elbow_band = 1.0 - elbow_norm_error**2

        r_band = (
            self.cfg.shoulder_band_weight * r_shoulder_band
            + self.cfg.elbow_band_weight * r_elbow_band
        )

        # =====================================================
        # MOTION REWARD — penalise freezing at center
        # =====================================================
        nom_speed = np.linalg.norm(theta_nom_dot)
        actual_speed = np.linalg.norm(theta_dot)
        r_motion = np.exp(-((actual_speed - nom_speed) ** 2) / (nom_speed**2 + 1e-6))

        # =====================================================
        # ACTION PENALTY — discourage large/jerky actions
        # =====================================================
        r_action = -self.cfg.action_penalty_weight * np.linalg.norm(action) ** 2

        # =====================================================
        # TOTAL
        # =====================================================        
        reward = (
            self.cfg.band_reward_weight * r_band
            + self.cfg.motion_reward_weight * r_motion
            + r_action
        )
        # Diagnostics
        structure_error = np.linalg.norm(theta - theta_nom)
        jerk = np.linalg.norm(theta_dot - self.prev_theta_dot)

        terms = {
            "reward": float(reward),
            "r_band": float(r_band),
            "r_shoulder_band": float(r_shoulder_band),
            "r_elbow_band": float(r_elbow_band),
            "shoulder_error": float(shoulder_error),
            "elbow_error": float(elbow_error),
            "structure_error": float(structure_error),
            "theta_dot_nom_1": float(theta_nom_dot[0]),
            "theta_dot_nom_2": float(theta_nom_dot[1]),
            "theta_dot_1": float(theta_dot[0]),
            "theta_dot_2": float(theta_dot[1]),
            "phase_error": float(structure_error),
            "amplitude": float(abs(theta[1] - self.elbow_center)),
            "speed": float(actual_speed),
            "jerk": float(jerk),
        }

        return float(reward), terms

    def get_episode_metrics(self) -> Dict[str, float]:
        return {
            "avg_reward": float(np.mean(self.episode_rewards)),
            "avg_phase_error": float(np.mean(self.phase_errors)),
            "avg_amplitude": float(np.mean(self.amplitudes)),
            "avg_speed": float(np.mean(self.speeds)),
            "avg_jerk": float(np.mean(self.jerks)),
            "episode_length": float(self.steps),
        }

    def render(self) -> None:
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((750, 600))
            pygame.display.set_caption(f"Expressive Waving Arm - {self.emotion}")
            self.clock = pygame.time.Clock()

        if self.clock is not None:
            self.clock.tick(60)

        self.screen.fill((255, 255, 255))

        nominal_theta = self._nominal_theta(self.phase)

        self._draw_arm(nominal_theta, color=(160, 160, 160), width=4)
        self._draw_arm(self.theta, color=(30, 80, 220), width=8)

        font = pygame.font.SysFont(None, 24)

        lines = [
            f"Emotion: {self.emotion}",
            "Grey = nominal wave",
            "Blue = learned motion",
            f"Amplitude: {np.linalg.norm(self.theta - self.theta_center):.3f}",
            f"Speed: {np.linalg.norm(self.theta_dot):.3f}",
            f"Phase error: {np.linalg.norm(self.theta - nominal_theta):.3f}",
        ]

        y = 20
        for line in lines:
            text = font.render(line, True, (0, 0, 0))
            self.screen.blit(text, (20, y))
            y += 26

        pygame.display.flip()

    def _draw_arm(
        self,
        theta: np.ndarray,
        color: Tuple[int, int, int],
        width: int,
    ) -> None:
        theta1, theta2 = theta

        p0 = self.origin

        p1 = np.array(
            [
                p0[0] + self.l1 * np.cos(theta1),
                p0[1] + self.l1 * np.sin(theta1),
            ],
            dtype=np.float32,
        )

        p2 = np.array(
            [
                p1[0] + self.l2 * np.cos(theta1 + theta2),
                p1[1] + self.l2 * np.sin(theta1 + theta2),
            ],
            dtype=np.float32,
        )

        pygame.draw.line(self.screen, color, p0.astype(int), p1.astype(int), width)
        pygame.draw.line(self.screen, color, p1.astype(int), p2.astype(int), width)

        pygame.draw.circle(self.screen, (0, 0, 0), p0.astype(int), 7)
        pygame.draw.circle(self.screen, (0, 0, 0), p1.astype(int), 6)
        pygame.draw.circle(self.screen, (0, 0, 0), p2.astype(int), 6)


def get_model(cfg: argparse.Namespace, env: DummyVecEnv) -> PPO:

    if cfg.use_lstm:
        return RecurrentPPO(
            "MlpLstmPolicy",
            env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            gamma=cfg.gamma,
            verbose=1,
            policy_kwargs={
                "lstm_hidden_size": cfg.lstm_hidden_size,
                "n_lstm_layers": cfg.n_lstm_layers,
            },
        )

    return PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        verbose=1,
    )


def load_model(cfg: argparse.Namespace, model_file: Path) -> Any:
    if cfg.use_lstm:
        return RecurrentPPO.load(str(model_file))

    return PPO.load(str(model_file))


def model_stem(cfg: argparse.Namespace) -> str:
    algo_name = "lstm" if cfg.use_lstm else "ppo"
    return f"{cfg.emotion}_{algo_name}_waving_arm"


# ============================================================
#  TRAINING
# ============================================================
def train_emotion(
    cfg: argparse.Namespace,
) -> Tuple[Any, List[float], List[int]]:
    env = DummyVecEnv([lambda: Monitor(ExpressiveWavingArmEnv(cfg=cfg))])

    model = get_model(cfg, env)
    model.learn(total_timesteps=cfg.timesteps)

    model_path = MODEL_DIR / model_stem(cfg)
    model.save(str(model_path))

    print(f"Saved model: {model_path}.zip")

    episode_rewards = env.envs[0].get_episode_rewards()
    episode_lengths = env.envs[0].get_episode_lengths()

    return model, episode_rewards, episode_lengths


# ============================================================
#  PLOTS WITH SAVING LOGIC
# ============================================================
def save_current_plot(filename: str) -> None:
    path = RESULTS_DIR / filename
    plt.savefig(str(path), dpi=300, bbox_inches="tight")
    print(f"Saved plot: {path}")


def plot_training_rewards(
    episode_rewards: List[float],
    emotion: str,
    save: bool = True,
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title(f"Training Reward Curve - {emotion}")
    plt.grid(True)

    if save:
        save_current_plot(f"{emotion}_training_rewards.png")

    plt.show()


def plot_evaluation(
    env: ExpressiveWavingArmEnv,
    cfg: argparse.Namespace,
    save: bool = True,
) -> None:
    theta_history = np.array(env.theta_history)
    nominal_history = np.array(env.nominal_history)
    t = np.arange(len(theta_history)) * env.dt

    # Joint angles
    plt.figure(figsize=(10, 5))

    shoulder_lower = nominal_history[:, 0] - cfg.shoulder_band_width
    shoulder_upper = nominal_history[:, 0] + cfg.shoulder_band_width

    elbow_lower = nominal_history[:, 1] - cfg.elbow_band_width
    elbow_upper = nominal_history[:, 1] + cfg.elbow_band_width

    # Light grey shaded regions
    plt.fill_between(
        t,
        shoulder_lower,
        shoulder_upper,
        color="lightgrey",
        alpha=0.5,
        label="Shoulder band",
    )

    plt.fill_between(
        t,
        elbow_lower,
        elbow_upper,
        color="lightgrey",
        alpha=0.3,
        label="Elbow band",
    )

    plt.plot(t, theta_history[:, 0], label="Learned shoulder")
    plt.plot(t, nominal_history[:, 0], "--", label="Nominal shoulder")

    plt.plot(t, theta_history[:, 1], label="Learned elbow")
    plt.plot(t, nominal_history[:, 1], "--", label="Nominal elbow")

    plt.xlabel("Time (s)")
    plt.ylabel("Joint angle (rad)")
    plt.title(f"Joint Angles - {cfg.emotion}")
    plt.legend()
    plt.grid(True)

    if save:
        save_current_plot(f"{cfg.emotion}_joint_angles.png")

    plt.show()

    # Style metrics
    plt.figure(figsize=(10, 5))
    plt.plot(t, env.amplitudes, label="Amplitude")
    plt.plot(t, env.speeds, label="Speed")
    plt.plot(t, env.jerks, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Metric value")
    plt.title(f"Style Metrics - {cfg.emotion}")
    plt.legend()
    plt.grid(True)

    if save:
        save_current_plot(f"{cfg.emotion}_style_metrics.png")

    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(t, env.phase_errors, label="Structure error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error")
    plt.title(f"Structure Error - {cfg.emotion}")
    plt.legend()
    plt.grid(True)

    if save:
        save_current_plot(f"{cfg.emotion}_structure_error.png")

    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(t, env.episode_rewards, label="Reward")
    plt.xlabel("Time (s)")
    plt.ylabel("Reward")
    plt.title(f"Evaluation Reward Over Time - {cfg.emotion}")
    plt.legend()
    plt.grid(True)

    if save:
        save_current_plot(f"{cfg.emotion}_evaluation_reward.png")

    plt.show()


# ============================================================
#  VISUALIZATION
# ============================================================
def visualize(cfg: argparse.Namespace, model: Any) -> None:
    env = ExpressiveWavingArmEnv(cfg=cfg, render_mode="human")
    obs, _ = env.reset()

    pygame.init()

    running = True
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    while running:
        if cfg.use_lstm:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_done = terminated or truncated
        episode_start = np.array([episode_done], dtype=bool)

        env.render()

        if env.screen is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        if episode_done:
            metrics = env.get_episode_metrics()

            print("\nEvaluation metrics:")
            for key, value in metrics.items():
                print(f"{key}: {value:.4f}")

            running = False

    pygame.quit()
    plot_evaluation(env, cfg, save=True)


# ============================================================
#  MAIN
# ============================================================
def main() -> None:
    cfg = build_parser().parse_args()
    dump_config(cfg)

    model_file = MODEL_DIR / f"{model_stem(cfg)}.zip"

    if model_file.exists():
        print("Loading existing model...")
        model = load_model(cfg, model_file)
    else:
        print("Training new model...")
        model, episode_rewards, episode_lengths = train_emotion(cfg=cfg)
        plot_training_rewards(episode_rewards, cfg.emotion, save=True)

    visualize(cfg=cfg, model=model)

    print(f"\nAll plots saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
