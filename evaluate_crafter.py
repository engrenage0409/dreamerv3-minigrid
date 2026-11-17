
import argparse
import json
import pathlib

import imageio.v2 as imageio
import numpy as np
import ruamel.yaml as yaml
import torch
from PIL import Image, ImageDraw, ImageFont

import dreamer
import tools


ACTION_NAMES = [
    "noop", "move_left", "move_right", "move_up",
    "move_down", "do", "sleep", "place_stone",
    "place_table", "place_furnace", "place_plant",
    "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword",
    "make_stone_sword", "make_iron_sword",
]


def merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value


def load_config():
    configs = yaml.safe_load(
        pathlib.Path("configs.yaml").read_text(encoding="utf-8")
    )
    merged = {}
    for name in ("defaults", "crafter", "crafter_8gb"):
        merge(merged, configs[name])
    values = {
        key: tools.args_type(value)(value)
        for key, value in merged.items()
    }
    return argparse.Namespace(**values)


def load_font(size):
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSansMono.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_frame(
    observation,
    step,
    action_name,
    reward,
    total_reward,
    inventory,
    achievements,
    font,
    title_font,
):
    # Enlarge the original 64x64 image without blurring.
    image = Image.fromarray(
        np.asarray(observation["image"], dtype=np.uint8)
    ).convert("RGB")

    image = image.resize((512, 512), Image.Resampling.NEAREST)

    canvas = Image.new("RGB", (920, 580), (19, 23, 34))
    canvas.paste(image, (20, 48))

    draw = ImageDraw.Draw(canvas)

    def line(y, text, color=(230, 235, 245)):
        draw.text((555, y), text, font=font, fill=color)

    draw.text(
        (20, 12),
        "DreamerV3 | Crafter Evaluation",
        font=title_font,
        fill=(255, 255, 255),
    )

    line(55, f"Step: {step}")
    line(85, f"Action: {action_name}", (118, 207, 255))
    line(115, f"Reward: {reward:+.2f}")
    line(145, f"Total Reward: {total_reward:+.2f}",
         (120, 235, 160))

    draw.line((555, 185, 890, 185), fill=(65, 75, 95))

    line(200, "SURVIVAL", (255, 210, 110))

    for index, key in enumerate(
        ("health", "food", "drink", "energy")
    ):
        value = inventory.get(key, "N/A")
        line(230 + index * 28, f"{key}: {value}")

    draw.line((555, 355, 890, 355), fill=(65, 75, 95))

    line(370, "ACHIEVEMENTS", (255, 210, 110))

    unlocked = [
        (key, value)
        for key, value in achievements.items()
        if value > 0
    ]

    if not unlocked:
        line(402, "None yet", (165, 175, 190))
    else:
        for index, (key, value) in enumerate(unlocked[:5]):
            name = key.replace("_", " ")
            line(402 + index * 28, f"{name}: {value}")

        if len(unlocked) > 5:
            line(548, f"+ {len(unlocked) - 5} more")

    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="logdir/crafter_smoke_test/latest.pt",
    )
    parser.add_argument(
        "--output",
        default="logdir/crafter_smoke_test/evaluation",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--policy", choices=["mode", "sample", "both"], default="both")
    parser.add_argument("--video", action="store_true", help="Save a dashboard video for the first episode of each policy")
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()

    config = load_config()
    config.envs = 1
    config.parallel = False
    config.action_repeat = 1

    tools.set_seed_everywhere(config.seed)

    output = pathlib.Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    checkpoint_path = pathlib.Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    config.time_limit = min(
        int(config.time_limit), args.max_steps
    )

    env = dreamer.make_env(config, "eval", 0)
    config.num_actions = env.action_space.shape[0]

    train_dir = checkpoint_path.parent / "train_eps"
    episodes = tools.load_episodes(
        train_dir, limit=config.dataset_size
    )

    if not episodes:
        raise RuntimeError(
            f"No training episodes found in {train_dir}"
        )

    dataset = dreamer.make_dataset(episodes, config)
    logger = tools.Logger(output, 0)

    agent = dreamer.Dreamer(
        env.observation_space,
        env.action_space,
        config,
        logger,
        dataset,
    ).to(config.device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=config.device,
        weights_only=False,
    )
    agent.load_state_dict(checkpoint["agent_state_dict"])
    agent.requires_grad_(False)
    agent.eval()

    print("Checkpoint loaded:", checkpoint_path)

    font = load_font(20)
    title_font = load_font(24)


    # Keep evaluation deterministic across policy comparisons, while allowing
    # independent action sampling in sample mode.
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")

    summaries = {}
    for policy_kind in (["mode", "sample"] if args.policy == "both" else [args.policy]):
        episode_results = []
        for episode_index in range(args.episodes):
            # Each policy uses the same environment seed for a given episode.
            # Recreate the environment so resets do not depend on prior episodes.
            if episode_index == 0:
                env.close()
            config.seed = int(load_config().seed) + episode_index
            episode_env = dreamer.make_env(config, "eval", 0)
            obs = episode_env.reset()
            state = None
            done = True
            total_reward = 0.0
            achievements = {}
            inventory = {}
            action_counts = {}
            prob_sums = np.zeros(config.num_actions, dtype=np.float64)
            entropy_sum = 0.0
            steps = 0
            video_path = output / f"crafter_{policy_kind}_episode_001.mp4"
            writer = None
            try:
                if args.video and episode_index == 0:
                    writer = imageio.get_writer(
                        str(video_path), fps=args.fps,
                        codec="libx264", macro_block_size=1,
                    )
                    writer.append_data(draw_frame(
                        obs, 0, "START", 0.0, 0.0,
                        inventory, achievements, font, title_font,
                    ))
                for step in range(args.max_steps):
                    batch = {
                        key: np.expand_dims(np.asarray(value), 0)
                        for key, value in obs.items()
                        if not key.startswith("log_")
                    }
                    with torch.no_grad():
                        # Call _policy once to advance the recurrent state.
                        # It returns mode in evaluation, and we optionally
                        # sample from the SAME actor distribution.
                        policy, next_state = agent(
                            batch, np.array([done], dtype=bool),
                            state, training=False,
                        )
                        latent, _ = next_state
                        feat = agent._wm.dynamics.get_feat(latent)
                        actor = agent._task_behavior.actor(feat)
                        probabilities = actor.probs[0].detach().cpu().numpy()
                        if policy_kind == "sample":
                            chosen = actor.sample()[0].detach()
                        else:
                            chosen = policy["action"][0].detach()
                        action_index = int(torch.argmax(chosen).item())
                        action = torch.nn.functional.one_hot(
                            torch.tensor(action_index, device=chosen.device),
                            num_classes=config.num_actions,
                        ).to(chosen.dtype).cpu().numpy()
                        # Recurrent state must use the action actually executed.
                        state = (latent, torch.as_tensor(
                            action, device=chosen.device,
                        ).unsqueeze(0))
                    prob_sums += probabilities
                    entropy_sum += float(
                        -(probabilities * np.log(
                            np.clip(probabilities, 1e-12, 1.0)
                        )).sum()
                    )
                    action_name = (
                        ACTION_NAMES[action_index]
                        if action_index < len(ACTION_NAMES)
                        else f"action_{action_index}"
                    )
                    obs, reward, done, info = episode_env.step({"action": action})
                    steps += 1
                    total_reward += float(reward)
                    action_counts[action_name] = action_counts.get(action_name, 0) + 1
                    if info:
                        inventory = dict(info.get("inventory", inventory))
                        achievements = dict(info.get("achievements", achievements))
                    if writer is not None:
                        writer.append_data(draw_frame(
                            obs, steps, action_name, float(reward),
                            total_reward, inventory, achievements,
                            font, title_font,
                        ))
                    if done:
                        break
            finally:
                if writer is not None:
                    writer.close()
                episode_env.close()
            item = {
                "episode": episode_index + 1,
                "seed": config.seed,
                "episode_return": total_reward,
                "episode_length": steps,
                "episode_done": bool(done),
                "achievements": achievements,
                "final_inventory": inventory,
                "action_counts": action_counts,
                "mean_action_probabilities": {
                    name: float(prob_sums[i] / max(steps, 1))
                    for i, name in enumerate(ACTION_NAMES[:config.num_actions])
                },
                "mean_actor_entropy_nats": entropy_sum / max(steps, 1),
            }
            if writer is not None:
                item["video"] = str(video_path)
            episode_results.append(item)
            print(
                f"{policy_kind} episode {episode_index + 1}/{args.episodes}: "
                f"return={total_reward:.3f} length={steps} "
                f"entropy={item['mean_actor_entropy_nats']:.3f}"
            )
        returns = np.array([x["episode_return"] for x in episode_results])
        lengths = np.array([x["episode_length"] for x in episode_results])
        achievement_success = {
            name: float(np.mean([
                x["achievements"].get(name, 0) > 0
                for x in episode_results
            ]))
            for name in episode_results[0]["achievements"]
        }
        summaries[policy_kind] = {
            "num_episodes": len(episode_results),
            "mean_return": float(returns.mean()),
            "std_return": float(returns.std()),
            "mean_length": float(lengths.mean()),
            "mean_actor_entropy_nats": float(np.mean([
                x["mean_actor_entropy_nats"] for x in episode_results
            ])),
            "achievement_success_rate": achievement_success,
            "episodes": episode_results,
        }
    result = {
        "checkpoint": str(checkpoint_path),
        "policy": args.policy,
        "max_steps": args.max_steps,
        "summaries": summaries,
    }
    result_path = output / "eval_diagnostics.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\\nDiagnostics saved:", result_path)
    for name, summary in summaries.items():
        print(
            f"{name}: mean_return={summary['mean_return']:.3f} "
            f"std={summary['std_return']:.3f} "
            f"mean_entropy={summary['mean_actor_entropy_nats']:.3f}"
        )


if __name__ == "__main__":
    main()
