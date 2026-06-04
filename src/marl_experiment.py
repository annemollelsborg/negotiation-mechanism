import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray.*")
warnings.filterwarnings("ignore", message=".*Logger.*deprecated.*", category=DeprecationWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SUBMODULE_SRC = PROJECT_ROOT / "ciceroscm-surrogate" / "src"
_CICEROSCM_PKG = PROJECT_ROOT / "ciceroscm-surrogate" / "ciceroscm" / "src"
for _p in [str(_CICEROSCM_PKG), str(_SUBMODULE_SRC), str(PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.marl_env import ClimateMARL  # noqa: E402
from src.utils.marl_utils import load_marl_setup, _to_jsonable  # noqa: E402
from src.utils.config_utils import load_yaml_config


class ClimateMarlExperiment:
    def __init__(self, env_config, emission_data, economics_config, actions_config, training_cfg):
        self.env_config = env_config
        self.emission_data = emission_data
        self.economics_config = economics_config
        self.actions_config = actions_config
        self.training_cfg = training_cfg
        self.lever_names = list(actions_config["lever_names"])
        self.lever_count = len(self.lever_names)
        self.adaptation_idx = self.lever_count  # adaptation is at index lever_count in action vector
        N = env_config["N"]
        # full action: levers + adaptation + 2*N proposals + N evaluations
        self.action_dim = self.lever_count + 1 + 3 * N

    def rollout_fixed_actions(self, seed=None):
        env = ClimateMARL(
            self.env_config,
            self.emission_data,
            self.economics_config,
            self.actions_config,
        )

        obs, _ = env.reset(seed=seed) if seed is not None else env.reset()

        done = False
        t = 0
        trajectory = []
        per_agent_return = {f"country_{i}": 0.0 for i in range(self.env_config["N"])}
        total_return = 0.0
        info_dict = {}

        base_action = np.zeros(self.action_dim, dtype=np.int64)
        if "energy" in self.lever_names:
            energy_idx = self.lever_names.index("energy")
            energy_levels = self.actions_config["lever_levels"]["energy"]
            energy_choice = 2 if len(energy_levels) > 1 else 0
            energy_choice = min(energy_choice, len(energy_levels) - 1)
            base_action[energy_idx] = energy_choice

        while not done:
            actions = {agent_id: base_action.copy() for agent_id in obs.keys()}
            step_log = {agent_id: base_action.tolist() for agent_id in obs.keys()}

            trajectory.append(step_log)
            obs, rewards, terminated, truncated, info_dict = env.step(actions)

            for aid, r in rewards.items():
                per_agent_return[aid] += float(r)
                total_return += float(r)

            done = terminated.get("__all__", False) or truncated.get("__all__", False)
            t += 1

        return trajectory, per_agent_return, total_return, t, obs, info_dict

    def rollout_greedy_actions(self, algo, policy_mapping_fn, seed=None):
        env = ClimateMARL(
            self.env_config,
            self.emission_data,
            self.economics_config,
            self.actions_config,
        )

        obs, _ = env.reset(seed=seed) if seed is not None else env.reset()

        done = False
        t = 0
        trajectory = []
        per_agent_return          = {f"country_{i}": 0.0 for i in range(self.env_config["N"])}

        # ms: accumulate reward components
        per_agent_climate_reward   = {f"country_{i}": 0.0 for i in range(self.env_config["N"])}
        per_agent_tariff_cost = {f"country_{i}": 0.0 for i in range(self.env_config["N"])}
        per_agent_noncompliance_cost = {f"country_{i}": 0.0 for i in range(self.env_config["N"])}

        rnn_state = {}
        for agent_id in obs.keys():
            pol_id = policy_mapping_fn(agent_id)
            pol = algo.get_policy(pol_id)
            rnn_state[agent_id] = pol.get_initial_state()

        while not done:
            actions, step_log = {}, {}

            for agent_id, agent_obs in obs.items():
                pol_id = policy_mapping_fn(agent_id)
                pol = algo.get_policy(pol_id)

                act, new_state, _ = pol.compute_single_action(
                    agent_obs,
                    state=rnn_state[agent_id],
                    explore=False,
                )
                rnn_state[agent_id] = new_state

                actions[agent_id] = np.asarray(act, dtype=np.int64)
                step_log[agent_id] = np.asarray(act).tolist()

            trajectory.append(step_log)
            obs, rewards, terminated, truncated, info_dict = env.step(actions)

            for aid, r in rewards.items():
                per_agent_return[aid] += float(r)
                info = info_dict.get(aid, {})
                per_agent_climate_reward[aid]     += info.get("climate_reward", 0.0)
                per_agent_tariff_cost[aid]        += info.get("tariff_cost", 0.0)
                per_agent_noncompliance_cost[aid] += info.get("noncompliance_cost", 0.0)

            done = terminated.get("__all__", False) or truncated.get("__all__", False)
            t += 1

        return trajectory, per_agent_return, t, obs, info_dict, per_agent_climate_reward, per_agent_tariff_cost, per_agent_noncompliance_cost

    def print_greedy_summary(self, greedy_traj):
        if not greedy_traj:
            return {}

        lever_levels = {
            name: np.asarray(self.actions_config["lever_levels"][name], dtype=float)
            for name in self.lever_names
        }
        adaptation_levels = np.asarray(
            self.actions_config["adaptation_levels"], dtype=float
        )

        agents = [f"country_{i}" for i in range(self.env_config["N"])]
        N = self.env_config["N"]

        # Cycle order in trajectory: stage 0=climate, 1=proposal, 2=eval.
        # The first climate step has no preceding negotiation (min_reduction=0),
        # so proposal_traj/eval_traj are each one entry shorter than climate_traj.
        # Prepend None so negotiation[j] aligns with climate_traj[j] for all j.
        climate_traj  = [step for i, step in enumerate(greedy_traj) if i % 3 == 0]
        proposal_traj = [None] + [step for i, step in enumerate(greedy_traj) if i % 3 == 1]
        eval_traj     = [None] + [step for i, step in enumerate(greedy_traj) if i % 3 == 2]

        T = len(climate_traj)
        start_year = int(self.env_config["hist_end"]) + 1
        years = list(range(start_year, start_year + T))

        # Action-vector slice boundaries (matching ms_marl_env.py)
        proposal_start = self.lever_count + 1           # after levers + adaptation
        proposal_end   = proposal_start + 2 * N         # 2N proposal values

        print(f"\nGreedy policy choices over {T} years ({years[0]}–{years[-1]}):")

        policy_logger = {}
        for ag in agents:
            idx_matrix = np.array(
                [np.asarray(step[ag], dtype=int) for step in climate_traj], dtype=int
            )

            lever_series = {}
            for j, name in enumerate(self.lever_names):
                levels = lever_levels[name]
                selections = np.clip(idx_matrix[:, j], 0, len(levels) - 1)
                efforts = levels[selections]
                lever_series[name] = efforts.tolist()

            adapt_idx = np.clip(idx_matrix[:, self.adaptation_idx], 0, len(adaptation_levels) - 1)
            adapt_levels = adaptation_levels[adapt_idx]

            # Negotiation: proposals (promises + requests) and evaluations.
            # None sentinel marks year 0 where no negotiation preceded the climate step.
            promises_series = []
            requests_series = []
            for step in proposal_traj:
                if step is None:
                    promises_series.append(None)
                    requests_series.append(None)
                else:
                    action = np.asarray(step[ag], dtype=int)
                    prop = action[proposal_start:proposal_end]
                    promises_series.append(prop[:N].tolist())   # what I promise to each agent
                    requests_series.append(prop[N:].tolist())   # what I request from each agent

            accepts_series = []
            for step in eval_traj:
                if step is None:
                    accepts_series.append(None)
                else:
                    action = np.asarray(step[ag], dtype=int)
                    accepts_series.append(action[proposal_end:].tolist())  # 0/1 per agent

            print(f"- {ag}:")
            for name in self.lever_names:
                series = lever_series[name]
                print(
                    f"  {name} effort (fraction): " + ", ".join(f"{v:.2f}" for v in series)
                )
            print(
                "  adaptation investment (fraction): "
                + ", ".join(f"{v:.2f}" for v in adapt_levels)
            )
            # Print negotiation actions (skip year-0 None sentinel)
            for j, other in enumerate(agents):
                if other == ag:
                    continue
                prom = [s[j] for s in promises_series if s is not None]
                req  = [s[j] for s in requests_series if s is not None]
                acc  = [s[j] for s in accepts_series  if s is not None]
                print(f"  -> {other}: promised={prom}  requested={req}  accepted={acc}")

            policy_logger[ag] = {
                "lever_effort_fraction": lever_series,
                "adaptation_investment_fraction": adapt_levels.tolist(),
                "negotiation": {
                    "promises_idx": promises_series,   # T x N methane-level indices promised to each agent
                    "requests_idx": requests_series,   # T x N methane-level indices requested from each agent
                    "acceptances":  accepts_series,    # T x N accept (1) / reject (0) per agent
                },
            }

        return policy_logger

    def register_env_rl(self):
        def env_creator(cfg):
            return ClimateMARL(
                cfg["env_config"],
                cfg["emission_data"],
                cfg["economics_config"],
                cfg["actions_config"],
            )

        register_env("climate_marl", env_creator)

    def run_marl_experiment(self):
        self.register_env_rl()

        env = ClimateMARL(
            self.env_config,
            self.emission_data,
            self.economics_config,
            self.actions_config,
        )
        obs_sp = env.observation_space("country_0")
        act_sp = env.action_space("country_0")
        N = self.env_config["N"]

        policies = {f"country_{i}": (None, obs_sp, act_sp, {}) for i in range(N)}
        policy_mapping_fn = lambda agent_id, *_, **__: agent_id  # noqa: E731

        ppo_cfg = dict(self.training_cfg.get("ppo", {}))

        model_cfg = dict(ppo_cfg.get("model", {}))
        model_cfg.setdefault("use_lstm", True)
        model_cfg.setdefault("vf_share_layers", False)
        model_cfg.setdefault("lstm_cell_size", 64)
        model_cfg.setdefault("max_seq_len", self.env_config["horizon"] * 3)  # 3 stages per climate step

        train_batch_multiplier = ppo_cfg.get("train_batch_multiplier", 8)
        minibatch_multiplier = ppo_cfg.get("minibatch_multiplier", 2)
        train_batch_size = train_batch_multiplier * N * self.env_config["horizon"] * 3  # 3 stages per climate step
        minibatch_size = minibatch_multiplier * N * self.env_config["horizon"] * 3

        lr_schedule = ppo_cfg.get("lr_schedule")
        entropy_schedule = ppo_cfg.get("entropy_coeff_schedule")
        rollout_cfg = ppo_cfg.get("rollout", {})
        resources_cfg = ppo_cfg.get("resources", {})

        fragment_length = rollout_cfg.get("fragment_length", self.env_config["horizon"] * 3)
        if isinstance(fragment_length, str) and fragment_length.lower() == "horizon":
            fragment_length = self.env_config["horizon"] * 3  # 3 stages per climate step

        base_config = (
            PPOConfig()
            .framework(ppo_cfg.get("framework", "torch"))
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .environment(
                env="climate_marl",
                env_config={
                    "env_config": self.env_config,
                    "emission_data": self.emission_data,
                    "economics_config": self.economics_config,
                    "actions_config": self.actions_config,
                },
            )
            .training(
                model=model_cfg,
                train_batch_size=train_batch_size,
                minibatch_size=minibatch_size,
                num_epochs=ppo_cfg.get("num_epochs", 5),
                gamma=ppo_cfg.get("gamma", 0.999) ** (1.0 / (env.num_negotiation_stages + 1)),
                lr=ppo_cfg.get("lr", 2e-4),
                clip_param=ppo_cfg.get("clip_param", 0.3),
                entropy_coeff=ppo_cfg.get("entropy_coeff", 0.02),
                lr_schedule=lr_schedule,
                entropy_coeff_schedule=entropy_schedule,
            )
            .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
            .resources(num_gpus=resources_cfg.get("num_gpus", 0))
            .env_runners(
                rollout_fragment_length=fragment_length,
                batch_mode="complete_episodes",
                num_env_runners=rollout_cfg.get("num_env_runners", 1),
                num_envs_per_env_runner=rollout_cfg.get("num_envs_per_runner", 1),
                num_gpus_per_env_runner=rollout_cfg.get("num_gpus_per_env_runner", 0),
                sample_timeout_s=rollout_cfg.get("sample_timeout_s", 1200),
            )
        )

        base_config.seed = self.training_cfg.get("seed", 0)
        base_config.storage_path = "/tmp/ray_results"
        algo = base_config.build_algo()


        results_dir = Path(self.env_config["output_dir"])

        with open(results_dir / "env_config.json", "w") as f:
            json.dump(_to_jsonable(self.env_config), f, indent=4)
        with open(results_dir / "economics_config.json", "w") as f:
            json.dump(_to_jsonable(self.economics_config), f, indent=4)
        with open(results_dir / "actions_config.json", "w") as f:
            json.dump(_to_jsonable(self.actions_config), f, indent=4)
        with open(results_dir / "training_config.json", "w") as f:
            json.dump(_to_jsonable(self.training_cfg), f, indent=4)

        num_iterations = self.training_cfg.get("num_iterations", 150)
        eval_interval = max(1, self.training_cfg.get("greedy_eval_interval", 2))

        num_env_steps = 0
        per_agent_reward_logger_train = {}
        per_agent_reward_logger_greedy = {}
        per_agent_climate_reward_logger      = {}
        per_agent_tariff_cost_logger         = {}
        per_agent_noncompliance_cost_logger  = {}
        per_agent_policy_logger_greedy = {}
        temperature_logger = {}
        training_time_stats = {}
        t0 = time.time()

        for i in range(num_iterations):
            # trajectory, per_agent_return, _, _, _, info_dict = self.rollout_fixed_actions()
            # print("[fixed] per-agent returns:", {k: round(v, 2) for k, v in per_agent_return.items()})
            # print("[fixed] temperature trajectory:", info_dict['country_0']['Temperature_trajectory'])
            # self.print_greedy_summary(trajectory)

            result = algo.train()
            per_agent_rewards = result["env_runners"]["policy_reward_mean"]
            num_env_steps += result["env_runners"]["episodes_timesteps_total"]

            per_agent_reward_logger_train[num_env_steps] = per_agent_rewards
            print(f"Per-agent eval rewards at iteration {i} after {num_env_steps} timesteps")
            for policy_id, mean_rew in per_agent_rewards.items():
                print(f"  {policy_id}: {mean_rew:.4f}")

            training_time_stats[num_env_steps] = time.time() - t0

            if i % eval_interval == 0:
                traj, agent_ret, _, _, info_dict, climate_ret, tariff_ret, noncompliance_ret = self.rollout_greedy_actions(algo, policy_mapping_fn)
                per_agent_reward_logger_greedy[num_env_steps] = agent_ret
                per_agent_climate_reward_logger[num_env_steps] = climate_ret
                per_agent_tariff_cost_logger[num_env_steps] = tariff_ret
                per_agent_noncompliance_cost_logger[num_env_steps] = noncompliance_ret
                print("[greedy] per-agent returns:", {k: round(v, 2) for k, v in agent_ret.items()})
                print("[greedy] temperature trajectory:", info_dict['country_0']['Temperature_trajectory'])
                policy_logger = self.print_greedy_summary(traj)
                per_agent_policy_logger_greedy[num_env_steps] = policy_logger
                temperature_logger[num_env_steps] = info_dict['country_0']['Temperature_trajectory']

                intermediate = {
                    "train_reward": per_agent_reward_logger_train,
                    "greedy_reward": per_agent_reward_logger_greedy,
                    "climate_reward": per_agent_climate_reward_logger,
                    "tariff_cost": per_agent_tariff_cost_logger,
                    "noncompliance_cost": per_agent_noncompliance_cost_logger,
                    "temperature_trajectory": temperature_logger,
                    "greedy_policy": per_agent_policy_logger_greedy,
                    "training_time_stats": training_time_stats,
                }
                with open(results_dir / "ms_marl_experiment_results_intermediate.json", "w") as f:
                    json.dump(intermediate, f, indent=4)

            print("=" * 60)

        final_results = {
            "train_reward": per_agent_reward_logger_train,
            "greedy_reward": per_agent_reward_logger_greedy,
            "climate_reward": per_agent_climate_reward_logger,
            "tariff_cost": per_agent_tariff_cost_logger,
            "noncompliance_cost": per_agent_noncompliance_cost_logger,
            "temperature_trajectory": temperature_logger,
            "greedy_policy": per_agent_policy_logger_greedy,
            "training_time_stats": training_time_stats,
        }

        with open(results_dir / "ms_marl_experiment_results.json", "w") as f:
            json.dump(final_results, f, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="marl_homogenous.yaml", help="MARL config file name (looked up in config/ dir)")
    args = parser.parse_args()

    marl_config = load_yaml_config(args.config, "marl")
    env_config, emission_data, economics_config, actions_config, training_cfg = load_marl_setup(marl_config)

    experiment = ClimateMarlExperiment(
        env_config,
        emission_data,
        economics_config,
        actions_config,
        training_cfg,
    )
    experiment.run_marl_experiment()


if __name__ == "__main__":
    main()