import numpy as np
import pandas as pd
from pathlib import Path
import torch
import time

from utils.cicero_utils import load_cicero_inputs  # noqa: E402
from src.utils.config_utils import (
    load_data_run_configs,
    load_training_run_config,
    load_yaml_config,
)  # noqa: E402
from utils.data_utils import (
    build_emission_data,
    determine_variable_gases,
    load_processed_data,
)  # noqa: E402

def prepare_net_params(training_run, model_cfg, device, use_half, autocast, processed_dir):
    state = torch.load(training_run / "model.pth", map_location="cpu")
    state_np = {k: v.numpy() for k, v in state.items()}

    X_train, *_, mu, std = load_processed_data(processed_dir)

    hidden_dims = model_cfg["hidden_dims"]
    hidden = hidden_dims[0] if isinstance(hidden_dims, (list, tuple)) else hidden_dims

    params = {
        "state_dict_np": state_np,
        "device": device,
        "mu": mu.tolist(),
        "std": std.tolist(),
        "autocast": bool(autocast),
        "use_half": bool(use_half),
        "model_type": model_cfg["model_type"].lower(),
        "hidden": hidden,
        "num_layers": model_cfg["num_layers"],
        "kernel_size": model_cfg.get("kernel_size", 3),
        "n_gas": int(X_train.shape[2]),
    }
    return params

def load_marl_setup(cfg):
    general_cfg = cfg["general"]
    env_cfg = cfg["environment"]
    economics_cfg = cfg["economics"]
    actions_cfg = cfg["actions"]
    training_cfg = cfg["training"]

    data_dir = Path(general_cfg["data_dir"]).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    data_configs = load_data_run_configs(data_dir)
    cicero_cfg = data_configs["cicero"]
    data_gen_cfg = data_configs["data_generation"]
    data_proc_cfg = data_configs["data_processing"]

    gaspam_data, conc_data, em_data, nat_ch4_data, nat_n2o_data = load_cicero_inputs(
        cicero_cfg
    )
    gas_names = list(em_data.columns)

    controlled_gases = determine_variable_gases(data_gen_cfg, gas_names)

    baseline_shares = np.array(env_cfg["baseline_emission_shares"], dtype=np.float32)
    
    env_config = {
        "N": int(env_cfg["num_agents"]),
        "engine": general_cfg["climate_engine"],
        "horizon": int(env_cfg["horizon"]),
        "hist_end": int(cicero_cfg["em_data_policy"]),
        "future_end": int(cicero_cfg["em_data_policy"]) + int(env_cfg["horizon"]),
        "random_seed": int(general_cfg.get("random_seed", 0)),
        "controlled_gases": controlled_gases,
        "window_size": int(data_proc_cfg["window_size"]),
        "run_name": general_cfg["marl_run_name"],
        "data_dir": general_cfg["data_dir"],
        "training_run_id": general_cfg["training_run_id"],
        "rollout_length": int(env_cfg["rollout_length"]),
        "log_episode_trajectories": bool(general_cfg["log_episode_trajectories"]),
        "log_dir": general_cfg["log_dir"],
    }

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    data_dir = Path(env_config["data_dir"])
    if env_config['engine'] == 'net':
        training_run_id = env_config['training_run_id']
    else:
        training_run_id = env_config['engine']
    marl_run_name = env_config['run_name']
    folder_name = f"{timestamp}_{training_run_id}_{marl_run_name}"

    results_dir = Path(training_cfg["output_dir"])
    results_dir = data_dir / results_dir
    results_dir = results_dir / folder_name
    results_dir.mkdir(parents=True, exist_ok=False)
    env_config["output_dir"] = str(results_dir.expanduser().absolute())



    if len(baseline_shares) != env_config["N"]:
        raise ValueError("baseline_emission_shares length must match num_agents")
    if not np.isclose(baseline_shares.sum(), 1.0):
        raise ValueError(f"baseline_emission_shares must sum to 1.0 but it sums to {baseline_shares.sum()}")

    emission_data = build_emission_data(em_data, baseline_shares, cicero_cfg["em_data_policy"])

    emission_data.update(
        {
            "gaspam_data": gaspam_data,
            "conc_data": conc_data,
            "nat_ch4_data": nat_ch4_data,
            "nat_n2o_data": nat_n2o_data,
        }
    )

    scm_params = {
        "gaspam_data": gaspam_data,
        "conc_data": conc_data,
        "nat_ch4_data": nat_ch4_data,
        "nat_n2o_data": nat_n2o_data,
        "pamset_udm": cicero_cfg["cfg"][0]["pamset_udm"],
        "pamset_emiconc": cicero_cfg["cfg"][0]["pamset_emiconc"],
        "em_data_start": cicero_cfg["em_data_start"],
        "em_data_policy": cicero_cfg["em_data_policy"],
        "udir": cicero_cfg["test_data_dir"],
        "idtm": 24,
        "scenname": general_cfg.get("scm_scenname", "marl_scm_run"),
    }
    env_config["scm_params"] = scm_params

    if env_config["engine"].lower() == "net":
        training_run_dir = Path("training_runs") / Path(general_cfg["training_run_id"])
        if not training_run_dir.is_absolute():
            training_run_dir = data_dir / training_run_dir
        if not training_run_dir.exists():
            raise FileNotFoundError(f"training_run not found: {training_run_dir}")

        train_cfg = load_training_run_config(training_run_dir)
        model_cfg = train_cfg["model"]

        net_params = prepare_net_params(
            training_run_dir,
            model_cfg,
            device=general_cfg.get("surrogate_device", "cpu"),
            use_half=general_cfg.get("surrogate_use_half", False),
            autocast=general_cfg.get("surrogate_autocast", False),
            processed_dir=data_dir / "processed",
        )
        if net_params["n_gas"] != len(controlled_gases):
            raise ValueError(
                "Surrogate input dimension does not match variable gas list: "
                f"{net_params['n_gas']} vs {len(controlled_gases)}"
            )
        env_config["net_params"] = net_params
    else:
        env_config["net_params"] = None

    prevention_decay = float(economics_cfg.get("prevention_decay", 0.95))
    max_prevention_benefit = float(economics_cfg.get("max_prevention_benefit", 0.5))

    costs_cfg = economics_cfg.get("costs")
    if costs_cfg is None:
        raise ValueError("economics.costs section is required in MARL config")

    required_cost_keys = (
        "climate_damage_costs",
        "energy_costs",
        "methane_costs",
        "agriculture_costs",
        "adaptation_costs",
    )
    optional_cost_keys = (
        "noncompliance_fine",
        "tariff",
    )
    processed_costs = {}
    for key in required_cost_keys:
        if key not in costs_cfg:
            raise ValueError(f"Missing economics.costs entry: {key}")
        arr = np.asarray(costs_cfg[key], dtype=np.float32)
        if arr.shape != (env_config["N"],):
            raise ValueError(
                f"economics.costs.{key} must have length {env_config['N']}, got {len(arr)}"
            )
        processed_costs[key] = arr
    for key in optional_cost_keys:
        if key in costs_cfg:
            arr = np.asarray(costs_cfg[key], dtype=np.float32)
            if arr.shape != (env_config["N"],):
                raise ValueError(
                    f"economics.costs.{key} must have length {env_config['N']}, got {len(arr)}"
                )
            processed_costs[key] = arr

    economics_config = {
        "costs": processed_costs,
        "prevention_decay": prevention_decay,
        "max_prevention_benefit": max_prevention_benefit,
    }

    lever_names = ["energy", "methane", "agriculture"]
    lever_level_keys = {
        "energy": "energy_levels",
        "methane": "methane_levels",
        "agriculture": "agri_levels",
    }

    lever_levels = {}
    action_sizes = []
    for lever in lever_names:
        key = lever_level_keys[lever]
        if key not in actions_cfg:
            raise ValueError(f"Missing actions entry: {key}")
        levels = np.asarray(actions_cfg[key], dtype=np.float32)
        if levels.ndim != 1 or levels.size == 0:
            raise ValueError(f"actions.{key} must be a non-empty 1D list; got shape {levels.shape}")
        lever_levels[lever] = levels
        action_sizes.append(int(levels.size))

    if "adaptation_levels" not in actions_cfg:
        raise ValueError("actions.adaptation_levels is required")

    adaptation_levels = np.asarray(actions_cfg["adaptation_levels"], dtype=np.float32)
    if adaptation_levels.ndim != 1 or adaptation_levels.size == 0:
        raise ValueError("actions.adaptation_levels must be a non-empty 1D list")

    action_sizes.append(int(adaptation_levels.size))

    policy_matrix_cfg = actions_cfg.get("policy_matrix", {})
    gas_index = {gas: idx for idx, gas in enumerate(controlled_gases)}
    policy_matrix = np.zeros((len(lever_names), len(controlled_gases)), dtype=np.float32)

    for row, lever in enumerate(lever_names):
        mapping = policy_matrix_cfg.get(lever)
        if mapping is None:
            raise ValueError(f"Missing policy_matrix entry for lever '{lever}'")
        for gas, coeff in mapping.items():
            if gas not in gas_index:
                raise ValueError(
                    f"Gas '{gas}' in policy_matrix[{lever}] is not among controlled gases {controlled_gases}"
                )
            policy_matrix[row, gas_index[gas]] = float(coeff)

    gas_totals = policy_matrix.sum(axis=0)  # (G,)
    if np.any(gas_totals < -0.051):
        raise ValueError(
            "Policy matrix reduction exceeds the -5% yearly budget for at least one gas"
        )

    actions_config = {
        "lever_names": lever_names,
        "lever_levels": lever_levels,
        "policy_matrix": policy_matrix,
        "policy_matrix_config": policy_matrix_cfg,
        "adaptation_levels": adaptation_levels,
        "action_sizes": action_sizes,
    }

    training_cfg.setdefault("num_iterations", 150)
    training_cfg.setdefault("greedy_eval_interval", 2)
    training_cfg.setdefault("output_dir", "marl_results")

    return env_config, emission_data, economics_config, actions_config, training_cfg


# Save configs
def _to_jsonable(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj