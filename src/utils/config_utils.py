from pathlib import Path

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_DIR = _PROJECT_ROOT / "config"


def load_yaml_config(path, nested_key=None):
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        candidate = _DEFAULT_CONFIG_DIR / path
        if candidate.exists():
            path = candidate
    with path.open("r") as stream:
        data = yaml.safe_load(stream)
    if nested_key is None:
        return data
    if nested_key not in data:
        raise KeyError(f"Expected key '{nested_key}' in {path}")
    return data[nested_key]


def load_data_run_configs(data_dir):
    data_dir = Path(data_dir)
    configs_dir = data_dir / "configs"
    if not configs_dir.exists():
        raise FileNotFoundError(f"Configs directory not found: {configs_dir}")
    cicero = load_yaml_config(configs_dir / "cicero_scm.yaml")
    data_generation = load_yaml_config(
        configs_dir / "data_generation.yaml", "data_generation_params"
    )
    data_processing = load_yaml_config(
        configs_dir / "data_processing.yaml", "data_processing_params"
    )
    data_cfg = {
        "cicero": cicero,
        "data_generation": data_generation,
        "data_processing": data_processing,
    }
    training_cfg_path = configs_dir / "train.yaml"
    if training_cfg_path.exists():
        data_cfg["train"] = load_yaml_config(training_cfg_path)
    return data_cfg


def load_training_run_config(training_run):
    training_run = Path(training_run)
    train_config_path = training_run / "train_config.yaml"
    if not train_config_path.exists():
        raise FileNotFoundError(f"Training config not found: {train_config_path}")
    return load_yaml_config(train_config_path)
