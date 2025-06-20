import os

from omegaconf import OmegaConf


def load_configs():
    """Load all configs from 'conf' folder."""
    conf_folder = "conf"

    merged_configs = OmegaConf.create()
    for root, _, files in os.walk(conf_folder):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                filepath = os.path.join(root, file)

                try:
                    config = OmegaConf.load(filepath)
                    merged_configs = OmegaConf.merge(merged_configs, config)
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")

    return merged_configs
