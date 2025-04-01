import yaml

FIXED_CONFIG_PATH = "conf/config.yaml"


def configs():
    """Returns all configs from FIXED_CONFIG_PATH.
    This configuration path should not change unless the
    config.yaml is shifted elsewhere."""
    with open(FIXED_CONFIG_PATH, "r") as file:
        configs = yaml.safe_load(file)

    return configs
