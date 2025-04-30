import glob
import os

import omegaconf
from omegaconf import OmegaConf

RELATIVE_CONF_DIR = "conf"  # from project root


def configs() -> omegaconf.DictConfig:
    """
    Loads all .yaml file configurations in the /conf folder.
    """
    yaml_files = glob.glob(os.path.join(RELATIVE_CONF_DIR, "*.yaml"))
    configs = [OmegaConf.load(file) for file in yaml_files]
    merged_config = OmegaConf.merge(*configs)

    return merged_config
