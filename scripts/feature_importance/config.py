import logging
import pathlib

import yaml

logger = logging.getLogger(__name__)

def get_configs():
    return Config().cfg

class Config:
    def __init__(self):
        cfgs = []
        for d in [pathlib.Path.cwd(), pathlib.Path(__file__).parent/'../../static']:
            if (d/'model_settings.yml').exists():
                with (d/'model_settings.yml').open() as f:
                    cfgs.append(yaml.unsafe_load(f))
        self.cfg = cfgs[0]