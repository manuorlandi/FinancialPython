#########################################################################################################
# SETTINGS
#########################################################################################################

# libraries
import os
import yaml

#########################################################################################################
# CFG
#########################################################################################################


def __cfg_path(env: str = "") -> str:
    """Path where to find conf.yml file.
    Info like bucket name are saved on a dedicated conf files, according to running environment

    Args:
        env (str, optional): env used. Defaults to "".

    Returns:
        str: conf.yml file path
    """

    if env == "":
        return os.path.join(os.getcwd(), "environment", "conf.yml")
    else:
        return os.path.join(os.getcwd(), "environment", f"{env}-conf.yml")


def __cfg_reading(env: str = "") -> yaml:
    """Read conf.yml file.
    Info like bucket name are saved on a dedicated conf files, according to running environment

    Args:
        env (str, optional): env used. Defaults to "".

    Returns:
        yaml: conf.yml content
    """

    path = __cfg_path(env=env)

    with open(path, "r", encoding="utf-8") as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)
    return cfg
