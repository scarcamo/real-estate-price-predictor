import os

import yaml

path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")


def load_config(file_path=path):
    """Loads YAML configuration from the given file path."""
    try:
        with open(file_path, "r") as stream:
            config = yaml.safe_load(stream)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {file_path} was not found.")

    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML file: {exc}")
