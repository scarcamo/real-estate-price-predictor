import yaml


def load_config(file_path="config/config.yaml"):
    """Loads YAML configuration from the given file path."""
    try:
        with open(file_path, "r") as stream:
            config = yaml.safe_load(stream)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: The file {file_path} was not found.")

    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML file: {exc}")
