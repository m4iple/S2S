import toml

def load_config(config_path="configs/s2s.toml"):
    with open(config_path, 'r') as f:
        return toml.load(f)