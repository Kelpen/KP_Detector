import json


def load_configs():
    with open('Configs/COM_config.json', 'r') as cfg_file:
        cfg = json.load(cfg_file)
    return cfg


if __name__ == '__main__':
    c = load_configs()
    print(c)
