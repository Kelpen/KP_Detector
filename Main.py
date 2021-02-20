from Engine import Trainer
from Configs import Configs as Cfg

if __name__ == '__main__':
    cfg = Cfg.load_configs()

    t = Trainer.COMTrainer(cfg)
    t.train()
