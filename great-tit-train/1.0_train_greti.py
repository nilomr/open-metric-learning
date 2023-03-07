import hydra
from omegaconf import DictConfig

from oml.lightning.entrypoints.train import pl_train


@hydra.main(config_path=".", config_name="train_greti.yaml")
def main_hydra(cfg: DictConfig) -> None:
    print("Training model on great tit dataset")
    pl_train(cfg)


if __name__ == "__main__":
    main_hydra()
