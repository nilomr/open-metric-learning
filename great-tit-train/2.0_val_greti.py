import hydra
from omegaconf import DictConfig

from oml.lightning.entrypoints.validate import pl_val


@hydra.main(config_path=".", config_name="2.0_val_greti.yaml")
def main_hydra(cfg: DictConfig) -> None:
    pl_val(cfg)


if __name__ == "__main__":
    main_hydra()