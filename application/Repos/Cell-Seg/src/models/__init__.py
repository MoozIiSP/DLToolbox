from omegaconf import DictConfig
from src.utils.utils import load_obj


__all__ = ['get_unet_model']


def get_unet_model(cfg: DictConfig = None):
    """
    Get model

    Args:
        cfg: config

    Returns:

    """
    model = load_obj(cfg.model.backbone.class_name)
    model = model(**cfg.model.backbone.params)

    return model
