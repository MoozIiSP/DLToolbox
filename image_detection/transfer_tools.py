# TODO We will remove keras-vis in the future
from vis.utils.utils import find_layer_idx


def extract_weights_of_layer(model, layer_name):
    """According to layer name, extracting weights from model."""
    layer_idx = find_layer_idx(model, layer_name)

    return model.layers[layer_idx].get_weights()


def inject_weights_to_layer(model, layer_name, weights):
    layer_idx = find_layer_idx(model, layer_name)
    model.layers[layer_idx].set_weights(weights)
