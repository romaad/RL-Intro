import torch.nn as nn


class Model(nn.Module):
    """Abstract base class for AlphaZero neural network models.

    Subclasses define the full forward pass; the ``p_shape`` and
    ``v_shape`` attributes let callers know the output dimensions.
    """

    def __init__(self, input_shape: tuple, p_shape: tuple, v_shape: tuple) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.p_shape = p_shape
        self.v_shape = v_shape
