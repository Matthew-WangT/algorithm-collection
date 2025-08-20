from typing import List, Tuple

import torch
from torch import nn, Tensor


class SymmetricAutoencoder(nn.Module):
    def __init__(self, layer_sizes: List[int]) -> None:
        """
        A fully-connected autoencoder with mirrored encoder/decoder.

        layer_sizes: [input_size, ..., code_size]
        """
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements: [input_size, code_size]")

        # Encoder
        encoder_layers: List[nn.Module] = []
        input_size = layer_sizes[0]
        for i, output_size in enumerate(layer_sizes[1:]):
            encoder_layers.append(nn.Linear(input_size, output_size))
            # ReLU on all encoder layers
            encoder_layers.append(nn.ReLU())
            input_size = output_size
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror of encoder without the final activation; add Tanh at the end)
        decoder_layers: List[nn.Module] = []
        reversed_layer_sizes = list(layer_sizes)[::-1]
        input_size = reversed_layer_sizes[0]
        for i, output_size in enumerate(reversed_layer_sizes[1:]):
            decoder_layers.append(nn.Linear(input_size, output_size))
            # ReLU for all but the last linear layer
            if i < len(layer_sizes) - 2:
                decoder_layers.append(nn.ReLU())
            input_size = output_size
        # Clamp outputs to [-1, 1] since inputs will be normalized to that range
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        code = self.encoder(x)
        recovered = self.decoder(code)
        return recovered, code


