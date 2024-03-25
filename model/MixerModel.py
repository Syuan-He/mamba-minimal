import json

import torch
from torch import nn

from model.Block import Block
from utils.RMSNorm import RMSNorm
from utils.ModelArgs import ModelArgs
from utils.InferenceParams import InferenceParams

class MixerModel(nn.Module):
    def __init__(self, args: ModelArgs):
        """Full Mamba model."""
        super().__init__()
        self.args = args

        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.layers = nn.ModuleList([Block(args, layer_idx=i) for i in range(args.n_layer)])
        self.norm_f = RMSNorm(args.d_model)

        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.lm_head.weight = (
            self.embedding.weight
        )  # Tie output projection to embedding weights.
        # See "Weight Tying" paper

    def forward(self, input_ids, inference_params: InferenceParams=None):
        """
        Args:
            input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)

        Returns:
            logits: shape (b, l, vocab_size)

        Official Implementation:
            class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

        """
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x, inference_params=inference_params)
        # print(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        # print(x)
        return logits

    @staticmethod
    def from_pretrained(pretrained_model_name: str):
        """Load pretrained weights from HuggingFace into model.

        Args:
            pretrained_model_name: One of
                * 'state-spaces/mamba-2.8b-slimpj'
                * 'state-spaces/mamba-2.8b'
                * 'state-spaces/mamba-1.4b'
                * 'state-spaces/mamba-790m'
                * 'state-spaces/mamba-370m'
                * 'state-spaces/mamba-130m'

        Returns:
            model: Mamba model with weights loaded

        """
        from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
        from transformers.utils.hub import cached_file

        def load_config_hf(model_name):
            resolved_archive_file = cached_file(
                model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False
            )
            return json.load(open(resolved_archive_file))

        def load_state_dict_hf(model_name, device=None, dtype=None):
            resolved_archive_file = cached_file(
                model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False
            )
            return torch.load(
                resolved_archive_file, weights_only=True, map_location="cpu", mmap=True
            )

        config_data = load_config_hf(pretrained_model_name)
        args = ModelArgs(
            d_model=config_data["d_model"],
            n_layer=config_data["n_layer"],
            vocab_size=config_data["vocab_size"],
        )
        model = MixerModel(args)

        state_dict = load_state_dict_hf(pretrained_model_name)
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace("backbone.", "")
            new_state_dict[new_key] = state_dict[key]
        model.load_state_dict(new_state_dict)

        return model

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }