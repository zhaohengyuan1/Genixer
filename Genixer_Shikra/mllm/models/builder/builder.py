from typing import Dict, Any, Tuple

from torch import nn

from .build_model import load_pretrained_genixer_shikra, load_pretrained_genixer_shikra_lora, load_pretrained_shikra

PREPROCESSOR = Dict[str, Any]


# TODO: Registry
def load_pretrained(model_args, training_args) -> Tuple[nn.Module, PREPROCESSOR]:
    type_ = model_args.type
    if type_ == 'genixer_shikra':
        return load_pretrained_genixer_shikra(model_args, training_args)
    elif type_ == 'genixer_shikra_lora':
        return load_pretrained_genixer_shikra_lora(model_args, training_args)
    elif type_ == 'shikra':
        return load_pretrained_shikra(model_args, training_args)
    else:
        assert False
