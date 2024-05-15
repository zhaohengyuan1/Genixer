import os
from typing import Dict, Any, Tuple

import torch
import transformers
from torch import nn

from ..models import GenixerShikraLlamaForCausalLM

PREPROCESSOR = Dict[str, Any]

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def load_pretrained_genixer_shikra_lora(model_args, training_args, **kwargs) -> Tuple[nn.Module, PREPROCESSOR]:\
    
    if model_args.base_model_path != None:
        model = GenixerShikraLlamaForCausalLM.from_pretrained(
            model_args.base_model_path,
            cache_dir=model_args.cache_dir,
            **kwargs
        )
    else:
        model = GenixerShikraLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            **kwargs
        )
    
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16

    if training_args.do_train:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
        print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    else: # for load pretrained lora model

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    assert model_args.version == 'v1'
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token

    model_vision_dict = model.get_model().initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        mm_vision_select_layer=model_args.mm_vision_select_layer,
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
    )
    
    # HACK for quantization
    if model.get_vision_tower().device != torch.device('meta'):
        model.get_vision_tower().to(dtype=dtype, device=training_args.device)
    else:
        from transformers import CLIPVisionModel
        model.get_model().vision_tower = CLIPVisionModel.from_pretrained(model_args.vision_tower)  # not quantize clip
        # model.model.vision_tower[0] = CLIPVisionModel.from_pretrained(model_args.vision_tower, **kwargs)  # quantize clip、
    vision_config = model_vision_dict['vision_config']

    if training_args.do_train:
        ## default: tune adapter and input/output embeddings
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        
        for p in model.get_input_embeddings().parameters():
            p.requires_grad = True
        for p in model.get_output_embeddings().parameters():
            p.requires_grad = True

    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = model_args.freeze_mm_mlp_adapter

    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    vision_config.use_im_start_end = model_args.mm_use_im_start_end
    model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end,
                                      tokenizer=tokenizer,
                                      device=training_args.device,
                                      tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
                                      pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)


    if training_args.do_train == False:

        if os.path.exists(os.path.join(os.path.join(model_args.model_name_or_path, 'non_lora_trainables.bin'))):
            non_lora_trainables = torch.load(os.path.join(model_args.model_name_or_path, 'non_lora_trainables.bin'), map_location='cpu')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}

            if any(k.startswith('model.model.embed_tokens') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            non_lora_trainables = {k: v.to(dtype) for k, v in non_lora_trainables.items()}

            model.load_state_dict(non_lora_trainables, strict=False)

    ### print all trainable parameters
    a = [print(n, p.shape, p.dtype) for n, p in model.named_parameters() if p.requires_grad]

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(len(params_no_grad),
                                                                                                                 params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
                    len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen parameters, this is experimental.")
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            # from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)

                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    preprocessor = dict(
        image=model_vision_dict['image_processor'],
        text=tokenizer,
        conv=dict(
            image_token_len=model_args.image_token_len,
            sep_image_conv_front=model_args.sep_image_conv_front,
            use_im_start_end=model_args.mm_use_im_start_end,
        )
    )
    return model, preprocessor

def load_pretrained_genixer_shikra(model_args, training_args, **kwargs) -> Tuple[nn.Module, PREPROCESSOR]:
    model = GenixerShikraLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        **kwargs
    )

    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    assert model_args.version == 'v1'
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token

    model_vision_dict = model.model.initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        mm_vision_select_layer=model_args.mm_vision_select_layer,
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
    )
    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16
    # HACK for quantization
    if model.model.vision_tower[0].device != torch.device('meta'):
        model.model.vision_tower[0].to(dtype=dtype, device=training_args.device)
    else:
        from transformers import CLIPVisionModel
        model.model.vision_tower[0] = CLIPVisionModel.from_pretrained(model_args.vision_tower)  # not quantize clip
        # model.model.vision_tower[0] = CLIPVisionModel.from_pretrained(model_args.vision_tower, **kwargs)  # quantize clip、
    vision_config = model_vision_dict['vision_config']

    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        # model.requires_grad_(False)
        for p in model.model.model.mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = model_args.freeze_mm_mlp_adapter
    if model_args.freeze_mm_mlp_adapter:
        for p in model.model.model.mm_projector.parameters():
            p.requires_grad = False

    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    vision_config.use_im_start_end = model_args.mm_use_im_start_end
    model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end,
                                      tokenizer=tokenizer,
                                      device=training_args.device,
                                      tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
                                      pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(len(params_no_grad),
                                                                                                                 params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
                    len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen parameters, this is experimental.")
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            # from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)

                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)
    
    ### print all trainable parameters
    a = [print(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
    
    preprocessor = dict(
        image=model_vision_dict['image_processor'],
        text=tokenizer,
        conv=dict(
            image_token_len=model_args.image_token_len,
            sep_image_conv_front=model_args.sep_image_conv_front,
            use_im_start_end=model_args.mm_use_im_start_end,
        )
    )
    return model, preprocessor

def load_pretrained_shikra(model_args, training_args, **kwargs) -> Tuple[nn.Module, PREPROCESSOR]:
    model = ShikraLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        **kwargs
    )
    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=model_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    assert model_args.version == 'v1'
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token

    model_vision_dict = model.model.initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        mm_vision_select_layer=model_args.mm_vision_select_layer,
        pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
    )
    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16
    # HACK for quantization
    if model.model.vision_tower[0].device != torch.device('meta'):
        model.model.vision_tower[0].to(dtype=dtype, device=training_args.device)
    else:
        from transformers import CLIPVisionModel
        model.model.vision_tower[0] = CLIPVisionModel.from_pretrained(model_args.vision_tower)  # not quantize clip
        # model.model.vision_tower[0] = CLIPVisionModel.from_pretrained(model_args.vision_tower, **kwargs)  # quantize clip、
    vision_config = model_vision_dict['vision_config']

    model.config.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.model.mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = model_args.freeze_mm_mlp_adapter
    if model_args.freeze_mm_mlp_adapter:
        for p in model.model.mm_projector.parameters():
            p.requires_grad = False

    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    vision_config.use_im_start_end = model_args.mm_use_im_start_end
    model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end,
                                      tokenizer=tokenizer,
                                      device=training_args.device,
                                      tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter,
                                      pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(len(params_no_grad),
                                                                                                                 params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
                    len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen parameters, this is experimental.")
            print(
                "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            # from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)

                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)
    
    ### print all trainable parameters
    a = [print(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]

    preprocessor = dict(
        image=model_vision_dict['image_processor'],
        text=tokenizer,
        conv=dict(
            image_token_len=model_args.image_token_len,
            sep_image_conv_front=model_args.sep_image_conv_front,
            use_im_start_end=model_args.mm_use_im_start_end,
        )
    )
    return model, preprocessor


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
