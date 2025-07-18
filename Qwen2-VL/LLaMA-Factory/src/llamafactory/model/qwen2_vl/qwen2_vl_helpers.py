import os
from copy import deepcopy

import torch
from llamafactory.model.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration, make_smaller
from torchinfo import summary
from transformers.modeling_utils import load_state_dict


def _pkd_infer_distillation_information_from_model_path(model_path):
    """
    Infers two pieces of information from the given model path:
      - fixed_depth_reduction: determined by checking if the model_path contains "depthreduce4" or "depthreduce2".
      - reduction_percantage: read from a file named "reduction_percantage.txt" in the checkpoint folder.
    Example model_path:
      '...../qwen2vl_lora_sft_qlora_500000_unfreeze8_0.5mmdrop_336res_578imgtoks_pkd_depthreduce4/checkpoint-52500'
    """

    # Determine fixed_depth_reduction from model_path string
    if "depthreduce4" in model_path:
        fixed_depth_reduction = 4
    elif "depthreduce2" in model_path:
        fixed_depth_reduction = 2
    else:
        fixed_depth_reduction = 1

    # Determine the parent directory containing all checkpoints
    parent_dir = os.path.dirname(model_path.rstrip("/"))
    current_ckpt = os.path.basename(model_path.rstrip("/"))

    # Attempt to read reduction_percantage.txt from the previous checkpoint folder
    file_path = os.path.join(parent_dir, current_ckpt, "reduction_percantage.txt")
    with open(file_path, "r") as f:
        reduction_percentage = float(f.read().strip())

    print(f'Inferred fixed_depth_reduction: {fixed_depth_reduction}, reduction_percantage: {reduction_percentage}')
    return fixed_depth_reduction, reduction_percentage


def load_pretrained_model(model_path):
    if '_pkd' in model_path:  # this is a seperate case then     load_state_dict
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            'Qwen2-VL/LLaMA-Factory/saves/qwen2vl_lora_sft_qlora_1000000_unfreeze8_0.5mmdrop_336res_578imgtoks_pkd_teacher',
            torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager")
        del model.visual.image_pooler.temporal_cross_attention
        del model.visual.image_pooler.temporal_segment_embedding
        del model.visual.image_pooler.point_transformer
        del model.visual.image_pooler.point_pooling
        del model.visual.image_pooler.project_audio
        # we can rather reliable infer reduction_percantage and fixed_depth_reduction. We will attempt this.
        fixed_depth_reduction, reduction_percentage = _pkd_infer_distillation_information_from_model_path(model_path)
        model.original_config = deepcopy(model.config)
        if fixed_depth_reduction > 1:
            model = make_smaller(model, reduction_percentage=reduction_percentage, only_llm=True, fixed_depth_reduction=fixed_depth_reduction)
        else:
            model = make_smaller(model, reduction_percentage=reduction_percentage, only_llm=True)
        safe_tensors_path = os.path.join(model_path, "model.safetensors")
        state_dict = load_state_dict(safe_tensors_path)
        model.load_state_dict(state_dict, strict=False)
        # Check if visual_components directory exists
        visual_block = os.path.join(model_path, "visual_block.pt")
        if os.path.exists(visual_block):
            print("Loading visual block")
            visual_block = torch.load(visual_block, map_location=None if torch.cuda.is_available() else 'cpu')  # force cpu if necessary
            model.visual.load_state_dict(visual_block, strict=False)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto", attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager")
        # Check if visual_components directory exists
        visual_block = os.path.join(model_path, "visual_block.pt")
        if os.path.exists(visual_block):
            print("Loading visual block")
            visual_block = torch.load(visual_block, map_location=None if torch.cuda.is_available() else 'cpu')
            model.visual.load_state_dict(visual_block, strict=False)  # TODO because of the recent cross attention adjustment it does not work with strict=True

    summary(model, col_names=['num_params', 'trainable'], depth=5)
    return model
