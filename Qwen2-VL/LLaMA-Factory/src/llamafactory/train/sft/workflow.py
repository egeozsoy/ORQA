# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING, List, Optional

import torch
from torch import nn
from torchinfo import summary
from transformers.trainer_utils import get_last_checkpoint

from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from .trainer import CustomSeq2SeqTrainer
from ..trainer_utils import create_modelcard_and_push
from ...data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps, get_logits_processor
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

logger = get_logger(__name__)


def run_sft(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[List["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    try:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    except Exception as e:
        print('No existing checkpoint found')
        last_checkpoint = None
    curriculum_training = model_args.previous_model_weights is not None  # if this is already set, then we are continuing from previous weights for a different model like for temporal training
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint {last_checkpoint}")
        if model_args.previous_model_weights is not None:
            print(f'Warning: previous_model_weights is not None but actually {model_args.previous_model_weights}, but a checkpoint is found. previous_model_weights will be ignored.')
        model_args.previous_model_weights = last_checkpoint
        training_args.resume_from_checkpoint = last_checkpoint

    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    continue_from_previous_weights = model_args.previous_model_weights is not None  # this can be true either if we are continuing from previous weights or if we are resuming training from a checkpoint
    # we have three cases. 1) First time init 2) resume training 3) continue from previous weights.
    # unless it is 3) we want to properly manage the requires_grad of everything
    first_time_init = not continue_from_previous_weights
    resume_training_from_last_checkpoint = continue_from_previous_weights and not curriculum_training

    if first_time_init:
        print("First time init, resetting weights of image pooler and point transformer")
        model.visual.image_pooler.bert = model.visual.image_pooler.bert.apply(model.visual.image_pooler.bert._init_weights)  # only needed if first time initting, SHALL NOT BE DONE IF resume training
        model.visual.image_pooler.point_transformer._init_weights(dtype=torch.float32, device=training_args.device)  # important!
    if first_time_init or resume_training_from_last_checkpoint:
        print("Not curriculum training, setting necessary requires_grad")
        if finetuning_args.unfreeze_last_n_vision_tower_layers is not None:
            # unfreeze last N layers of model.visual.blocks as well as, model.visual.merger
            for layer in model.visual.blocks[-finetuning_args.unfreeze_last_n_vision_tower_layers:] + [model.visual.merger]:
                for param in layer.parameters():
                    param.requires_grad = True
        # image pooler should be unfrozen
        for param in model.visual.image_pooler.parameters():
            param.requires_grad = True
    if curriculum_training:
        for layer in model.visual.blocks + [model.visual.merger]:
            for param in layer.parameters():
                param.requires_grad = False
        for param in model.visual.image_pooler.parameters():
            param.requires_grad = False
    # Reinit temporal_cross_attention regardless
    model.visual.image_pooler.temporal_cross_attention = nn.MultiheadAttention(embed_dim=model.visual.image_pooler.temporal_cross_attention.embed_dim, num_heads=8, batch_first=True).to(
        training_args.device, torch.bfloat16)
    model.visual.image_pooler.temporal_segment_embedding = nn.Embedding(100, model.visual.image_pooler.temporal_segment_embedding.embedding_dim).to(training_args.device, torch.bfloat16)
    # always set to requires grad = True, both of them
    for param in model.visual.image_pooler.temporal_cross_attention.parameters():
        param.requires_grad = True
    for param in model.visual.image_pooler.temporal_segment_embedding.parameters():
        param.requires_grad = True
    # make sure point_transformer is in float32
    model.visual.image_pooler.point_transformer.float()
    model.visual.image_pooler.fix_number_of_image_tokens = data_args.fix_number_of_image_tokens
    model.visual.image_pooler.use_past_visual_embeds = data_args.use_past_visual_embeds
    summary(model, col_names=['num_params', 'trainable'], depth=5)
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        **tokenizer_module,
    )

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = training_args.generation_max_length or data_args.cutoff_len
    training_args.generation_num_beams = data_args.eval_num_beams or training_args.generation_num_beams
    training_args.remove_unused_columns = False  # important for multimodal dataset

    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor

    # Initialize our Trainer
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )

    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["logits_processor"] = get_logits_processor()

    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"  # use left-padding in generation

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        if training_args.predict_with_generate:  # eval_loss will be wrong if predict_with_generate is enabled
            metrics.pop("eval_loss", None)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.warning_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        if training_args.predict_with_generate:  # predict_loss will be wrong if predict_with_generate is enabled
            predict_results.metrics.pop("predict_loss", None)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
