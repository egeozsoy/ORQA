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
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional

import torch
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
from ...model.qwen2_vl.modeling_qwen2_vl import make_smaller

if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments

logger = get_logger(__name__)


def run_pkd(
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
    # teacher_model needs slightly different arguments. Deepcopy and adjust
    old_finetuning_type = finetuning_args.finetuning_type
    finetuning_args.finetuning_type = 'lora'
    teacher_model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    finetuning_args.finetuning_type = old_finetuning_type
    del old_finetuning_type
    # merge and unload
    teacher_model = teacher_model.merge_and_unload()
    model_args.previous_model_weights = None  # we don't need this anymore
    # freeze everything about this model, set to eval etc.
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()
    teacher_model.visual.image_pooler.point_transformer.float()
    teacher_model.visual.image_pooler.fix_number_of_image_tokens = data_args.fix_number_of_image_tokens
    teacher_model.visual.image_pooler.use_past_visual_embeds = data_args.use_past_visual_embeds

    try:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    except Exception as e:
        print('No existing checkpoint found')
        last_checkpoint = None
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint {last_checkpoint}")
        training_args.resume_from_checkpoint = last_checkpoint

    student_model = load_model(tokenizer, model_args, finetuning_args,
                               training_args.do_train)  # either will be the same as teacher_model, or will resume from last checkpoint. This might be tricky because we will be making this model smaller.
    # now we will copy the weights from teacher_model to student_model
    for (teacher_name, teacher_param), (student_name, student_param) in zip(teacher_model.named_parameters(), student_model.named_parameters()):
        assert teacher_name == student_name, f"Teacher and student_model model have different parameter names: {teacher_name} != {student_name}"
        student_param.data.copy_(teacher_param.data)

    # unfreeze all params
    for param in student_model.parameters():
        param.requires_grad = True
    # student won't have any temporal cross attention, temporal embeddings, point transformer or audio. We delete these now. Also the teacher does not need these for PKD.
    del student_model.visual.image_pooler.temporal_cross_attention
    del student_model.visual.image_pooler.temporal_segment_embedding
    del student_model.visual.image_pooler.point_transformer
    del student_model.visual.image_pooler.point_pooling
    del student_model.visual.image_pooler.project_audio

    del teacher_model.visual.image_pooler.temporal_cross_attention
    del teacher_model.visual.image_pooler.temporal_segment_embedding
    del teacher_model.visual.image_pooler.point_transformer
    del teacher_model.visual.image_pooler.point_pooling
    del teacher_model.visual.image_pooler.project_audio
    student_model.visual.image_pooler.fix_number_of_image_tokens = data_args.fix_number_of_image_tokens
    student_model.visual.image_pooler.use_past_visual_embeds = data_args.use_past_visual_embeds
    student_model.original_config = deepcopy(student_model.config)
    if finetuning_args.pkd_fixed_depth_reduction is not None:
        student_model = make_smaller(student_model, reduction_percentage=finetuning_args.pkd_start_reduction, only_llm=True, fixed_depth_reduction=finetuning_args.pkd_fixed_depth_reduction)
    else:
        student_model = make_smaller(student_model, reduction_percentage=finetuning_args.pkd_start_reduction, only_llm=True)

    import gc
    torch.cuda.empty_cache()
    gc.collect()
    summary(student_model, col_names=['num_params', 'trainable'], depth=5)
    if getattr(student_model, "is_quantized", False) and not training_args.do_train:
        setattr(student_model, "_hf_peft_config_loaded", True)  # hack here: make model compatible with prediction

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(student_model.config, "_attn_implementation", None),
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
        model=student_model,
        teacher_model=teacher_model,
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
    if training_args.do_train and not training_args.do_predict:
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
    if training_args.do_eval and not training_args.do_predict:
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
