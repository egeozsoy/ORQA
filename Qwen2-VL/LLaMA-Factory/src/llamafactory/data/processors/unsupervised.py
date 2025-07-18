# Copyright 2024 the LlamaFactory team.
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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from .processor_utils import infer_seqlen
from ..data_utils import Role
from ...extras import logging

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template

logger = logging.get_logger(__name__)


def _encode_unsupervised_example(
        prompt: Sequence[Dict[str, str]],
        response: Sequence[Dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: Sequence["ImageInput"],
        videos: Sequence["VideoInput"],
        template: "Template",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        cutoff_len: int,
        audio: Optional[str] = None,
        pc: Optional[str] = None,
        segmasks: Optional[str] = None,
        id: Optional[str] = None,
) -> Tuple[List[int], List[int]]:
    if len(response) == 1:
        messages = prompt + response
    else:
        messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]

    messages = template.mm_plugin.process_messages(messages, images, videos, processor, audio, pc, segmasks)
    input_ids, labels = template.encode_oneturn(tokenizer, messages, system, tools)
    if template.efficient_eos:
        labels += [tokenizer.eos_token_id]

    input_ids, _ = template.mm_plugin.process_token_ids(input_ids, None, images, videos, tokenizer, processor)
    source_len, target_len = infer_seqlen(len(input_ids), len(labels), cutoff_len)
    input_ids = input_ids[:source_len]
    labels = labels[:target_len]
    return input_ids, labels


def preprocess_unsupervised_dataset(
        examples: Dict[str, List[Any]],
        template: "Template",
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build inputs with format `<bos> X` and labels with format `Y <eos>`
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        input_ids, labels = _encode_unsupervised_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
            audio=examples["_audio"][i] if "_audio" in examples else None,
            pc=examples["_pc"][i] if "_pc" in examples else None,
            segmasks=examples["_segmasks"][i] if "_segmasks" in examples else None,
        )
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])
        model_inputs["audio"].append(examples["_audio"][i] if "_audio" in examples else None)
        model_inputs["pc"].append(examples["_pc"][i] if "_pc" in examples else None)
        model_inputs["segmasks"].append(examples["_segmasks"][i] if "_segmasks" in examples else None)
        model_inputs["id"].append(examples["_id"][i] if "_id" in examples else None)

    return model_inputs


def print_unsupervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
