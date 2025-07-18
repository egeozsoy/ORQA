# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from time import time
from types import SimpleNamespace

import numpy as np
import torch
from llamafactory.data import get_template_and_fix_tokenizer, SFTDataCollatorWith4DAttentionMask
from llamafactory.model import load_tokenizer
from llamafactory.model.qwen2_vl.qwen2_vl_helpers import load_pretrained_model
from tqdm import tqdm

from scene_graph_prediction.scene_graph_helpers.model.eval_helpers import parse_and_eval_answer


class ORQAWrapperQA:
    def __init__(self, config, model_path, with_sg_grounding=False):
        self.config = config
        self.with_sg_grounding = with_sg_grounding

        self.model = load_pretrained_model(model_path)
        self.model.eval()

        # Load tokenizer and template
        model_base_name = 'Qwen/Qwen2-VL-2B-Instruct' if '2B' in model_path else 'Qwen/Qwen2-VL-7B-Instruct'

        model_args = SimpleNamespace(model_name_or_path=model_base_name, cache_dir=None, model_revision=None, hf_hub_token=None, use_fast_tokenizer=True, split_special_tokens=False,
                                     new_special_tokens=None, image_resolution=self.config['image_resolution'], video_resolution=128 * 128, video_fps=2.0, video_maxlen=64)
        data_args = SimpleNamespace(template='qwen2_vl', train_on_prompt=False, ignore_pad_token_for_loss=True, tool_format=None, fix_number_of_image_tokens=self.config['fix_number_of_image_tokens'],
                                    use_past_visual_embeds=self.config.get('use_past_visual_embeds', False))

        print(f'Fix number of image tokens: {data_args.fix_number_of_image_tokens}')
        self.model.visual.image_pooler.fix_number_of_image_tokens = data_args.fix_number_of_image_tokens
        self.model.visual.image_pooler.use_past_visual_embeds = data_args.use_past_visual_embeds
        tokenizer_module = load_tokenizer(model_args)
        tokenizer_module['tokenizer'].padding_side = 'left'  # important for generation
        self.tokenizer = tokenizer_module["tokenizer"]
        self.processor = tokenizer_module["processor"]
        self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)
        self.mm_plugin = self.template.mm_plugin

        # Set up data collator for inference (no label training needed, but we must still provide fields)
        self.data_collator = SFTDataCollatorWith4DAttentionMask(
            template=self.template,
            label_pad_token_id=-100,
            block_diag_attn=False,
            attn_implementation=self.model.config._attn_implementation,
            compute_dtype=torch.bfloat16,
            **tokenizer_module
        )

    def _get_input_features(self, batch):
        all_features = []
        for elem in batch:
            messages = elem["messages"]  # a list of dicts: role/user, content=prompt
            images = elem.get("images", [])
            videos = elem.get("videos", [])
            # Possibly collect other modalities if needed
            pc = elem['pc'] if 'pc' in elem and elem['pc'] is not None else None
            audio = elem['audio'] if 'audio' in elem and elem['audio'] is not None else None
            # segmasks = item.get('segmasks', [])
            processed_messages = self.mm_plugin.process_messages(messages, images, videos, self.processor)
            # encode
            input_ids, _ = self.template.encode_oneturn(self.tokenizer, processed_messages, system=None, tools=None)
            features = {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "images": images,
                "videos": videos,
                "pc": pc,
                "audio": audio,
                "id": elem["id"],
                # "segmasks": segmasks,
            }
            all_features.append(features)

        batch_features = self.data_collator(all_features)  # returns a batch ready for model
        # Move batch to model device if not done
        for k, v in batch_features.items():
            if isinstance(v, torch.Tensor):
                batch_features[k] = v.to(self.model.device)

        _multimodal_extras = getattr(batch_features, "_multimodal_extras", None)
        for k, v in _multimodal_extras.items():
            # new_input.data is a dict
            batch_features.data[k] = v
        return batch_features

    def forward(self, batch):

        batchsize = len(batch)
        batch_features = self._get_input_features(batch)
        # generation
        gen_kwargs = {
            "max_new_tokens": 300,
            "do_sample": False,
            "use_cache": True,
        }
        with torch.inference_mode():
            start = time()
            output_ids = self.model.generate(**batch_features, **gen_kwargs)
            time_per_sample = (time() - start) / batchsize
            num_generated_tokens = output_ids.shape[1] - batch_features.input_ids.shape[1]
            tokens_per_second = num_generated_tokens / time_per_sample
            # print(f"Tokens per second: {tokens_per_second:.2f}") # optional to count

        if batchsize == 1:
            outputs = [self.tokenizer.decode(output_ids[0, batch_features.input_ids.shape[1]:]).strip()]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, batch_features.input_ids.shape[1]:].tolist(), skip_special_tokens=True)

        # add outputs to the batch itself for easier debugging
        for idx, elem in enumerate(batch):
            elem['output'] = outputs[idx]

        return outputs

    def infer(self, dataloader):
        # return self.validate(dataloader, return_raw_predictions=True)
        return self.validate(dataloader, return_predictions_with_sample_ids=True)

    def validate(self, dataloader, limit_val_batches=None, logging_information=None, return_raw_predictions=False, return_predictions_with_sample_ids=False):
        take_success = defaultdict(list)
        dataset_success = defaultdict(list)
        qtype_success = defaultdict(list)
        sample_id_to_predictions = {}  # dictionary to store predicted scene graphs
        predictions_with_sample_ids = []
        # if limit_val_batches is int, then limit the number of batches to this number, if float, then limit the number of batches to this fraction of the total number of batches.
        limit_counter = None
        if isinstance(limit_val_batches, int):
            limit_counter = limit_val_batches
        elif isinstance(limit_val_batches, float):
            limit_counter = int(limit_val_batches * len(dataloader))
        for batch in tqdm(dataloader):
            if limit_counter is not None:
                if limit_counter <= 0:
                    break
                limit_counter -= 1

            outputs = self.forward(batch)
            for idx, elem in enumerate(batch):
                sample_id = elem['id']
                # get question_type, call corresponding function to handle it.
                question_type = elem['_question_type']
                answer_gt = elem['_answer']
                answer_pred = outputs[idx].replace('<|im_end|>', '').strip()
                parsed_pred_answer, success = parse_and_eval_answer(question_type, answer_gt, answer_pred, self.with_sg_grounding)
                take_success[elem['_take_name']].append(success)
                dataset_success[elem['_dataset_name']].append(success)
                qtype_success[question_type].append(success)
                sample_id_to_predictions[sample_id] = parsed_pred_answer
                predictions_with_sample_ids.append({'sample_id': sample_id, 'answer': parsed_pred_answer, 'question_type': question_type, 'raw_answer': answer_pred})

        if return_raw_predictions:
            return sample_id_to_predictions
        if return_predictions_with_sample_ids:
            return predictions_with_sample_ids
        self.evaluate_predictions(take_success, dataset_success, qtype_success, logging_information=logging_information)

    def evaluate_predictions(self, take_success, dataset_success, qtype_success, logging_information=None):
        print("\nSuccess by Take:")
        for take_name, success in take_success.items():
            avg_success = np.mean(success)
            print(f"{take_name}: {avg_success:.5f}")
            if logging_information is not None:
                logging_information["logger"].log_metrics({f"{take_name}": avg_success}, step=logging_information["checkpoint_id"])

        print("\nSuccess by Dataset:")
        for dataset_name, success in dataset_success.items():
            avg_success = np.mean(success)
            print(f"{dataset_name}: {avg_success:.5f}")
            if logging_information is not None:
                logging_information["logger"].log_metrics({f"{dataset_name}_{logging_information['split']}": avg_success}, step=logging_information["checkpoint_id"])

        print("\nSuccess by Question Type:")
        for qtype, success in qtype_success.items():
            avg_success = np.mean(success)
            print(f"{qtype}: {avg_success:.5f}")
            if logging_information is not None:
                logging_information["logger"].log_metrics({f"{qtype}_{logging_information['split']}": avg_success}, step=logging_information["checkpoint_id"])

        # Overall Success across all samples
        all_success = []
        for success in take_success.values():
            all_success.extend(success)
        overall_success = np.mean(all_success) if all_success else 0.0
        print(f"\nOverall Success: {overall_success:.5f}")
        if logging_information is not None:
            logging_information["logger"].log_metrics({f"Overall_{logging_information['split']}": overall_success}, step=logging_information["checkpoint_id"])
