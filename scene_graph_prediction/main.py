import os
import sys
import types

from llamafactory.model.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from llamafactory.model.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

# Step 2: Create fake modules to override transformers' imports
local_qwen2_vl_processing = types.ModuleType("transformers.models.qwen2_vl.processing_qwen2_vl")
local_qwen2_vl_processing.Qwen2VLProcessor = Qwen2VLProcessor

local_qwen2_vl_image_processing = types.ModuleType("transformers.models.qwen2_vl.image_processing_qwen2_vl")
# <class 'transformers.models.qwen2_vl.image_processing_qwen2_vl.Qwen2VLImageProcessor'>
local_qwen2_vl_image_processing.Qwen2VLImageProcessor = Qwen2VLImageProcessor

# Step 3: Inject into sys.modules to override the original references
sys.modules["transformers.models.qwen2_vl.processing_qwen2_vl"] = local_qwen2_vl_processing
sys.modules["transformers.models.qwen2_vl.image_processing_qwen2_vl"] = local_qwen2_vl_image_processing

os.environ["WANDB_DIR"] = os.path.abspath("wandb")
os.environ["TMPDIR"] = os.path.abspath("wandb")

import warnings

warnings.filterwarnings('ignore')
import argparse
from pathlib import Path

import json_tricks as json  # Allows to load integers etc. correctly
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from scene_graph_prediction.scene_graph_helpers.dataset.or_dataset import ORDataset
from scene_graph_prediction.scene_graph_helpers.dataset.orqa_dataset import ORQADataset
from scene_graph_prediction.scene_graph_helpers.model.scene_graph_prediction_model_oracle import ORQAWrapperQA


def config_loader(config_path: str):
    config_path = Path('scene_graph_prediction/scene_graph_helpers/configs') / config_path
    with open(config_path, 'r') as f:
        config = json.load(f, ignore_comments=True)
    return config


def load_checkpoint_data(file_path):
    if Path(file_path).exists():
        with open(file_path, 'r') as file:
            return json.load(file)
    return {}


def update_checkpoint_data(file_path, model_name, checkpoint_id, wandb_run_id=None):
    data = load_checkpoint_data(file_path)
    if model_name not in data:
        data[model_name] = {"checkpoints": [], "wandb_run_id": wandb_run_id}
    if checkpoint_id not in data[model_name]["checkpoints"]:
        data[model_name]["checkpoints"].append(checkpoint_id)
    if wandb_run_id:
        data[model_name]["wandb_run_id"] = wandb_run_id
    with open(file_path, 'w') as file:
        json.dump(data, file)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='example.json', help='configuration file name. Relative path under given path')
    parser.add_argument('--model_path', type=str, default=None, help='path to model checkpoint')
    args = parser.parse_args()
    pl.seed_everything(42, workers=True)
    config = config_loader(args.config)
    mode = 'evaluate'  # can be evaluate/infer/eval_all/history_generation # TODO adjust as necessary
    task = 'QA'  # can be either QA or SGG
    eval_size = 'small'  # TODO can be small, medium, large
    with_sg_grounding = False
    assert task in ['QA', 'SGG']
    print(f'Running {mode} for {task}, with sg grounding: {with_sg_grounding} and eval size: {eval_size}')
    shuffle = True
    batch_size = 8

    name = args.config.replace('.json', '')

    if mode == 'evaluate':
        print(f'Model path: {args.model_path}')
        if task == 'QA':
            eval_dataset = ORQADataset(config, 'val', concise=True, with_sg_grounding=with_sg_grounding, eval_size=eval_size)
            model = ORQAWrapperQA(config, model_path=args.model_path, with_sg_grounding=with_sg_grounding)
        else:
            raise ValueError(f'Invalid task: {task}')
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=lambda x: x)
        model.validate(eval_loader)
    elif mode == 'eval_all':
        print('Evaluating all checkpoints')
        evaluated_file = f'evaluated_checkpoints_{task}.json'
        checkpoint_data = load_checkpoint_data(evaluated_file)
        model_path = Path(args.model_path)
        model_name = model_path.name
        if 'temporality' in config and config['temporality'] == 'PRED':
            print('Modifying model name for temporality')
            model_name += '_pred_temporality'
        if with_sg_grounding:
            model_name += '_sg_grounding'
        eval_every_n_checkpoints = 4
        wandb_run_id = checkpoint_data.get(model_name, {}).get("wandb_run_id", None)
        logger = pl.loggers.WandbLogger(project='orqa_evals', name=model_name, save_dir='logs', offline=False, id=wandb_run_id)
        if task == 'SGG':
            train_dataset = ORDataset(config, 'train')
            eval_dataset = ORDataset(config, 'val')
            eval_dataset_for_train = ORDataset(config, 'train')
        elif task == 'QA':
            train_dataset = ORQADataset(config, 'train', with_sg_grounding=with_sg_grounding, eval_size=eval_size)
            eval_dataset = ORQADataset(config, 'val', with_sg_grounding=with_sg_grounding, eval_size=eval_size)
            eval_dataset_for_train = ORQADataset(config, 'train', with_sg_grounding=with_sg_grounding, eval_size=eval_size)
        else:
            raise ValueError(f'Invalid task: {task}')
        # always eval last checkpoint
        checkpoints = sorted(list(model_path.glob('checkpoint-*')), key=lambda x: int(str(x).split('-')[-1]))
        # if model_name included _pkd, we will force certain checkpoints, specifically before a reduction. This is hardcoded and works only for the current settings
        if '_pkd' in model_name:
            # interesting_pkd_checkpoints = [20000, 57500, 82500] # for reducing down to 0.5
            interesting_pkd_checkpoints = [12500, 37500, 52500, 77500, 92500]  # for reducing down to 0.75
            # automatically add the last checkpoint
            interesting_pkd_checkpoints.append(int(checkpoints[-1].name.split('-')[-1]))
            print(f'PKD Detected: Forcing interesting PKD checkpoints: {interesting_pkd_checkpoints}')
            checkpoints = [c for c in checkpoints if int(c.name.split('-')[-1]) in interesting_pkd_checkpoints]
            assert len(checkpoints) == len(interesting_pkd_checkpoints)
            eval_every_n_checkpoints = 1
        print(checkpoints)
        checkpoint_idx = 0
        while checkpoint_idx < len(checkpoints):
            checkpoint = checkpoints[checkpoint_idx]
            if checkpoint_idx % eval_every_n_checkpoints != 0 and checkpoint_idx != len(checkpoints) - 1:
                print(f'Skipping checkpoint: {checkpoint}')
                checkpoint_idx += 1
                continue
            if checkpoint_idx == 0 and 'continue' not in model_name and '_pkd' not in model_name:
                print(f'Skipping checkpoint: {checkpoint}')
                checkpoint_idx += 1
                continue
            checkpoint_id = int(checkpoint.name.split('-')[-1])
            if model_name in checkpoint_data and checkpoint_id in checkpoint_data[model_name]["checkpoints"]:
                print(f'Checkpoint {checkpoint_id} for model {model_name} already evaluated. Skipping.')
                checkpoint_idx += 1
                continue
            print(f'Evaluating checkpoint: {checkpoint}...')
            torch.cuda.empty_cache()
            train_loader = DataLoader(eval_dataset_for_train, batch_size=batch_size, shuffle=shuffle, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=lambda x: x)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=lambda x: x)
            if task == 'QA':
                model = ORQAWrapperQA(config, model_path=str(checkpoint), with_sg_grounding=with_sg_grounding)
            else:
                raise ValueError(f'Invalid task: {task}')
            model.validate(train_loader, limit_val_batches=1000 // batch_size, logging_information={'split': 'train', "logger": logger, "checkpoint_id": checkpoint_id})
            model.validate(eval_loader, logging_information={'split': 'val', "logger": logger, "checkpoint_id": checkpoint_id})
            # cleanup before next run
            del model
            update_checkpoint_data(evaluated_file, model_name, checkpoint_id, logger.experiment.id)
            checkpoint_idx += 1
            checkpoints = sorted(list(model_path.glob('checkpoint-*')), key=lambda x: int(str(x).split('-')[-1]))  # update checkpoints in case new ones were added
            # if model_name included _pkd, we will force certain checkpoints, specifically before a reduction. This is hardcoded and works only for the current settings
            if '_pkd' in model_name:
                print(f'PKD Detected: Forcing interesting PKD checkpoints: {interesting_pkd_checkpoints}')
                checkpoints = [c for c in checkpoints if int(c.name.split('-')[-1]) in interesting_pkd_checkpoints]
                assert len(checkpoints) == len(interesting_pkd_checkpoints)

    elif mode == 'infer':
        print('INFER')
        print(f'Model path: {args.model_path}')
        infer_split = 'val'
        if task == 'QA':
            eval_dataset = ORQADataset(config, infer_split, concise=True, with_sg_grounding=with_sg_grounding)
            model = ORQAWrapperQA(config, model_path=args.model_path, with_sg_grounding=with_sg_grounding)
        else:
            raise ValueError(f'Invalid task: {task}')
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=lambda x: x)
        results = model.infer(eval_loader)
        # results should be batch scan id -> list of relations
        output_name = f'scan_relations_{task}_{name}_{infer_split}_concise.json'
        with open(output_name, 'w') as f:
            json.dump(results, f)
    elif mode == 'history_generation':  # we are not doing online temporality, instead we will generate the history consisting of scene graphs and actions and save it, so that we can load it
        print('Generating history')
        assert task == 'QA' and not 'temporality' in config
        model_path = Path(args.model_path)
        model_name = model_path.parent.name
        history_save_path_train = Path('data/pred_history') / f'{model_name}_train.json'
        history_save_path_val = Path('data/pred_history') / f'{model_name}_val.json'
        history_save_path_test = Path('data/pred_history') / f'{model_name}_test.json'

        train_dataset = ORQADataset(config, 'train', with_sg_grounding=False, only_history_generation=True)
        eval_dataset = ORQADataset(config, 'val', with_sg_grounding=False, only_history_generation=True)
        test_dataset = ORQADataset(config, 'test', with_sg_grounding=False, only_history_generation=True)
        model = ORQAWrapperQA(config, model_path=args.model_path, with_sg_grounding=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=lambda x: x)
        val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=lambda x: x)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=config['NUM_WORKERS'], pin_memory=True, collate_fn=lambda x: x)
        sample_id_to_predictions_train = model.validate(train_loader, return_raw_predictions=True)
        with open(history_save_path_train, 'w') as f:
            json.dump(sample_id_to_predictions_train, f)
        sample_id_to_predictions_val = model.validate(val_loader, return_raw_predictions=True)
        with open(history_save_path_val, 'w') as f:
            json.dump(sample_id_to_predictions_val, f)
        sample_id_to_predictions_test = model.validate(test_loader, return_raw_predictions=True)
        with open(history_save_path_test, 'w') as f:
            json.dump(sample_id_to_predictions_test, f)
    else:
        raise ValueError('Invalid mode')


if __name__ == '__main__':
    import subprocess

    subprocess.call(['nvidia-smi', '-L'])
    main()
