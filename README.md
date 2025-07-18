# ORQA: Specialized Foundation Models for Intelligent Operating Rooms

<img align="right" src="figures/teaser.jpg" alt="teaser" width="100%" style="margin-left: 10px">

Official code of the paper "Specialized Foundation Models for Intelligent Operating Rooms" (https://arxiv.org/abs/2505.12890). Surgical procedures unfold in complex environments demanding coordination
between surgical teams, tools, imaging and increasingly, intelligent robotic systems. Ensuring safety and efficiency in ORs of the future requires intelligent systems, like surgical robots, smart
instruments and digital copilots, capable of understanding complex activities and hazards of surgeries. Yet, existing computational approaches, lack the breadth, and generalization needed for
comprehensive OR understanding. We introduce ORQA, a multimodal foundation model unifying visual, auditory, and structured data for holistic surgical understanding. ORQA's question-answering framework
empowers diverse tasks, serving as an intelligence core for a broad spectrum of surgical technologies. We benchmark ORQA against generalist vision-language models, including ChatGPT and Gemini, and
show that while they struggle to perceive surgical scenes, ORQA delivers substantially stronger, consistent performance. Recognizing the extensive range of deployment settings across clinical
practice, we design, and release a family of smaller ORQA models tailored to different computational requirements. This work establishes a foundation for the next wave of intelligent surgical
solutions, enabling surgical teams and medical technology providers to create smarter and safer operating rooms.

**Authors**: [Ege Özsoy][eo], [Chantal Pellegrini][cp], [David Bani-Harouni][db], [Kun Yuan][ky], [Matthias Keicher][mk], [Nassir Navab][nassir]

[eo]:https://www.cs.cit.tum.de/camp/members/ege-oezsoy/

[cp]:https://www.cs.cit.tum.de/camp/members/chantal-pellegrini/

[db]:https://www.cs.cit.tum.de/camp/members/david-bani-harouni/

[ky]:https://scholar.google.com/citations?user=zId4EqoAAAAJ&hl=en&oi=sra

[mk]:https://www.cs.cit.tum.de/camp/members/matthias-keicher/

[nassir]:https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/

```
@article{özsoy2025specializedfoundationmodelsintelligent,
  title={Specialized Foundation Models for Intelligent Operating Rooms},
  author={{\"O}zsoy, Ege and Pellegrini, Chantal and Bani-Harouni, David and Yuan, Kun and Keicher, Matthias and Navab, Nassir},
  journal={arXiv preprint arXiv:2505.12890},
  year={2025}
}
```

## What is not included in this repository?

**4D-OR, MM-OR, EgoSurgery and MVOR are not included in this repository. Please refer to their respective repositories, and put them into the folders at the same level as this repository. The file
structure should look like this:**

- ORQA: This repository
- 4D-OR
- MM-OR
- EgoSurgery
- MVOR (We create and provide additional files in data/mvor_additional.zip, please unzip and place them in the MVOR folder)

You should also download the additional synthetic data from https://huggingface.co/egeozsoy/ORQA/resolve/main/synthetic_mv_hybridor.zip?download=true, unzip it and place it into the root folder
as `synthetic_dataset_generation/synthetic_mv_hybridor`.

## Installation

- Create an environment. Conda works great, but you can also use virtualenv or any other environment manager.
- Run `pip install -r requirements.txt`. Optionally install pytorch before this step with
  conda: `conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia`.
- Potentially you need to explicitly install flash-attn like `pip install flash-attn==2.6.1 --no-build-isolation`. flash-attn==2.6.1 works for sure.
- cd Qwen2-VL/LLaMA-Factory and then run `pip install -e ".[torch,metrics]"`
- If you want to work with point clouds, install spconv like `pip install spconv-cu117`. Similarly, you also need torch-scatter, torch-sparse, torch-cluster. With conda you can install them
  by `conda install  pytorch-scatter pytorch-sparse pytorch-cluster -c pyg`
- If scipy causes an issue, downgrade it to `scipy==1.12.0`
- Finally, you can download the model weights from https://huggingface.co/egeozsoy/ORQA/tree/main/checkpoints. Unzip it and put it in Qwen2-VL/LLaMA-Factory/saves/.
- PS: If you receive an error from the cumm library, make sure no .gitignore is in ENV/lib/python3.10/site-packages/.gitignore. For some reason it interprets this as an editable installation and
  causes problems.

## Data Generation for LVLM Training

- The ORQA dataset json is included as zip file in unzip data/final_qa_pairs.zip to data. You can skip the next two steps and jump directly to the last step, where we create our training script.
  Alternatively, you can run the following two steps to create the ORQA dataset from scratch.
- Optional: Run `python -m scene_graph_prediction.data_helpers.generate_qa_dataset` to generate the question-answering dataset. Manually change the split between train val test to generate all 3
  splits.
- Optional: Run `python -m scene_graph_prediction.data_helpers.diverse_sample_qa_dataset` to sample the dataset to a more diverse set of questions. Again, you can skip these two steps and instead just
  unzip data/final_qa_pairs.zip to data.
- To generate json for non-temporal variant, run `python -m scene_graph_prediction.data_helpers.generate_dataset_format_for_qwen2`
- To generate json for temporal variant, manually change the ADD_TEMPORAL = False to ADD_TEMPORAL = True in generate_dataset_format_for_qwen2 and then
  run `python -m scene_graph_prediction.data_helpers.generate_dataset_format_for_qwen2`.
- These steps will result in data/qwen2vl_samples/train_1000000_Falsetemp_Truetempaug_0.5mmdrop_QA_drophistory0.5.json and
  data/qwen2vl_samples/train_1000000_Truetemp_Truetempaug_0.5mmdrop_QA_drophistory0.5.json which we will use for training.

## Training

- cd into Qwen2-VL/LLaMA-Factory. We have 4 configs, qwen2vl_lora_sft_QA.yaml, qwen2vl_lora_sft_QA_temporality.yaml, qwen2vl_lora_pkd_QA_pkd.yaml, qwen2vl_lora_pkd_QA_pkd_depthreduce.yaml.
- You first need to adjust the paths in those yaml files to link to the ORQA directory. Replace /path/to/ORQADIRECTORY, with your actually path (needs to be an absolute path)
- To train base non temporal model, run `python -m src.train examples/train_qlora/qwen2vl_lora_sft_QA.yaml`.
- To train temporal model, run `python -m src.train examples/train_qlora/qwen2vl_lora_sft_QA_temporality.yaml`.
- Once you have these, you can also create the distilled versions using the following commands:
    - For the PKD model, run `python -m src.train examples/train_qlora/qwen2vl_lora_pkd_QA_pkd.yaml`
    - For the PKD depth reduced model, run `python -m src.train examples/train_qlora/qwen2vl_lora_pkd_QA_pkd_depthreduce.yaml`

## Evaluation

- To evaluate any non temporal model, such base non temporal model:
  run `python -u -m scene_graph_prediction.main --config orqa.json --model_path MODELWEIGHTPATH` e.g for base
  model `python -u -m scene_graph_prediction.main --config orqa.json --model_path Qwen2-VL/LLaMA-Factory/saves/qwen2vl_lora_sft_qlora_1000000_unfreeze8_0.5mmdrop_336res_578imgtoks/checkpoint-124806`
- To evaluate the temporal model, you first need to generate history. To this end first run, `python -m scene_graph_prediction.data_helpers.generate_temporal_history_precomputation_dataset`. Manually
  change the SPLIT between train/val/test to generate it for all SPLITS
- Then **set the mode flag in scene_graph_prediction.main to history_generation** and
  run `python -u -m scene_graph_prediction.main --config orqa.json --model_path Qwen2-VL/LLaMA-Factory/saves/qwen2vl_lora_sft_qlora_1000000_unfreeze8_0.5mmdrop_336res_578imgtoks/checkpoint-124806`
- Now you can finally run the temporal evaluation, by **setting the mode flag back to 'evaluate'** and then
  running `python -u -m scene_graph_prediction.main --config orqa_temporal_pred.json --model_path Qwen2-VL/LLaMA-Factory/saves/qwen2vl_lora_sft_qlora_1000000_unfreeze8_0.5mmdrop_336res_578imgtoks_temporality/checkpoint-124806`