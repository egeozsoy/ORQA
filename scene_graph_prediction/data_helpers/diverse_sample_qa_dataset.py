import json
import random
from collections import defaultdict
from pathlib import Path

from scene_graph_prediction.data_helpers.generate_qa_dataset import DATASET_WEIGHTS

QUESTION_TYPE_TO_MOST_COMMON_ANSWER = {}


def compute_detailed_weights(samples):
    exact_question_freq = defaultdict(int)
    answer_freq = defaultdict(int)
    weights = []

    # Count frequency of each answer
    for sample in samples:
        sample['_norm_answer'] = sample["_answer"]
        if sample['_question_type'] in ['current_scene_graph']:  # random orders should be unrandomized for proper counting
            sg_str = sample["_answer"].replace('<SG>', '').replace('</SG>', '').strip()
            triplets_str = sg_str.split('; ')
            triplets = []
            for triplet_str in triplets_str:
                triplet = triplet_str.lower().split(',')
                if len(triplet) != 3:
                    continue
                sub, obj, pred = triplet
                sub = sub.strip()
                obj = obj.strip()
                pred = pred.strip()
                if pred not in ['closeto', 'lyingon']:
                    triplets.append(f'{sub},{obj},{pred}')
            sample['_norm_answer'] = '; '.join(sorted(triplets))
        elif sample['_question_type'] in ['list_all_entities', 'role_people', 'tools_used']:
            sample['_norm_answer'] = ', '.join(sorted(sample["_answer"].split(', ')))
        answer_freq[sample["_norm_answer"]] += 1
        exact_question_freq[sample["_question"]] += 1

    # Assign inverse frequency as weight
    for sample in samples:
        norm_answer = sample.pop('_norm_answer')
        # norm_answer = sample["_norm_answer"]
        combined_freq = answer_freq[norm_answer] * exact_question_freq[sample["_question"]]
        weight = 1 / combined_freq
        weights.append(weight)
        # delete _norm_answer
    return weights


def subsample_by_question_type(dataset_question_type_groups, per_question_type_count):
    # Subsample each question_type
    subsampled_samples = []
    for qtype, samples in dataset_question_type_groups.items():
        # Compute answer-based weights
        weights = compute_detailed_weights(samples)
        # Weighted sampling
        sampled_data = random.choices(samples, weights=weights, k=per_question_type_count)
        subsampled_samples.extend(sampled_data)

    return subsampled_samples


def main():
    # Example usage
    SPLIT = 'train'
    original_size = 100_000_000 if SPLIT == 'train' else 10_000_000
    output_size = 1_000_000 if SPLIT == 'train' else 100_000
    input_root = Path(f'data/qa_pairs_{SPLIT}_{original_size}')
    output_json = f'data/final_qa_pairs_{SPLIT}_{output_size}.json'

    chunk_files = list(input_root.glob('qa_pairs_chunk_*.json'))
    # Load all chunks
    data = []
    for chunk_file in chunk_files:
        print(f"Loading {chunk_file}")
        with open(chunk_file, 'r') as f:
            data.extend(json.load(f))
    print(f"Total samples loaded: {len(data)}")

    # Process each dataset individually, then merge.
    dataset_groups = defaultdict(list)
    for sample in data:
        dataset_groups[sample['dataset']].append(sample)

    final_samples = []
    for dataset, samples in dataset_groups.items():
        dataset_target_size = int(DATASET_WEIGHTS[dataset] * output_size)
        # Split samples by question_type
        dataset_question_type_groups = defaultdict(list)
        for sample in samples:
            dataset_question_type_groups[sample['_question_type']].append(sample)
        per_question_type_count = int(dataset_target_size / len(dataset_question_type_groups))
        print(f'Sampling {dataset} dataset with target size {dataset_target_size} and per question type count {per_question_type_count}')
        print(f'Included {len(list(dataset_question_type_groups.keys()))} Questions: {sorted(list(dataset_question_type_groups.keys()))}')
        # Subsample data
        subsampled_samples = subsample_by_question_type(dataset_question_type_groups, per_question_type_count)
        final_samples.extend(subsampled_samples)

    random.shuffle(final_samples)
    # Save to output file
    with open(output_json, 'w') as f:
        json.dump(final_samples, f, indent=4)

    print(f"Output saved to {output_json}")


if __name__ == '__main__':
    main()
