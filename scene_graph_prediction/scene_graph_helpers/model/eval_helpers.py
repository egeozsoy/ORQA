from collections import defaultdict

import editdistance
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu


def _evaluate_count_people(gt_count, pred_count):
    """
    Success is 1.0 if off by 0, 0.5 if off by 1, 0.0 otherwise
    """
    try:  # it will be something like 1, convert it to int
        gt_count = int(gt_count.strip())
        pred_count = int(pred_count.strip())
    except Exception as e:
        print(f'Error parsing count people: {e}, gt: {gt_count}, pred: {pred_count}')
        return None, 0.0
    error = abs(gt_count - pred_count)
    success = 0.0
    if error == 0:
        success = 1.0
    elif error <= 1.1:
        success = 0.5

    return pred_count, success


def _evaluate_time_until_step(gt_time_s, pred_time_s):
    """
    More mistake is ok, if the ground truth time is big.
    If GT is 1, the model predits 10, this is not very good. IF gt is 1000, and model predicts 1010, this is much better.
    """
    try:  # it will be something like 10s, convert it to float
        gt_time_s = int(gt_time_s.replace('seconds', '').replace('s', '').strip())
        pred_time_s = int(pred_time_s.replace('seconds', '').replace('s', '').strip())
    except Exception as e:
        print(f'Error parsing time until step: {e}, gt: {gt_time_s}, pred: {pred_time_s}')
        return None, 0.0
    if gt_time_s == 0:
        # Avoid division by zero; define success based on whether prediction is also zero.
        success = 1.0 if pred_time_s == 0 else 0.0
    else:
        # Calculate relative error
        relative_error = abs(gt_time_s - pred_time_s) / gt_time_s
        # Define brackets and corresponding scores
        if relative_error < 0.10:
            success = 1.0  # Perfect
        elif relative_error < 0.25:
            success = 0.5  # Partial credit
        else:
            success = 0.0  # Incorrect

    return pred_time_s, success


def _evaluate_status_action(gt_progress_pct, pred_progress_pct):
    """
    For phase-progression in [0..100].
    Example:
      - < 10% difference => 1.0
      - < 25% difference => 0.5
      - else => 0.0
    """
    try:  # it will be something like 10%, convert it to int
        gt_progress_pct = int(gt_progress_pct.replace('%', '').strip())
        pred_progress_pct = int(pred_progress_pct.replace('%', '').strip())
    except Exception as e:
        print(f'Error parsing status action: {e}, gt: {gt_progress_pct}, pred: {pred_progress_pct}')
        return None, 0.0
    error = abs(gt_progress_pct - pred_progress_pct)
    success = 0.0
    if error < 10:
        success = 1.0
    elif error < 25:
        success = 0.5

    return pred_progress_pct, success


def _evaluate_where_2d(gt_bbox, pred_bbox):
    """
    Evaluate 2D bounding boxes via IoU → map IoU to bracket.
    gt_bbox, pred_bbox: (xmin, xmax, ymin, ymax) or (xmin, ymin, xmax, ymax).
    Here, we assume (xmin, xmax, ymin, ymax) from the user’s code.
    We'll correct to (xmin, ymin, xmax, ymax) for IoU computation.
    """
    try:  # convert comma separated strings to lists
        gt_bbox = [int(x.strip()) for x in gt_bbox.split(',')]
        pred_bbox = [int(x.strip()) for x in pred_bbox.split(',')]
        # Convert to (xmin, ymin, xmax, ymax) in case the order is different
        gt_xmin, gt_xmax, gt_ymin, gt_ymax = gt_bbox
        pd_xmin, pd_xmax, pd_ymin, pd_ymax = pred_bbox
    except Exception as e:
        print(f'Error parsing 2D bounding boxes: {e}, gt: {gt_bbox}, pred: {pred_bbox}')
        return None, 0.0

    # Ensure min < max
    gxmin, gymin = min(gt_xmin, gt_xmax), min(gt_ymin, gt_ymax)
    gxmax, gymax = max(gt_xmin, gt_xmax), max(gt_ymin, gt_ymax)
    pxmin, pymin = min(pd_xmin, pd_xmax), min(pd_ymin, pd_ymax)
    pxmax, pymax = max(pd_xmin, pd_xmax), max(pd_ymin, pd_ymax)

    # Calculate IoU
    inter_xmin = max(gxmin, pxmin)
    inter_ymin = max(gymin, pymin)
    inter_xmax = min(gxmax, pxmax)
    inter_ymax = min(gymax, pymax)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    gt_area = (gxmax - gxmin) * (gymax - gymin)
    pd_area = (pxmax - pxmin) * (pymax - pymin)
    union_area = gt_area + pd_area - inter_area
    if union_area == 0:
        success = 0.0
    else:
        iou = inter_area / union_area
        if iou >= 0.75:  # this can be considered as a perfect match
            success = 1.0
        elif iou >= 0.5:  # this is still quite good
            success = 0.75
        elif iou >= 0.25:  # this is ok
            success = 0.50
        elif iou >= 0.125:  # this is not very good
            success = 0.25
        else:
            success = 0.0
    return pred_bbox, success


def _evaluate_where_3d(gt_coord, pred_coord):
    """
    Evaluate 3D coordinate error by Euclidean distance, then bracket.
    """
    try:  # convert comma separated strings to lists
        gt_coord = [float(x.strip()) for x in gt_coord.split(',')]
        pred_coord = [float(x.strip()) for x in pred_coord.split(',')]
        gx, gy, gz = gt_coord
        px, py, pz = pred_coord
    except Exception as e:
        print(f'Error parsing 3D coordinates: {e}, gt: {gt_coord}, pred: {pred_coord}')
        return None, 0.0
    dist = math.sqrt((gx - px) ** 2 + (gy - py) ** 2 + (gz - pz) ** 2)
    success = 0.0
    if dist < 50:
        success = 1.0
    elif dist < 100:
        success = 0.75
    elif dist < 200:
        success = 0.50
    elif dist < 400:
        success = 0.25
    return pred_coord, success


def _evaluate_distance_3d(gt_dist, pred_dist):
    """
    Similar bracket approach for distance in mm.
    """
    try:  # it will be something like 1200 mm, convert it to float
        gt_dist = int(gt_dist.replace('mm', '').strip())
        pred_dist = int(pred_dist.replace('mm', '').strip())
    except Exception as e:
        print(f'Error parsing distance: {e}, gt: {gt_dist}, pred: {pred_dist}')
        return None, 0.0
    relative_error = abs(gt_dist - pred_dist) / gt_dist
    success = 0.0
    if relative_error < 0.10:
        success = 1.0
    elif relative_error < 0.25:
        success = 0.5
    elif relative_error < 0.50:
        success = 0.25

    return pred_dist, success


def _evaluate_gaze_location(gt_2d, pred_2d):
    """
    2D distance-based bracket (e.g. pixel coordinates)
    """
    try:
        gt_2d = gt_2d.split(',')
        gx, gy = gt_2d
        gx = int(gx.strip())
        gy = int(gy.strip())
        pred_2d = pred_2d.split(',')
        px, py = pred_2d
        px = int(px.strip())
        py = int(py.strip())
    except Exception as e:
        print(f'Error parsing gaze location: {e}, gt: {gt_2d}, pred: {pred_2d}')
        return None, 0.0
    dist = math.sqrt((gx - px) ** 2 + (gy - py) ** 2)
    success = 0.0
    if dist < 25:
        success = 1.0
    elif dist < 50:
        success = 0.75
    elif dist < 100:
        success = 0.50
    elif dist < 200:
        success = 0.25
    return (px, py), success


def _evaluate_set_based(gt_set, pred_set):
    try:  # convert comma separated strings to sets
        gt_set = {x.strip().lower() for x in gt_set.split(',')}
        pred_set = {x.strip().lower() for x in pred_set.split(',')}
    except Exception as e:
        print(f'Error parsing set: {e}, gt: {gt_set}, pred: {pred_set}')
        return None, 0.0

    if not gt_set and not pred_set:
        # both empty => perfect
        overlap_ratio = 1.0
    else:
        intersection = gt_set.intersection(pred_set)
        union = gt_set.union(pred_set)
        overlap_ratio = len(intersection) / len(union)
    return pred_set, overlap_ratio


def _evaluate_single_label_classification(gt_label, pred_label):
    gt_label = gt_label.lower().strip()
    pred_label = pred_label.lower().strip()
    success = 1.0 if gt_label == pred_label else 0.0
    return pred_label, success


def _evaluate_scene_graph(gt_graph, pred_graph):
    """
    We only provide a placeholder.
    You can do your own triplet overlap, recall, etc.
    For now:
    """

    try:  # both are represented first as semicolon separated triplets, and the triplets themslves are comma separated
        gt_graph = gt_graph.replace('<SG>', '').replace('</SG>', '').strip()
        pred_graph = pred_graph.replace('<SG>', '').replace('</SG>', '').strip()
        gt_triplet_strs = [x.strip() for x in gt_graph.split(';') if x.strip()]
        pred_triplet_strs = [x.strip() for x in pred_graph.split(';') if x.strip()]
        # Convert triplet strings to tuples (subject, predicate, object)
        gt_triplets = []
        for triplet_str in gt_triplet_strs:
            parts = [part.strip() for part in triplet_str.split(',')]
            if len(parts) != 3:
                continue
            gt_triplets.append(parts)

        pred_triplets = []
        for triplet_str in pred_triplet_strs:
            parts = [part.strip() for part in triplet_str.split(',')]
            if len(parts) != 3:
                continue
            pred_triplets.append(parts)
    except Exception as e:
        print(f'Error parsing scene graph: {e}, gt: {gt_graph}, pred: {pred_graph}')
        return None, 0.0
    gt_predicate_dict = defaultdict(set)  # predicate -> set of (subject, object)
    for subject, obj, predicate in gt_triplets:
        gt_predicate_dict[predicate].add((subject, obj))

    pred_predicate_dict = defaultdict(set)  # predicate -> set of (subject, object)
    for subject, obj, predicate in pred_triplets:
        pred_predicate_dict[predicate].add((subject, obj))

    predicate_recalls_dict = {}
    for predicate, gt_entities in gt_predicate_dict.items():
        pred_entities = pred_predicate_dict.get(predicate, set())
        correct_entities = gt_entities.intersection(pred_entities)
        if len(gt_entities) == 0:
            recall = 0.0  # Define recall as 0 if no ground truth entities
        else:
            recall = len(correct_entities) / len(gt_entities)
        predicate_recalls_dict[predicate] = recall

    # 7. Calculate Macro Average Recall
    if len(predicate_recalls_dict) == 0:
        macro_avg_recall = 1.0  # Both scene graphs are empty
    else:
        macro_avg_recall = sum(predicate_recalls_dict.values()) / len(predicate_recalls_dict)
    return pred_triplets, macro_avg_recall


def _evaluate_monitor_reading(gt_text, pred_text):
    """
    Classification score based on BLEU brackets.
    """
    try:
        gt_text = gt_text.lower().strip()
        pred_text = pred_text.lower().strip()
        # Tokenize the texts
        reference = [nltk.word_tokenize(gt_text.lower())]
        hypothesis = nltk.word_tokenize(pred_text.lower())
    except Exception as e:
        print(f'Error parsing monitor reading: {e}, gt: {gt_text}, pred: {pred_text}')
        return None, 0.0

    try:
        # Handle empty ground truth
        if not reference[0]:
            success = 1.0
        else:
            bleu1_score = sentence_bleu(reference, hypothesis, weights=(1, 0, 0, 0))
            success = bleu1_score
    except Exception as e:
        print(f'Error computing BLEU score: {e}, gt: {gt_text}, pred: {pred_text}')
        return None, 0.0
    return pred_text, success


def _evaluate_ordered_entities_2d(gt_list, pred_list):
    """
    Evaluate the left-to-right ordering of entities based on minimal edit distance.
    Args:
        gt_list (List[str]): Ground truth ordered list of entities.
        pred_list (List[str]): Predicted ordered list of entities.

    Returns:
        float: Classification score (1.0, 0.5, or 0.0).
    """
    try:  # convert comma separated strings to lists
        gt_list = [x.strip().lower() for x in gt_list.split(',')]
        pred_list = [x.strip().lower() for x in pred_list.split(',')]
    except Exception as e:
        print(f'Error parsing ordered entities: {e}, gt: {gt_list}, pred: {pred_list}')
        return None, 0.0
    # Handle both lists empty
    if not gt_list and not pred_list:
        success = 1.0
    else:
        # Compute Levenshtein distance directly on the lists
        distance = editdistance.eval(gt_list, pred_list)
        normalized_distance = min((distance / len(gt_list), 1.0))  # Normalize to [0, 1]
        success = 1.0 - normalized_distance
    return pred_list, success


def parse_and_eval_answer(question_type, answer_gt, answer_pred, with_sg_grounding=False):
    # if <SG> is as part of the answer because it is there as context, remove it
    if with_sg_grounding:  # remove between the first occurence of <SG> and the first occurence of <\SG>. If the question type is SGG, there will be one more <SG> and <\SG> in the answer, keep them.
        # first <SG> location
        grounding_sg_start = answer_pred.find('<SG>')
        grounding_sg_end = answer_pred.find('</SG>')
        if grounding_sg_start != -1 and grounding_sg_end != -1:  # remove everything between these two.
            answer_pred = answer_pred[:grounding_sg_start] + answer_pred[grounding_sg_end + 5:]
            answer_pred = answer_pred.strip()

    parsed_pred_answer = ''
    success = 0.0
    if question_type == 'count_people':
        parsed_pred_answer, success = _evaluate_count_people(answer_gt, answer_pred)
    elif question_type in ['role_people', 'tools_used', 'list_all_entities']:
        parsed_pred_answer, success = _evaluate_set_based(answer_gt, answer_pred)
    elif question_type in ['interaction', 'is_base_array_visible', 'is_robot_calibrated', 'sterility_breach', 'did_happen', 'tool_equipment_attribute', 'current_action', 'current_robot_step',
                           'next_robot_step', 'gaze_object']:
        parsed_pred_answer, success = _evaluate_single_label_classification(answer_gt, answer_pred)
    elif question_type == 'time_until_step':
        parsed_pred_answer, success = _evaluate_time_until_step(answer_gt, answer_pred)
    elif question_type == 'status_action':
        parsed_pred_answer, success = _evaluate_status_action(answer_gt, answer_pred)
    elif question_type == 'where_2d':
        parsed_pred_answer, success = _evaluate_where_2d(answer_gt, answer_pred)
    elif question_type == 'where_3d':
        parsed_pred_answer, success = _evaluate_where_3d(answer_gt, answer_pred)
    elif question_type == 'distance_3d':
        parsed_pred_answer, success = _evaluate_distance_3d(answer_gt, answer_pred)
    elif question_type == 'current_scene_graph':
        parsed_pred_answer, success = _evaluate_scene_graph(answer_gt, answer_pred)
    elif question_type == 'list_all_entities_ordered_2D':
        parsed_pred_answer, success = _evaluate_ordered_entities_2d(answer_gt, answer_pred)
    elif question_type == 'monitor_reading':
        parsed_pred_answer, success = _evaluate_monitor_reading(answer_gt, answer_pred)
    elif question_type == 'gaze_location':
        parsed_pred_answer, success = _evaluate_gaze_location(answer_gt, answer_pred)

    return parsed_pred_answer, success
