from typing import Dict, List
import json
import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

def _evaluate_entity_extraction(extracted_entities, step_list):
    return

def _evaluate_commands(commands, step_list):
    return

def evaluate(output: Dict, data_item: Dict) -> Dict:
    answer_float = float(data_item["answer"].strip('%'))
    output_float = float(output["final_output"].strip('%'))
    is_float_match = np.isclose(answer_float, output_float, rtol=1e-03, equal_nan=False)

    step_list = data_item["step_list"]
    are_entities_correct = _evaluate_entity_extraction(output["extracted_entities"], step_list)
    are_commands_correct = _evaluate_commands(output["commands"], step_list)
    return {}


def evaluate_all(outputs: List[Dict], data_items: List[Dict]) -> Dict:
    all_metrics = {}
    for data_item, output in zip(data_items, outputs):
        metrics = evaluate(output, data_item)
        metrics.update(data_item)
        all_metrics.append(metrics)
    return all_metrics


if __name__ == '__main__':
    # Have a measure for the difficulty of the challenge. I.e. via # of variables
    data_items = json.load(open(os.path.join(dir_path, '../data/train_data_items.json')))
    outputs = json.load(open(os.path.join(dir_path, '../outputs/outputs.json')))

    evaluate_all(outputs, data_items)