from typing import Dict, List, Optional
import json
import os
import numpy as np
import re

import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))

def _evaluate_entity_extraction(extracted_values: List[str], step_list: List[str]):
    value_retrieval_steps = [step for step in step_list if step.startswith("Ask for")]
    expected_values = [_retrieve_float_from_string(step) for step in value_retrieval_steps]
    return set(extracted_values) == set(expected_values)

def _evaluate_commands(commands, step_list):
    operation_steps = [step for step in step_list if not step.startswith("Ask for")]
    return operation_steps == commands

def _find_n_decimal_of_float_string(float_string):
    float_string = float_string.strip('%')
    decimals_string = float_string.split('.')
    if decimals_string and len(decimals_string) == 2:
        return len(decimals_string[1])
    else:
        return 0

def _retrieve_float_from_string(string) -> Optional[float]:
    float_retrieved = re.findall("\d+\.\d+", string)
    if float_retrieved:
        return float_retrieved[0]
    else:
        return None

def evaluate(output: Dict, data_item: Dict) -> Dict:
    try:
        answer_float = float(data_item["answer"].strip('%'))
        n_decimal = _find_n_decimal_of_float_string(data_item["answer"])

        output_float = np.round(float(output["final_output"].strip('%')), n_decimal)

        is_float_match = np.isclose(answer_float, output_float, rtol=1e-05, equal_nan=False)
        step_list = data_item["step_list"]
        are_entity_values_correct = _evaluate_entity_extraction(
            extracted_values=output["extracted_entities"].get("values"),
            step_list=step_list
        )
        are_commands_correct = _evaluate_commands(output["commands"], step_list)
        return {
            "is_float_match": is_float_match,
            "are_entity_values_correct": are_entity_values_correct,
            "are_commands_correct": are_commands_correct
        }
    except Exception as e:
        print(f"Cannot evaluate {data_item['id']} due to error: {e} \n")
        print(f"output: \n {output}")
        return {}


def evaluate_all(outputs: List[Dict], data_items: List[Dict]) -> pd.DataFrame:
    all_metrics = []
    for data_item, output in zip(data_items, outputs):
        metrics = evaluate(output, data_item)
        metrics.update(output)
        metrics.update(data_item)
        all_metrics.append(metrics)

    metrics_table = pd.DataFrame.from_records(all_metrics)
    return metrics_table


if __name__ == '__main__':
    # Have a measure for the difficulty of the challenge. I.e. via # of variables
    data_items = json.load(open(os.path.join(dir_path, '../data/train_data_items.json')))
    outputs = json.load(open(os.path.join(dir_path, '../outputs/outputs.json')))

    metrics_table = evaluate_all(outputs, data_items)
    metrics_table.fillna(False, inplace=True)
    metrics_table.to_csv(os.path.join(dir_path, '../outputs/metrics_table.csv'))