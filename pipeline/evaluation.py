import ast
from typing import Dict, List, Optional
import json
import os
import numpy as np
import re
import matplotlib.pyplot as plt

import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))

def _evaluate_entity_extraction(extracted_values: List[str], step_list: List[str]):
    value_retrieval_steps = [step for step in step_list if step.startswith("Ask for")]
    expected_values = [_retrieve_float_from_string(step) for step in value_retrieval_steps]
    return set(expected_values).issubset(set(extracted_values))

def _evaluate_commands(commands, step_list):
    operation_steps = [step for step in step_list if not step.startswith("Ask for")]
    return operation_steps == commands

def evaluate_tools_executed(tools_executed, step_list):
    def extract_inputs_from_step(step:str):
        expected_method = None
        for method in ['subtract', 'add', 'divide', 'multiply']:
            if method in step:
                expected_method = method

        expected_args = re.search(r'\((.+)\)', step).group(0)
        expected_args = list(ast.literal_eval(expected_args))
        return expected_method, expected_args

    operation_steps = [step for step in step_list if not step.startswith("Ask for")]

    are_methods_correct = None
    are_arguments_correct = None
    for expected_step, tool_executed in zip(operation_steps, tools_executed):
        invoked_method, invoked_args = tool_executed
        expected_method, expected_args = extract_inputs_from_step(expected_step)
        are_methods_correct = invoked_method == expected_method
        are_arguments_correct = invoked_args == expected_args
    return all([are_methods_correct, are_arguments_correct])

def _find_n_decimal_of_float_string(float_string):
    float_string = float_string.strip('%')
    decimals_string = float_string.split('.')
    if decimals_string and len(decimals_string) == 2:
        return len(decimals_string[1])
    else:
        return 0

def _retrieve_float_from_string(string) -> Optional[float]:
    float_retrieved = re.findall(r'[\d\.\d]+', string)
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


def plot_metrics(results_table: pd.DataFrame):
    entity_extraction_success_rate = sum(results_table["are_entity_values_correct"]) / len(results_table)
    commands_generation_success_rate = sum(results_table["are_commands_correct"]) / len(results_table)
    final_output_success_rate = sum(results_table["is_float_match"]) / len(results_table)
    labels = ["entity extraction", "commands generation", "final output"]
    values = [entity_extraction_success_rate, commands_generation_success_rate, final_output_success_rate]
    plt.bar(x=labels, height=values)
    plt.title("Success rate")
    plt.show()

def entity_extraction_success_rate_by_n_steps(results_table: pd.DataFrame):
    entity_extraction_success_counts = results_table[["num_steps", "are_entity_values_correct"]].groupby("num_steps").sum("are_entity_values_correct")
    entity_extraction_success_rates = entity_extraction_success_counts/results_table.groupby("num_steps").count()
    entity_extraction_success_rates.plot.bar(stacked=True)
    plt.show()


if __name__ == '__main__':
    # Have a measure for the difficulty of the challenge. I.e. via # of variables
    data_items = json.load(open(os.path.join(dir_path, '../data/train_data_items.json')))
    outputs = json.load(open(os.path.join(dir_path, '../outputs/outputs_20.json')))

    results_table = evaluate_all(outputs, data_items)
    results_table.fillna(False, inplace=True)
    results_table["num_steps"] = results_table["step_list"].apply(lambda x: len(x))
    results_table.to_csv(os.path.join(dir_path, '../outputs/metrics_table.csv'))

    plot_metrics(results_table)
    entity_extraction_success_rate_by_n_steps(results_table)