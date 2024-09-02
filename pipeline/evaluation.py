import ast
from typing import Dict, List, Optional
import json
import os
import numpy as np
import re
import matplotlib.pyplot as plt

import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))


def _retrieve_numerical_from_string(string) -> Optional[float]:
    float_retrieved = re.findall(r'[\d\.\d]+', string)
    if float_retrieved:
        return float_retrieved[0]
    else:
        return None

def evaluate_entity_extraction(extracted_values: List[str], step_list: List[str]):
    value_retrieval_steps = [step for step in step_list if step.startswith("Ask for")]
    expected_values = [_retrieve_numerical_from_string(step) for step in value_retrieval_steps]
    return set(expected_values).issubset(set(extracted_values))

def evaluate_commands(commands, step_list):
    def cleanup_arguments(args):
        cleaned_args = []
        for arg in args:
            if str(arg).startswith("A"):
                cleaned_args.append(arg)
            else:
                numerical_arg = _retrieve_numerical_from_string(str(arg))
                cleaned_args.append(numerical_arg)
        return cleaned_args
    def extract_inputs_from_step(step:str):
        tool = None
        for method in ['subtract', 'add', 'divide', 'multiply']:
            if method in step:
                tool = method

        args = re.search(r'\((.+)\)', step).group(0)
        args = list(args.replace('(','').replace(')', '').split(","))
        args = cleanup_arguments(args)
        return tool, args

    operation_steps = [step for step in step_list if not step.startswith("Ask for")]
    expected_tool_calls = [extract_inputs_from_step(step) for step in operation_steps]
    actual_commands = [extract_inputs_from_step(command) for command in commands]
    return expected_tool_calls == actual_commands

def evaluate_methods_invoked(tools_executed, step_list):
    def extract_method_from_step(step:str):
        expected_method = None
        for method in ['subtract', 'add', 'divide', 'multiply']:
            if method in step:
                expected_method = method

        # expected_args = re.search(r'\((.+)\)', step).group(0)
        # expected_args = list(ast.literal_eval(expected_args))
        return expected_method

    expected_methods = [extract_method_from_step(step) for step in step_list if not step.startswith("Ask for")]
    methods_executed = [tool[0] for tool in tools_executed if tool[0] != "convert_to_percentage"]
    return expected_methods == methods_executed

def _find_n_decimal_of_float_string(float_string):
    float_string = float_string.strip('%')
    decimals_string = float_string.split('.')
    if decimals_string and len(decimals_string) == 2:
        return len(decimals_string[1])
    else:
        return 0

def evaluate(output: Dict, data_item: Dict) -> Dict:
    try:
        answer_float = float(data_item["answer"].strip('%'))
        n_decimal = _find_n_decimal_of_float_string(data_item["answer"])

        output_float = np.round(float(output["final_output"].strip('%')), n_decimal)

        is_float_match = np.isclose(answer_float, output_float, rtol=1e-05, equal_nan=False)
        step_list = data_item["step_list"]
        are_entity_values_correct = evaluate_entity_extraction(
            extracted_values=output["extracted_entities"].get("values"),
            step_list=step_list
        )
        are_commands_correct = evaluate_commands(output["commands"], step_list)
        are_invoked_methods_correct = evaluate_methods_invoked(output["intermediate_tools_executed"], step_list)
        return {
            "is_float_match": is_float_match,
            "are_entity_values_correct": are_entity_values_correct,
            "are_commands_correct": are_commands_correct,
            "are_invoked_method_names_correct": are_invoked_methods_correct,
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
    invoked_method_names_success_rate = sum(results_table["are_invoked_method_names_correct"]) / len(results_table)
    final_output_success_rate = sum(results_table["is_float_match"]) / len(results_table)
    labels = ["entity extraction", "invoked method names", "commands generation", "final output"]
    values = [entity_extraction_success_rate, commands_generation_success_rate, invoked_method_names_success_rate,  final_output_success_rate]
    plt.barh(y=labels, width=values)
    plt.title("Success rate of the pipeline")
    plt.xlabel("success rate")

    plt.show()

def autopct_format(values):
    """This function is used to display percentages on the pie chart"""
    def my_format(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))
        return '{:.1f}%\n({v:d})'.format(pct, v=val)
    return my_format


def distribution_of_logic_names(results_table: pd.DataFrame):
    distribution = results_table["logic_name"].value_counts()
    plt.pie(distribution, labels=distribution.index, autopct=autopct_format(distribution))
    plt.title('Distribution of logic name predicted')
    plt.show()
    return

def pipeline_success_rate_by_n_steps(results_table: pd.DataFrame):
    metrics = results_table[[
        "num_steps",
        "are_entity_values_correct",
        "are_invoked_method_names_correct",
        "are_commands_correct",
        "is_float_match"
    ]]
    success_counts = metrics.groupby("num_steps").sum()
    success_rates = success_counts/metrics.groupby("num_steps").count()
    success_rates.plot.bar(stacked=False)
    plt.title("Success rate of the pipeline by number of steps")
    plt.ylabel("success rate")
    plt.show()

def are_method_name_correct_by_logic_name(results_table: pd.DataFrame):
    metrics = results_table[[
        "logic_name",
        "are_invoked_method_names_correct",
        "is_float_match"
    ]]
    success_counts = metrics.groupby("logic_name").sum()
    success_rates = success_counts/metrics.groupby("logic_name").count()
    success_rates.plot.bar(stacked=False)
    plt.title("Are invoked method names correct")
    plt.ylabel("proportion of correct method names")
    plt.show()


if __name__ == '__main__':
    # Have a measure for the difficulty of the challenge. I.e. via # of variables
    data_items = json.load(open(os.path.join(dir_path, '../data/train_data_items.json')))
    outputs = json.load(open(os.path.join(dir_path, '../outputs/outputs.json')))

    results_table = evaluate_all(outputs, data_items)
    results_table.fillna(False, inplace=True)
    results_table["num_steps"] = results_table["step_list"].apply(lambda x: len(x))
    results_table.to_csv(os.path.join(dir_path, '../outputs/metrics_table.csv'))

    # plot_metrics(results_table)
    # pipeline_success_rate_by_n_steps(results_table)
    distribution_of_logic_names(results_table)
    are_method_name_correct_by_logic_name(results_table)