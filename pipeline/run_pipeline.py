from dotenv import load_dotenv
import json
import os

from create_dataset.data_item import DataItem
from pipeline.command_executor import execute_commands
from pipeline.command_generator import generate_commands
from pipeline.entity_extraction import extract_entities

load_dotenv()


dir_path = os.path.dirname(os.path.realpath(__file__))

data_items = json.load(open(os.path.join(dir_path, '../data/train_data_items.json')))

all_outputs = []
for data_item in data_items[1:2]:
    data_item = DataItem(**data_item)
    extracted_entities = extract_entities(data_item)  # evaluate against step_list
    commands = generate_commands(
        extracted_entities=extracted_entities,
        question=data_item.question
    )  # evaluate against step_list
    output = execute_commands(
        commands=commands,
        question=data_item.question
    )  # evaluate against answer
    intermediate_outputs = [step[1] for step in output['intermediate_steps']]

    outputs = {
        "id": data_item.id,
        "extracted_entities": extracted_entities.__dict__,
        "logic_name": commands.logic_name,
        "commands": commands.operation_commands_with_filled_variables,
        "final_output": output["output"],
        "intermediate_steps": intermediate_outputs,
    }
    all_outputs.append(outputs)

with open(f'{dir_path}/../outputs/outputs.json', 'w') as f:
    json.dump(all_outputs, f)
