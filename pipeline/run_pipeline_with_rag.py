from typing import Dict

from dotenv import load_dotenv
import json
import os

from tqdm import tqdm

from create_dataset.data_item import DataItem
from pipeline.pipeline_steps.command_execution import execute_commands
from pipeline.pipeline_steps.command_generation import generate_commands
from pipeline.pipeline_steps.entity_extraction_with_rag import extract_entities_with_rag
from pipeline.rag.rag import SimpleRAG

load_dotenv()


def execute_inference(data_item: DataItem) -> Dict:
    """This function runs the 3-steps pipeline to produce an answer to the question in the given data item."""
    simple_rag = SimpleRAG()
    simple_rag.create_index(data_item.__dict__)

    extracted_entities = extract_entities_with_rag(
        question=data_item.question,
        rag=simple_rag
    )
    commands = generate_commands(
        extracted_entities=extracted_entities,
        question=data_item.question
    )
    output = execute_commands(
        commands=commands,
        question=data_item.question
    )

    intermediate_outputs = [step[1] for step in output['intermediate_steps']]
    actions_executed = [(step[0].tool, list(step[0].tool_input.values())) for step in output['intermediate_steps']]
    outputs = {
        "id": data_item.id,
        "extracted_entities": extracted_entities.__dict__,
        "logic_name": commands.logic_name,
        "commands": commands.operation_commands_with_filled_variables,
        "final_output": output["output"],
        "intermediate_outputs": intermediate_outputs,
        "intermediate_tools_executed": actions_executed
    }
    return outputs


if __name__ == '__main__':
    dir_path = os.getenv("DATA_DIR")
    data_items = json.load(open(os.path.join(dir_path, 'train_data_items.json')))

    all_outputs = []
    for data_item in tqdm(data_items[:1]):
        try:
            data_item = DataItem(**data_item)
            outputs = execute_inference(data_item)
            all_outputs.append(outputs)
        except Exception as e:
            print(data_item.id)
            print(f'\n Error: {e} \n\n')

    with open(f'{dir_path}/../outputs/outputs_with_rag.json', 'w') as f:
        json.dump(all_outputs, f)
