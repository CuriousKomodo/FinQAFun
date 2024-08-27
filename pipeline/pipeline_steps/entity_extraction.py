from typing import Optional, Union, List

from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
import json
import os

from create_dataset.data_item import DataItem

load_dotenv()

dir_path = os.getenv("DATA_DIR")
data_items = json.load(open(os.path.join(dir_path, 'train_data_items.json')))


class Entities(BaseModel):
    names: List[str]
    values: List[Union[float, str]]


example = data_items[0]
example_information = (f"Table pre-text: {example['pre_text']} "
                f"\n Table: {example['table']} "
                f"\n Table post-text: {example['post_text']}"
                f"\n Question: {example['question']}"
                       )
expected_entities_from_example = {
        "names": ["net cash from operating activities in 2009", "net cash from operating activities in 2008"],
        "values": ["206588", "181001"],
}

def extract_entities(data_item: DataItem) -> Entities:
    information = f"""
                    Table pre-text: {data_item.pre_text}\n
                    Table: {data_item.table}\n
                    Table post-text: {data_item.post_text}\n
                    Question: {data_item.question}\n
                 """
    system_prompt = f"""
        "You are a helpful assistant who specialises in understanding tables of financial data. 
        "You work with an accountant to calculate some important statistics about these tables.
        You task is to generate a questions requesting for the entities needed for this calculation. 
        If you can find them in the table please record the values, otherwise leave the value as <NULL>.\n
        Here's an example of your task: \n\n table information: {example_information} \n 
        Entities required: {expected_entities_from_example}
    """
    user_prompt = (
        "You are given with the following information: "
        f"\n\n table information: {information} \n "
        "Entities required:"
    )

    client = OpenAI()
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=Entities,
    )
    entities = completion.choices[0].message.parsed
    return entities


# How to evaluate this?
# Compare Answers list - If the extracted Entities values overlap with values from answers list (excluding the variables), it's seen as successful.