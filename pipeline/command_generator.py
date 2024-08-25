from typing import Optional, Union, List

from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
import json
import os

load_dotenv()

client = OpenAI()


dir_path = os.path.dirname(os.path.realpath(__file__))

class Commands(BaseModel):
    operation_commands_with_filled_variables: List[str]


# Have a measure for the difficulty of the challenge. I.e. via # of variables
data_items = json.load(open(os.path.join(dir_path, '../data/train_data_items.json')))
logic_instructions = json.load(open(os.path.join(dir_path, '../knowledge_base/logic_instruction.json')))

information = (
    f"Table pre-text: {data_items[1]['pre_text']} "
    f"\n Table: {data_items[1]['table']} "
    f"\n Table post-text: {data_items[1]['post_text']}"
)

extracted_entities = {
    'names': ['revenue in 2008', 'revenue in 2007'],
    'values': ['9362.2', '9244.9']
                      }

system_prompt = (
    f"""You are an accountant who is calculating some financial metrics using a computer program. 

    Given any question, you need to: \n
     - identify the name of logic or logics required for the question, by looking for key words in the question.
     - find the corresponding operation commands. 
     - identify which values from the given entities that you should use, and fill the variables in the operation commands by these values.
     - the variables that you will replace are: "X", "Y", "*kwargs" (a list of values) and "constant". 
     - produce a list of operational commands with filled variables for the computer program
     \n
     
     Logic instruction: {logic_instructions} \n
     Note that we use 'A0' to denote the output from the first operation, and 'A1' to denote the output from the second operation.
     The same pattern applies for outputs from all further operations. 
     Note that there is no need to replace these variables in the operation commands. \n
     """
)
user_prompt = (
    f"Question: {data_items[1]['question']} \n"
    f"Entities required: {extracted_entities} \n "
    "Commands:"
)

completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_format=Commands,
)
event = completion.choices[0].message.parsed



