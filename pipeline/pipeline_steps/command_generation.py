from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
import json
import os

from pipeline.pipeline_steps.entity_extraction import Entities

load_dotenv()

client = OpenAI()

knowledge_base_dir =os.getenv("KNOWLEDGE_BASE_DIR")
logic_instructions = json.load(open(os.path.join(knowledge_base_dir, 'logic_instruction.json')))

class Commands(BaseModel):
    """Output model for structured commands generation"""
    logic_name: str
    operation_commands_with_filled_variables: List[str]

def generate_commands(question: str, extracted_entities: Entities) -> Commands:
    """Generate the commands required for calculation based on question and extracted entities"""
    system_prompt = f"""You are an accountant who is calculating some financial metrics using a computer program. 
    
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
         If you do not identify the logic name, return <NULL>.
         """
    user_prompt = f"""
        Question: {question} \n
        Entities required: {str(extracted_entities.__dict__)} \n
        Commands:
        """

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format=Commands,
    )
    commands = completion.choices[0].message.parsed
    return commands


