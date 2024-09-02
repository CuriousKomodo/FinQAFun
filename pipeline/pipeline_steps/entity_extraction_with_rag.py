import json
import os

from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from create_dataset.data_item import DataItem
from pipeline.pipeline_steps.entity_extraction import Entities
from pipeline.rag.rag import initialise_rag

load_dotenv()


simple_rag = initialise_rag()

dir_path = os.getenv("DATA_DIR")
data_items = json.load(open(os.path.join(dir_path, 'train_data_items.json')))

llm = ChatOpenAI(model="gpt-4o", temperature=0)

"""Here is a working example to be provided in the prompt for LLM"""
example = data_items[0]
example_information = f"\n Question: {example['question']}"

expected_entities_from_example = {
    "names": ["net cash from operating activities in 2009", "net cash from operating activities in 2008"],
    "values": ["206588", "181001"],
}

def extract_entities_with_rag(question: str, document_id: str, rag) -> Entities:
    """Calls LLM to extract the relevant entities from the table + context given the question"""
    rag.document_id = document_id
    @tool
    def run_rag(query):
        return simple_rag.run_query(query=query)

    system_prompt = f"""
        "You are a helpful assistant who specialises in understanding tables of financial data. 
        "You work with an accountant to calculate some important statistics about these tables.
        You task is to extract all the key entities needed for this calculation.  \n
        
        Instruction:
        - Identify a list of all the entities that are relevant for answering the question.
        - run rag to find the values of each of these entities, by asking yourself question in the format of "What is the value of <ENTITY NAME>?"
        
        If you cannot retrieve the values, fill them as <NULL>.\n
        
        Here's an example of your task: \n\n {example_information} \n 
        And the extraction should be in this format: {expected_entities_from_example} \n 

        Some context related to calculation: \n 
        - Cumulative return calculation typically requires extracting the earliest value recorded, 
        which is usually at 100%. Unless the return is calculated from a specific point in time. 
        - Calculation of a sum or an average based on a time period requires extraction of all the values recorded 
        from beginning till the end of the period. 
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "Question: {question}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    tools = [run_rag]

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True
    )
    output = agent_executor.invoke({"question": question})
    return output



if __name__ == "__main__":
    output = extract_entities_with_rag(
        question="what was the percent of the growth in the revenues from 2007 to 2008"
    )
    print(output)
