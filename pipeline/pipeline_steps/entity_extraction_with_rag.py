import json
import os

from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain_core.agents import AgentFinish, AgentActionMessageLog
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from create_dataset.data_item import DataItem
from pipeline.pipeline_steps.entity_extraction import Entities
from pipeline.rag.rag import SimpleRAG

load_dotenv()


dir_path = os.getenv("DATA_DIR")
data_items = json.load(open(os.path.join(dir_path, 'train_data_items.json')))

llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)

def parse(output):
    # If no function was invoked, return to user
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "Entities":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )

def extract_entities_with_rag(question: str, rag: SimpleRAG) -> Entities:
    """Calls LLM to extract the relevant entities from the table + context given the question"""
    @tool
    def run_rag(query):
        """Run RAG to extract the relevant entities given query"""
        return rag.run_query(query)

    llm_with_tools = llm.bind_functions([run_rag, Entities])

    system_prompt = f"""
        "You are a helpful assistant who specialises in understanding tables of financial data. 
        "You work with an accountant to calculate some important statistics about these tables.
        You task is to extract all the key entities needed for this calculation.  \n
        
        Instruction:
        - Identify a list of all the entities that are relevant for answering the question.
        - run rag to find the values of each of these entities, by querying in the format of "What is the value of ENTITY_NAME?" \n
        
        If you cannot retrieve the values, fill them as <NULL>.\n

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

    agent = (
            {
                "question": lambda x: x["question"],
                # Format agent scratchpad from intermediate steps
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | llm_with_tools
            | parse
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=[run_rag],
        verbose=True,
        return_intermediate_steps=True
    )
    output = agent_executor.invoke({"question": question})
    return output


if __name__ == "__main__":
    output = extract_entities_with_rag(
        question="what was the Net Cash from Operating Activities 2008?"
    )
    print(output)
