from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from pipeline.pipeline_steps.command_generator import Commands
from pipeline.pipeline_steps.execution_tools import add, subtract, multiply, divide, convert_to_percentage


llm = ChatOpenAI(model="gpt-4o", temperature=0)

def execute_commands(commands: Commands, question: str):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant to an accountant. 
                Your task is to execute the given list of commands in the correct order and produce the final output.
                Note that we use 'A0' to denote the output from the first operation, and 'A1' to denote the output from the second operation.
                The same pattern applies for outputs from all further operations. 
                If the question is asking for percentage, convert the final output to percentage.
                
                Return only the final output, such as  "5%" or "71.2"
                """

            ),
            ("user", "Question: {question} \n List of commands: {commands}."),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    tools = [add, subtract, multiply, divide, convert_to_percentage]

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    output = agent_executor.invoke({
        "commands": str(commands.__dict__),
        "question": question,
    })
    return output


if __name__ == "__main__":
    commands = Commands(
        operation_commands_with_filled_variables=[
            'subtract(9362.2, 9244.9)',
            'divide(A0, 9244.9)'
        ]
    )
    output = execute_commands(
        commands=commands,
        question="what was the percent of the growth in the revenues from 2007 to 2008"
    )
    print(output)