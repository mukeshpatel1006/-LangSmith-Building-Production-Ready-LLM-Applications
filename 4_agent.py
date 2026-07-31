import os
import json
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate

# Fallback imports to support both legacy and classic LangChain versions
try:
    from langchain.agents import create_react_agent, AgentExecutor
except ImportError:
    from langchain_classic.agents import create_react_agent, AgentExecutor

os.environ['LANGCHAIN_PROJECT'] = 'ReAct Agent Demo'

load_dotenv()

search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """Get the current weather for a city."""

    api_key = os.getenv("WEATHERSTACK_API_KEY", "f07d9636974c4120025fadf60678771b")

    url = (
        "https://api.weatherstack.com/current"
        f"?access_key={api_key}&query={city}"
    )

    try:
        response = requests.get(url, timeout=10)
        return json.dumps(response.json(), indent=2)
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

# Step 1: Initialize ChatGroq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_retries=2,
)

# Step 2: Standard ReAct prompt defined directly (No hub dependencies)
react_prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_prompt_template)

# Step 3: Create the ReAct agent
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# Step 5: Invoke
if __name__ == "__main__":
    response = agent_executor.invoke({"input": "What is the current temp of gurgaon"})

    print("\n--- Output ---")
    print(response['output'])