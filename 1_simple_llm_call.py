from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

# Groq LLM
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# Output parser
parser = StrOutputParser()

# Chain: Prompt → Model → Parser
chain = prompt | model | parser

# Run the chain
result = chain.invoke(
    {
        "question": "What is the capital of englend "
    }
)

print(result)