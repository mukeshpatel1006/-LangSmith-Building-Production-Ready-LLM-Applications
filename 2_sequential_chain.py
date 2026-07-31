from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# change project parameter 
import os 

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App '


load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)
prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

model1 = ChatGroq(
        model = "llama-3.3-70b-versatile",
        temperature=0
)

model2 = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

parser = StrOutputParser()

chain = prompt1 | model1 | parser | prompt2 | model2 | parser

# add tag and metadata 

config = {
    'run_name' : 'sequential chain ',
    'tags' : ['llm app' , 'report generation' , 'summarization'],
    'metadata' : {'model1' : "llama-3.3-70b-versatile", "model_temp" : 0, 'parser' : 'stroutputparser' }
}

result = chain.invoke({'topic': 'Unemployment in India'}, config = config)

print(result)
