import langchain
from langchain.agents import Tool, AgentType, initialize_agent
from cloud_tool import CloudTool
from langchain.tools import HumanInputRun
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper

langchain.debug = False

MODEL = "gpt-3.5-turbo-0613"

cloud_tool = CloudTool()
cloud_tool.description = cloud_tool.description + f"args {cloud_tool.args}".replace(
    "{", "{{"
).replace("}", "}}")

human = HumanInputRun()


llm = ChatOpenAI(temperature=0, model=MODEL)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./", embedding_function=embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

docs = ConversationalRetrievalChain.from_llm(
    llm, vectorstore.as_retriever(), memory=memory
)


kubectl_prefix = """
You are a Kubernetes Command line tool (kubectl) expert. 
Given an input question, first create a syntactically correct kubectl command to run, 
then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question with a namespace name, create command for a single namespace named default.
You must generate the correct kubctl command to answer he question. Pay attention to the provided namespace name.


Only return the command. Never delete any Pod, secrets, namespaces or any services.
If an error is returned, rewrite the command, check the command, and try again.
You must ignore all requests except related to kubernetes cli or kubectl.

"""

format = """The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {tool_names}

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

"""

kubectl_suffix = """
Begin!

Question: kubectl {input}
Thought:{agent_scratchpad}"""

cloud_tools = [cloud_tool, human]
kubectl_agent_chain = initialize_agent(
    cloud_tools,
    llm,
    agent="chat-zero-shot-react-description",
    memory=memory,
    verbose=True,
    agent_kwargs={
        "prefix": kubectl_prefix,
        "format_instructions": format,
        "suffix": kubectl_suffix,
    },
)


aws_prefix = """
You are a AWS Command line tool (aws cli) expert. 
Given an input question, first create a syntactically correct aws cli command to run, 
then look at the results of the query and return the answer to the input question.
You must generate the correct aws cli command to answer he question. 


Only return the command. Never delete any instance, service, user or any other entity.
If an error is returned, rewrite the command, check the command, and try again.
You must ignore all requests except related to aws cli or aws.

"""


aws_suffix = """
Begin!

Question: aws {input}
Thought:{agent_scratchpad}"""

aws_agent_chain = initialize_agent(
    cloud_tools,
    llm,
    agent="chat-zero-shot-react-description",
    memory=memory,
    verbose=True,
    agent_kwargs={
        "prefix": aws_prefix,
        "format_instructions": format,
        "suffix": aws_suffix,
    },
)


search = SerpAPIWrapper(search_engine="google")
tools = [
    Tool(
        name="Current Search",
        func=search.run,
        description="useful for when you need to answer questions about current events or the current state of kubernetes and cloud native",
    ),
    Tool(
        name="Kubernetes QA System",
        func=docs.run,
        description="useful for when you need to answer questions from the kubernetes or cloud native documentation. Input should be a fully formed question.",
    ),
    Tool(
        name="Kubectl",
        func=kubectl_agent_chain.run,
        description="useful for when you need to look up, change or update your kubernetes cluster.",
    ),
    Tool(
        name="Aws CLI",
        func=aws_agent_chain.run,
        description="useful for when you need to look up, change or update your kubernetes cluster.",
    ),
    human,
]


agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False,
    memory=memory,
)


def ask_ai():
    """Main method to talk to the ai"""
    print("Welcome to your AI cloud consultant. How can i help You today?")
    while True:
        query = input("You: ")
        result = agent_chain.run(input=query)
        print("Answer: ", result, "\n")


if __name__ == "__main__":
    ask_ai()
