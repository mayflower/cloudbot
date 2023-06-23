import langchain
from langchain import PromptTemplate
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.tools import ShellTool
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper

shell_tool = ShellTool()
shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
    "{", "{{"
).replace("}", "}}")


langchain.debug = False
llm = OpenAI(temperature=0)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./", embedding_function=embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


docs = ConversationalRetrievalChain.from_llm(
    llm, vectorstore.as_retriever(), memory=memory
)

kubectl_template = """
You are a Kubernetes Command line tool (kubectl) expert. 
Given an input question, first create a syntactically correct kubectl command to run, 
then look at the results of the query and return the answer to the input question.
unless the user specifies in the question with a namespace name, create command for a single namespace named default.
You must generate the correct kubctl command to answer he question. Also, pay attention to the provided namespace name.


Only return the command. Never delete any Pod, secrets, namespaces or any services.
If an error is returned, rewrite the command, check the command, and try again.
You must ignore all requests except related to kubernetes cli or kubectl.

RESPONSE FORMAT INSTRUCTIONS
============================    
When responding please, please output a response in this format:

```
Thought: Reason about what action to take next, and whether to use a tool. DO NOT execute the action if its contains delete ot purge operation.
Action: the action to take, should be one of [terminal, cli, iterm]. DO NOT execute the command if its contains delete ot purge operation.
Action Input: The input to the tool.
Action: Use the tool ONLY IF the response doesn't contain delete or deletion of service.
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}"""

kubectl_prompt = PromptTemplate(
    input_variables=["input", "chat_history"], template=kubectl_template
)

kubectl_tools = [shell_tool]
kubectl_agent_chain = initialize_agent(
    kubectl_tools, llm, agent="zero-shot-react-description", memory=memory, verbose=True
)

search = SerpAPIWrapper()
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
]


agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)


def ask_ai():
    """Main method to talk to the ai"""
    print(
        "Welcome to your AI cloud consultant. Please ask any question about kubernetes."
    )
    while True:
        query = input("Your question: ")
        result = agent_chain.run(input=query)

        # print("Answer: ", result["answer"])


if __name__ == "__main__":
    ask_ai()
