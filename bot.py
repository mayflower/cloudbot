from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


embeddings = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./", embedding_function=embeddings)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0), vectorstore.as_retriever(), memory=memory
)


def ask_ai():
    """Main method to talk to the ai"""
    print(
        "Welcome to your AI cloud consultant. Please ask any question about kubernetes."
    )
    while True:
        query = input("Your question: ")
        result = qa({"question": query})

        print("Answer: ", result["answer"])


if __name__ == "__main__":
    ask_ai()
