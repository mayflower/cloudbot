from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import download_loader

directory_reader = download_loader("SimpleDirectoryReader")
loader = directory_reader("./docs", recursive=True)
raw_documents = loader.load_data()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
langchain_documents = [d.to_langchain_format() for d in raw_documents]

documents = text_splitter.split_documents(langchain_documents)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents, embedding=embeddings, persist_directory="./"
)
vectorstore.persist()
