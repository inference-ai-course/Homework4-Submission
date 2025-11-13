import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from IPython.display import display, HTML
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from dotenv import load_dotenv, find_dotenv
import os

# Load .env from the current working directory upward and override placeholders
# _env_path = find_dotenv(usecwd=True)
# load_dotenv(_env_path, override=True)
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key or api_key.startswith("YOUR_") or api_key.strip() == "":
#     raise RuntimeError(f"OPENAI_API_KEY missing or placeholder. Ensure a valid key is set in your .env (loaded from: {_env_path or 'not found'}).")
# # Ensure downstream libs pick up the key
# os.environ["OPENAI_API_KEY"] = api_key


class RAGClass:
    def __init__(self, data_path: str):
        """
        Initialize the RAGClass with the path to the data file.
        """
        self.data_path = data_path
        self.documents = []
        self.text_chunks = []
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.model = ChatOllama(
            model="llama3.1",
            temperature=0,
        )
        self.agent = create_agent(self.model, tools=[], middleware=[self.prompt_with_context])

    def load_documents(self):
        """
        Loads documents from the specified data path and stores them in self.documents.
        Returns the loaded documents.
        """
        self.documents = TextLoader(self.data_path).load()
        # print(f"Loaded {len(self.documents)} documents.")
        return self.documents

    def split_documents(self, chunk_size=500, chunk_overlap=50):
        """
        Splits loaded documents into smaller chunks for processing.
        Returns the list of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"Split documents into {len(self.text_chunks)} chunks.")
        for i, chunk in enumerate(self.text_chunks):
            formatted_text = chunk.page_content.replace('. ', '.')
            display(HTML(f"Chunk {i+1}:{formatted_text}"))
        return self.text_chunks

    def create_vectorstore(self):
        """
        Creates a vector store from the split text chunks using OpenAI embeddings.
        Returns the vectorstore object.
        """
        if not self.text_chunks:
            raise ValueError("No text chunks found. Please split documents before creating the vector store.")
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(self.text_chunks, embedding=embeddings)
        print("Vectorstore created with embedded documents.")
        display(HTML("Vectorstore Contents:"))
        for i, doc in enumerate(self.text_chunks):
            formatted_text = doc.page_content.replace('. ', '.')
            display(HTML(f"Document {i+1}:{formatted_text}"))
        return self.vectorstore

    def setup_retriever(self):
        """
        Sets up a retriever from the vectorstore for similarity search.
        Returns the retriever object.
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")
        self.retriever = self.vectorstore.as_retriever()
        print("Retriever set up from vectorstore.")
        display(HTML(f"Retriever details: {self.retriever}"))
        return self.retriever

    @dynamic_prompt
    def prompt_with_context(self, request: ModelRequest) -> str:
        """Inject context into state messages."""
        last_query = request.state["messages"][-1].text
        retrieved_docs = self.retriever.invoke(last_query)
        
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        system_message = (
            "You are a helpful assistant. Use the following context in your response:"
            f"\n\n{docs_content}"
        )

        return system_message

    def answer_query(self, query: str):
        """
        Answers a query using the QA chain.
        Returns the answer string.
        """
        result = None
        for step in self.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            result = step["messages"][-1].content
        
        display(HTML(f"Query: {query}Answer: {result}"))
        return result

    def evaluate(self, queries: list, ground_truths: list):
        """
        Evaluates the QA system using a list of queries and ground truths.
        Returns the accuracy as a float.
        """
        if len(queries) != len(ground_truths):
            raise ValueError("Queries and ground truths must be of the same length.")
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized.")
        correct = 0
        for idx, (query, truth) in enumerate(zip(queries, ground_truths)):
            answer = self.qa_chain.run(query)
            display(HTML(f"Query {idx+1}: {query}Expected: {truth}Model Answer: {answer}"))
            if truth.lower() in answer.lower():
                correct += 1
        accuracy = correct / len(queries)
        display(HTML(f"Evaluation Accuracy: {accuracy * 100:.2f}%"))
        return accuracy