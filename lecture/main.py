import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from dotenv import load_dotenv, find_dotenv

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
            model="llama3",
            base_url="http://localhost:11434" ,
            temperature=0,
        )
        self.agent = None

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
            print(f"Chunk {i+1}:{formatted_text}")
        return self.text_chunks

    def create_vectorstore(self):
        """
        Creates a vector store from the split text chunks using OpenAI embeddings.
        Returns the vectorstore object.
        """
        if not self.text_chunks:
            raise ValueError("No text chunks found. Please split documents before creating the vector store.")
        embeddings = OllamaEmbeddings(model="llama3")
        self.vectorstore = Chroma.from_documents(self.text_chunks, embedding=embeddings)
        print("Vectorstore created with embedded documents.")
        print("Vectorstore Contents:")
        for i, doc in enumerate(self.text_chunks):
            formatted_text = doc.page_content.replace('. ', '.')
            print(f"Document {i+1}:{formatted_text}")
            
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
        print(f"Retriever details: {self.retriever}")
        return self.retriever

    def _get_rag_context(self, query: str) -> str:
        """Retrieve and format context from the vector store."""
        retrieved_docs = self.retriever.invoke(query)
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    def create_my_agent(self):
        """Creates the agent with RAG context middleware."""
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Please call setup_retriever() first.")
        
        # Create middleware function with decorator applied
        @dynamic_prompt
        def prompt_with_context(request: ModelRequest) -> str:
            last_query = request.state["messages"][-1].text
            docs_content = self._get_rag_context(last_query)
            return (
                "You are a helpful assistant. Use the following context in your response:"
                f"\n\n{docs_content}"
            )
        
        self.agent = create_agent(self.model, tools=[], middleware=[prompt_with_context])

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
        print(f"Query: {query}Answer: {result}")
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
            print(f"Query {idx+1}: {query}Expected: {truth}Model Answer: {answer}")
            if truth.lower() in answer.lower():
                correct += 1
        accuracy = correct / len(queries)
        print(f"Evaluation Accuracy: {accuracy * 100:.2f}%")
        return accuracy


rag = RAGClass(data_path="my_text_file.txt")

# Load and process documents
rag.load_documents()
rag.split_documents()
rag.create_vectorstore()
rag.setup_retriever()
rag.create_my_agent()
rag.answer_query("What is Retrieval-Augmented Generation?")

# Evaluate the system with sample queries and ground truths
sample_queries = ["Define RAG.", "Explain vector databases."]
sample_ground_truths = ["Retrieval-Augmented Generation", "Vector databases store embeddings"]
rag.evaluate(sample_queries, sample_ground_truths)
