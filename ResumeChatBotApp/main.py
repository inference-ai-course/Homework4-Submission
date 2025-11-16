import os
from dotenv import load_dotenv
from langchain_classic.document_loaders import PyPDFLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import FAISS
from langchain_classic.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_classic.chat_models import ChatOpenAI
# from langchain_core.vectorstores import InMemoryVectorStore

# 1. Load your documents
docs = []
for file in os.listdir("resume hub"):
    if file.endswith(".pdf"):
        resume = PyPDFLoader("./python-developer-resume-example.pdf").load()
        docs.extend(resume)
    elif file.endswith(".txt"):
        extras = TextLoader("./portfolio_notes.txt").load()
        docs.extend(extras)

# 2. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    chunks, embedding_model
)
# retriever = vectorstore.as_retriever(
#     search_type="mmr",
#     search_kwargs={
#         "k": 3,                  # Number of documents to return
#         "lambda_mult": 0.5,      # 0-1, higher values favor diversity
#         "fetch_k": 20            # Number of documents to fetch before applying MMR
#     }
# )

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3            # Number of documents to return
    }
)

# 4. Connect to your vLLM server running on port 8000 (via SSH tunnel or locally)
# llm = ChatOpenAI(
#     openai_api_base="http://localhost:8000/v1",
#     openai_api_key="not-needed",
#     model_name="Qwen/Qwen3-4B-Instruct-2507"  # matches your vLLM launch
# )

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_classic.llms import HuggingFacePipeline
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)

# 5. Retrieval QA chain
agent = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)
# IMPORTANT: using only the top-1 document by default

# 6. Ask a question
def ask_me(query): 
    print(agent.run(query))