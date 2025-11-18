"""
Retrieval Chain Module
Orchestrates the RAG pipeline using LangChain.
"""

from typing import Dict, List, Optional
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from loguru import logger


class RetrievalChain:
    """Manages the RAG retrieval and generation pipeline."""
    
    def __init__(
        self,
        vectorstore,
        llm_model: str = "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
        temperature: float = 0.1,
        max_tokens: int = 512,
        top_k: int = 5
    ):
        """
        Initialize RetrievalChain.
        
        Args:
            vectorstore: VectorStore instance
            llm_model: Name of the Ollama LLM model
            temperature: Temperature for generation (lower = more factual)
            max_tokens: Maximum tokens to generate
            top_k: Number of chunks to retrieve
        """
        self.vectorstore = vectorstore
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.qa_chain = None
        
        # Initialize LLM
        try:
            # Note: num_predict is not supported in langchain_community.llms.Ollama v0.0.13
            # The max_tokens parameter is stored but not used directly in LLM initialization
            # Output length can be controlled via prompt engineering or model configuration
            self.llm = Ollama(
                model=llm_model,
                temperature=temperature,
            )
            logger.info(f"LLM initialized: {llm_model} (max_tokens={max_tokens} configured but not enforced)")
            
            # Test LLM connection
            self._test_llm_connection()
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {e}")
            raise
    
    def _test_llm_connection(self):
        """Test connection to Ollama LLM."""
        try:
            test_response = self.llm.invoke("test")
            logger.info("LLM connection successful")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama LLM: {e}")
            raise ConnectionError(
                f"Cannot connect to Ollama LLM. Please ensure:\n"
                f"1. Ollama is installed and running\n"
                f"2. Model '{self.llm_model}' is downloaded\n"
                f"Run: ollama pull {self.llm_model}"
            )
    
    def build_qa_chain(self) -> bool:
        """
        Build the RetrievalQA chain.
        
        Returns:
            Success status
        """
        try:
            if self.vectorstore.vectorstore is None:
                logger.error("Vector store not initialized")
                return False
            
            logger.info("Building RetrievalQA chain...")
            
            # Calculate approximate character limit from max_tokens (1 token ≈ 4 characters)
            # Add some buffer for safety
            max_chars = int(self.max_tokens * 3.5)  # Conservative estimate
            
            # Create custom prompt template with output length constraint
            prompt_template = f"""You are a helpful AI assistant analyzing a resume. Use the following pieces of context from the resume to answer the question at the end. 

IMPORTANT INSTRUCTIONS:
1. Only use information from the provided context
2. If the answer is not in the context, say "I cannot find this information in the resume"
3. Be specific and cite relevant details from the context
4. Keep your answer concise and factual
5. IMPORTANT: Limit your answer to approximately {self.max_tokens} tokens (about {max_chars} characters). Be brief and to the point.

Context from resume:
{{context}}

Question: {{question}}

Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Get retriever
            retriever = self.vectorstore.get_retriever(k=self.top_k)
            
            if retriever is None:
                logger.error("Failed to get retriever from vector store")
                return False
            
            # Build RetrievalQA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Stuff all retrieved docs into prompt
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
            logger.info("RetrievalQA chain built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error building QA chain: {e}")
            return False
    
    def query(self, question: str) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with answer and source documents
        """
        try:
            if self.qa_chain is None:
                logger.error("QA chain not initialized")
                return {
                    'answer': 'Error: QA chain not initialized',
                    'source_documents': [],
                    'error': True
                }
            
            if not question or not question.strip():
                return {
                    'answer': 'Please provide a valid question',
                    'source_documents': [],
                    'error': True
                }
            
            logger.info(f"Processing query: '{question[:100]}...'")
            
            # Run the chain
            result = self.qa_chain.invoke({"query": question})
            
            # Extract answer and sources
            answer = result.get('result', 'No answer generated')
            source_docs = result.get('source_documents', [])
            
            logger.info(f"Query completed. Retrieved {len(source_docs)} source documents")
            
            return {
                'answer': answer,
                'source_documents': source_docs,
                'error': False
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f'Error processing query: {str(e)}',
                'source_documents': [],
                'error': True
            }
    
    def query_with_retrieval_details(self, question: str) -> Dict:
        """
        Query with detailed retrieval information (similarity scores).
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with answer, sources, and retrieval details
        """
        try:
            # First get retrieval results with scores
            retrieval_results = self.vectorstore.similarity_search(
                query=question,
                k=self.top_k
            )
            
            # Then get the answer
            result = self.query(question)
            
            # Add retrieval details
            retrieval_details = []
            for i, (doc, score) in enumerate(retrieval_results):
                retrieval_details.append({
                    'chunk_index': i + 1,
                    'text': doc.page_content,
                    'similarity_score': float(score),
                    'metadata': doc.metadata
                })
            
            result['retrieval_details'] = retrieval_details
            
            return result
            
        except Exception as e:
            logger.error(f"Error in detailed query: {e}")
            return {
                'answer': f'Error: {str(e)}',
                'source_documents': [],
                'retrieval_details': [],
                'error': True
            }
    
    def format_sources(self, source_documents: List) -> str:
        """
        Format source documents for display.
        
        Args:
            source_documents: List of source Document objects
            
        Returns:
            Formatted string of sources
        """
        if not source_documents:
            return "No sources found"
        
        formatted = []
        for i, doc in enumerate(source_documents, 1):
            metadata = doc.metadata
            text_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            
            source_info = f"**Source {i}:**\n"
            
            if 'page_num' in metadata:
                source_info += f"- Page: {metadata['page_num']}\n"
            if 'chunk_index' in metadata:
                source_info += f"- Chunk: {metadata['chunk_index'] + 1}\n"
            
            source_info += f"- Text: \"{text_preview}\"\n"
            
            formatted.append(source_info)
        
        return "\n".join(formatted)
