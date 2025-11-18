"""
ResumeBrain - RAG-Powered Resume Intelligence
Main Gradio application for resume Q&A system.
"""

import gradio as gr
import yaml
from pathlib import Path
from loguru import logger
import sys

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.document_loader import DocumentLoader
from modules.text_processor import TextProcessor
from modules.embedding_engine import EmbeddingEngine
from modules.vector_store import VectorStore
from modules.retrieval_chain import RetrievalChain


# Load configuration
def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config/config.yaml")
    if not config_path.exists():
        logger.warning("Config file not found, using defaults")
        return {}
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Initialize logger
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logger.add(
    "logs/resumebrain.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)


class ResumeBrainApp:
    """Main application class."""
    
    def __init__(self, config):
        """Initialize the application."""
        self.config = config
        
        # Initialize components
        self.doc_loader = DocumentLoader(
            max_file_size_mb=config.get('upload', {}).get('max_file_size_mb', 10),
            upload_directory=config.get('upload', {}).get('upload_directory', './data/uploads')
        )
        
        self.text_processor = TextProcessor(
            chunk_size=config.get('chunking', {}).get('chunk_size', 800),
            chunk_overlap=config.get('chunking', {}).get('chunk_overlap', 80),
            separators=config.get('chunking', {}).get('separators')
        )
        
        # Initialize embedding and vector store (will be set up when needed)
        self.embedding_engine = None
        self.vector_store = None
        self.retrieval_chain = None
        
        self.current_resume_path = None
        self.is_indexed = False
        
        logger.info("ResumeBrain application initialized")
    
    def initialize_models(self):
        """Initialize embedding and LLM models (lazy initialization)."""
        try:
            if self.embedding_engine is None:
                logger.info("Initializing embedding engine...")
                embedding_model = self.config.get('models', {}).get(
                    'embedding_model', 
                    'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
                )
                self.embedding_engine = EmbeddingEngine(model_name=embedding_model)
            
            if self.vector_store is None:
                logger.info("Initializing vector store...")
                self.vector_store = VectorStore(
                    persist_directory=self.config.get('vector_db', {}).get(
                        'persist_directory', './data/vector_db'
                    ),
                    collection_name=self.config.get('vector_db', {}).get(
                        'collection_name', 'resume_chunks'
                    )
                )
            
            return True, "Models initialized successfully"
            
        except Exception as e:
            error_msg = f"Error initializing models: {str(e)}\n\nPlease ensure Ollama is running and models are downloaded."
            logger.error(error_msg)
            return False, error_msg
    
    def process_resume(self, uploaded_file):
        """Process uploaded resume file."""
        try:
            if uploaded_file is None:
                return "❌ Error: No file uploaded", "", ""
            
            logger.info(f"Processing resume upload: {uploaded_file}")
            
            # Initialize models if not done yet
            success, message = self.initialize_models()
            if not success:
                return f"❌ {message}", "", ""
            
            # Save uploaded file
            saved_path, save_msg = self.doc_loader.save_uploaded_file(uploaded_file)
            if saved_path is None:
                return f"❌ Error saving file: {save_msg}", "", ""
            
            self.current_resume_path = saved_path
            
            # Extract text
            text, metadata, extract_msg = self.doc_loader.extract_text(saved_path)
            if text is None:
                return f"❌ Error extracting text: {extract_msg}", "", ""
            
            # Chunk text
            documents = self.text_processor.chunk_text(text, metadata)
            if not documents:
                return "❌ Error: No chunks created from document", "", ""
            
            chunk_stats = self.text_processor.get_chunk_statistics(documents)
            
            # Clear existing vector store
            self.vector_store.clear_collection()
            
            # Create vector store
            success = self.vector_store.create_vectorstore(
                documents=documents,
                embeddings=self.embedding_engine.embeddings
            )
            
            if not success:
                return "❌ Error creating vector store", "", ""
            
            # Build retrieval chain
            llm_model = self.config.get('models', {}).get(
                'llm_model',
                'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
            )
            
            self.retrieval_chain = RetrievalChain(
                vectorstore=self.vector_store,
                llm_model=llm_model,
                temperature=self.config.get('generation', {}).get('temperature', 0.1),
                max_tokens=self.config.get('generation', {}).get('max_tokens', 512),
                top_k=self.config.get('retrieval', {}).get('top_k', 3)
            )
            
            chain_success = self.retrieval_chain.build_qa_chain()
            if not chain_success:
                return "❌ Error building QA chain", "", ""
            
            self.is_indexed = True
            
            # Format success message
            status_msg = f"""✅ **Resume Processed Successfully!**
            
📄 **File:** {Path(saved_path).name}
📊 **Statistics:**
  - Pages: {metadata.get('total_pages', 'N/A')}
  - Characters: {metadata.get('total_characters', 'N/A'):,}
  - Estimated Tokens: {metadata.get('estimated_tokens', 'N/A'):,}
  - Chunks Created: {chunk_stats['total_chunks']}
  - Avg Chunk Size: {chunk_stats['avg_chunk_size']} chars
  
💾 **Vector Database:** Indexed and ready!
🤖 **Models:** Loaded and ready to answer questions!
"""
            
            # Create stats display
            stats_display = f"""**Document Statistics:**
- Total Chunks: {chunk_stats['total_chunks']}
- Avg Chunk Size: {chunk_stats['avg_chunk_size']} characters
- Min/Max Chunk Size: {chunk_stats['min_chunk_size']} / {chunk_stats['max_chunk_size']} chars
- Total Characters: {chunk_stats['total_characters']:,}
- Estimated Tokens: {chunk_stats['estimated_tokens']:,}
"""
            
            logger.info("Resume processing completed successfully")
            return status_msg, stats_display, "Ready to answer questions!"
            
        except Exception as e:
            error_msg = f"❌ Error processing resume: {str(e)}"
            logger.error(error_msg)
            return error_msg, "", ""
    
    def answer_question(self, question, history):
        """Answer a question about the resume."""
        try:
            if not self.is_indexed:
                return "⚠️ Please upload and process a resume first!"
            
            if not question or not question.strip():
                return "⚠️ Please enter a question."
            
            logger.info(f"Answering question: {question}")
            
            # Get answer with retrieval details
            result = self.retrieval_chain.query_with_retrieval_details(question)
            
            if result['error']:
                return f"❌ {result['answer']}"
            
            # Format response
            response = f"**Answer:**\n{result['answer']}\n\n"
            
            # Add sources
            if result.get('retrieval_details'):
                response += "**📚 Sources (Retrieved Chunks):**\n\n"
                for detail in result['retrieval_details']:
                    similarity = detail['similarity_score']
                    text_preview = detail['text'][:200] + "..." if len(detail['text']) > 200 else detail['text']
                    
                    response += f"**Chunk {detail['chunk_index']}** (Similarity: {similarity:.3f})\n"
                    response += f"```\n{text_preview}\n```\n\n"
            
            logger.info("Question answered successfully")
            return response
            
        except Exception as e:
            error_msg = f"❌ Error answering question: {str(e)}"
            logger.error(error_msg)
            return error_msg


def create_ui(app):
    """Create Gradio UI."""
    
    with gr.Blocks(title="ResumeBrain", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🧠 ResumeBrain - RAG-Powered Resume Intelligence
        
        Upload your resume and ask questions about it using advanced AI retrieval and generation!
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Step 1: Upload Resume")
                
                file_upload = gr.File(
                    label="Upload Resume (PDF)",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                process_btn = gr.Button("🚀 Process Resume", variant="primary", size="lg")
                
                status_output = gr.Markdown(label="Status")
                stats_output = gr.Markdown(label="Statistics")
                
            with gr.Column(scale=1):
                gr.Markdown("### 💬 Step 2: Ask Questions")
                
                chatbot = gr.Chatbot(
                    label="Q&A Chat",
                    height=400,
                    show_label=True
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="e.g., What is the resume owner's name?",
                        scale=4
                    )
                    ask_btn = gr.Button("Ask", variant="primary", scale=1)
                
                ready_status = gr.Textbox(
                    label="System Status",
                    value="⏳ Waiting for resume upload...",
                    interactive=False
                )
        
        # Example questions
        gr.Markdown("### 💡 Example Questions")
        example_questions = [
            "What is the resume owner's name?",
            "What is the domain field and how many years of experience?",
            "What major projects has the candidate contributed to?",
            "List the main technical skills.",
            "Which projects included NLP or AI?"
        ]
        
        gr.Examples(
            examples=[[q] for q in example_questions],
            inputs=[question_input]
        )
        
        # Event handlers
        def process_and_update(file):
            status, stats, ready = app.process_resume(file)
            return status, stats, ready
        
        process_btn.click(
            fn=process_and_update,
            inputs=[file_upload],
            outputs=[status_output, stats_output, ready_status]
        )
        
        def respond(message, chat_history):
            answer = app.answer_question(message, chat_history)
            chat_history.append((message, answer))
            return "", chat_history
        
        ask_btn.click(
            fn=respond,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot]
        )
        
        question_input.submit(
            fn=respond,
            inputs=[question_input, chatbot],
            outputs=[question_input, chatbot]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **🔧 Tech Stack:** LangChain • Ollama • ChromaDB • BGE Embeddings • Llama 3.2
        
        **📝 Note:** Make sure Ollama is running and required models are downloaded before processing.
        """)
    
    return interface


def main():
    """Main entry point."""
    try:
        # Load configuration
        config = load_config()
        
        # Create application
        app = ResumeBrainApp(config)
        
        # Create and launch UI
        interface = create_ui(app)
        
        logger.info("Launching Gradio interface...")
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"Error launching application: {e}")
        raise


if __name__ == "__main__":
    main()
