import os
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


class PDFTool:
    def __init__(self):
        self.vector_store: Optional[FAISS] = None
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self.llm: Optional[ChatOpenAI] = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize embeddings and LLM components."""
        if not self.embeddings:
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large", api_key=os.getenv("OPENAI_API_KEY")
            )

        if not self.llm:
            self.llm = ChatOpenAI(
                model="gpt-4", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
            )

    def create_vector_store(self, pdf_path: str = "./pdf_data/report_2023_2024.pdf"):
        """Create and populate the vector store with PDF content."""

        # Load and process PDF
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            all_splits = text_splitter.split_documents(docs)

            # Create FAISS vector store
            self.vector_store = FAISS.from_documents(
                documents=all_splits,
                embedding=self.embeddings
            )
            
            print(f"Successfully processed {len(all_splits)} document chunks")

        except FileNotFoundError:
            print(f"PDF file not found: {pdf_path}")
            raise
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise

    def search_text(
        self, question: str, chat_history: Optional[List[str]] = None
    ) -> Tuple[List, str]:
        """Search for relevant content and generate a response."""
        if not self.vector_store:
            raise ValueError(
                "Vector store not initialized. Call create_vector_store() first."
            )

        if not self.llm:
            raise ValueError("LLM not initialized.")

        if chat_history is None:
            chat_history = []

        # Search for relevant documents
        retrieved_docs = self.vector_store.similarity_search(question, k=4)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Create prompt template
        template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer as concise as possible.

        Context: {context}

        Chat history: {chat_history}

        Question: {question}

        Helpful Answer:"""

        custom_rag_prompt = PromptTemplate.from_template(template)
        messages = custom_rag_prompt.invoke(
            {
                "question": question,
                "context": docs_content,
                "chat_history": chat_history,
            }
        )

        response = self.llm.invoke(messages)
        return retrieved_docs, str(response.content)
