import uuid
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings():
    """Return a singleton embedding model instance for all document operations."""
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.EMBEDDING_BATCH_SIZE,
        },
    )


def load_and_split_pdf(pdf_path: str) -> list[Document]:
    """Load a PDF and split it into overlapping chunks while preserving page metadata."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(pages)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
        chunk.metadata.setdefault("page", 0)

    return chunks


def create_vector_store(pdf_path: str, collection_name: str = "documents"):
    """Create and persist a Chroma vector store from a PDF using smaller batches."""
    chunks = load_and_split_pdf(pdf_path)
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_DIR,
    )

    batch_size = settings.CHROMA_BATCH_SIZE
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        ids = [f"{collection_name}_{start + idx}_{uuid.uuid4().hex}" for idx in range(len(batch))]
        vectorstore.add_documents(documents=batch, ids=ids)

    return vectorstore, chunks


def get_retriever(vectorstore, k: int = 8, fetch_k: int = 20, use_multi_query: bool = True, llm=None):
    """Use MMR retrieval with optional multi-query expansion for stronger recall."""
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": 0.7},
    )

    if use_multi_query and llm is not None:
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "You are generating multiple search queries to retrieve the most relevant document passages. "
                "Create 3 short, diverse rewrites of the user's question. "
                "Each rewrite should focus on a different angle while preserving the original meaning. "
                "Return one query per line.\n\nOriginal question: {question}"
            ),
        )
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            prompt=query_prompt,
            include_original=True,
        )

    return base_retriever


def format_context_with_citations(docs: list[Document]) -> str:
    """Format retrieved chunks so the LLM can reason over them and cite page numbers."""
    formatted: list[str] = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "document")
        formatted.append(f"[Source: {source}, Page: {page + 1}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)