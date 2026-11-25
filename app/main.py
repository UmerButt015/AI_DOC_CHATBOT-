import asyncio
import json
import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from pydantic import BaseModel

from app.config import settings
from app.memory import clear_session, get_session_history, trim_session_history
from app.prompt import get_rag_prompt
from app.rag_engine import (
    create_vector_store,
    format_context_with_citations,
    get_retriever,
)
from app.redis_utils import (
    cache_get,
    cache_set,
    enqueue_job,
    get_queue_size,
    rate_limit_check,
    set_temp_state,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------- Global state ----------
vectorstore = None
retriever = None
llm = None
rag_chain_with_history = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create directories and initialize the LLM once at startup."""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.CHROMA_DIR, exist_ok=True)

    global llm

    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is missing. Set it in app/.env or environment variables.")

    model_name = settings.GROQ_MODEL or "openai/gpt-oss-120b"
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=model_name,
        temperature=0,
        streaming=settings.STREAMING,
    )

    print(f"✅ LLM ready: {model_name}")
    yield
    print("👋 Shutting down")


app = FastAPI(
    title="Production RAG Chatbot",
    description="LangChain + FastAPI + Groq + RAG + memory + optimized retrieval",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class UploadResponse(BaseModel):
    message: str
    chunks_count: int
    collection_name: str


def build_rag_chain():
    """Build the production chat chain with prompt + LLM + output parser."""
    prompt = get_rag_prompt()
    core_chain = prompt | llm | StrOutputParser()
    return RunnableWithMessageHistory(
        core_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form("default"),
):
    """Upload a PDF, index it, and prepare the retriever with safer memory handling."""
    global vectorstore, retriever, rag_chain_with_history

    logger.info("UPLOAD: Request received")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    safe_name = os.path.basename(file.filename.replace("\\", "/"))
    safe_name = f"{uuid.uuid4().hex}_{safe_name}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_name)
    max_size = settings.MAX_UPLOAD_BYTES
    total_size = 0

    try:
        logger.info("UPLOAD: Saving %s with size limit %s bytes", file.filename, max_size)
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > max_size:
                    raise HTTPException(413, "PDF exceeds the 10 MB upload limit.")
                buffer.write(chunk)

        if total_size == 0:
            raise HTTPException(400, "Uploaded PDF is empty.")

        logger.info("UPLOAD: Saved %s bytes; starting indexing", total_size)
        enqueue_job(
            "pdf_upload",
            {
                "filename": file.filename,
                "collection_name": collection_name,
                "status": "queued",
            },
        )

        vectorstore, chunks = await asyncio.to_thread(create_vector_store, file_path, collection_name)
        logger.info("UPLOAD: Indexed %s chunks", len(chunks))

        retriever = await asyncio.to_thread(
            get_retriever,
            vectorstore,
            settings.TOP_K,
            settings.FETCH_K,
            settings.MULTI_QUERY,
            llm,
        )
        rag_chain_with_history = build_rag_chain()

        set_temp_state("documents", safe_name, {"collection_name": collection_name, "chunks": len(chunks)})

        return UploadResponse(
            message=f"✅ '{file.filename}' processed successfully.",
            chunks_count=len(chunks),
            collection_name=collection_name,
        )
    except HTTPException:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    except Exception as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.exception("UPLOAD: Processing failed")
        raise HTTPException(500, f"Processing failed: {str(exc)}") from exc


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Answer a question using RAG with conversation memory."""
    if rag_chain_with_history is None or retriever is None:
        raise HTTPException(400, "Please upload a PDF first.")

    if not rate_limit_check(f"chat:{request.session_id}"):
        raise HTTPException(429, "Too many requests. Please slow down.")

    cache_key = f"chat:{request.session_id}:{request.question.strip().lower()}"
    cached = cache_get(cache_key)
    if cached is not None:
        return ChatResponse(answer=cached["answer"], session_id=request.session_id)

    try:
        docs = await asyncio.to_thread(retriever.invoke, request.question)
        context = format_context_with_citations(docs)

        history = get_session_history(request.session_id, settings.REDIS_URL)
        trim_session_history(history, token_counter=llm, max_tokens=settings.MAX_HISTORY_TOKENS)

        answer = await rag_chain_with_history.ainvoke(
            {
                "question": request.question,
                "context": context,
            },
            config={"configurable": {"session_id": request.session_id}},
        )

        set_temp_state("chat", request.session_id, {"last_question": request.question, "last_answer": answer})
        cache_set(cache_key, {"answer": answer}, ttl=settings.CACHE_TTL)

        return ChatResponse(answer=answer, session_id=request.session_id)
    except Exception as exc:
        raise HTTPException(500, f"Chat failed: {str(exc)}") from exc


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream the answer to the client token by token."""
    if rag_chain_with_history is None or retriever is None:
        raise HTTPException(400, "Please upload a PDF first.")

    if not rate_limit_check(f"chat-stream:{request.session_id}"):
        raise HTTPException(429, "Too many requests. Please slow down.")

    async def generate():
        try:
            docs = await asyncio.to_thread(retriever.invoke, request.question)
            context = format_context_with_citations(docs)

            history = get_session_history(request.session_id, settings.REDIS_URL)
            trim_session_history(history, token_counter=llm, max_tokens=settings.MAX_HISTORY_TOKENS)

            full_answer = ""
            async for chunk in rag_chain_with_history.astream(
                {
                    "question": request.question,
                    "context": context,
                },
                config={"configurable": {"session_id": request.session_id}},
            ):
                full_answer += str(chunk)
                payload = json.dumps({"answer": chunk})
                yield f"data: {payload}\n\n"

            cache_key = f"chat:{request.session_id}:{request.question.strip().lower()}"
            cache_set(cache_key, {"answer": full_answer}, ttl=settings.CACHE_TTL)
            set_temp_state("chat", request.session_id, {"last_question": request.question, "last_answer": full_answer})

            yield "data: [DONE]\n\n"
        except Exception as exc:
            payload = json.dumps({"error": str(exc)})
            yield f"data: {payload}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/clear-session")
async def clear_chat_session(session_id: str = Form(...)):
    """Clear the session history for a user."""
    clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}


@app.get("/")
async def root():
    """Return a friendly response for the base domain."""
    return {
        "status": "ok",
        "message": "API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    from app.redis_utils import get_redis_client

    redis_ready = get_redis_client() is not None

    return {
        "status": "ok",
        "llm_model": settings.GROQ_MODEL,
        "top_k": settings.TOP_K,
        "fetch_k": settings.FETCH_K,
        "multi_query": settings.MULTI_QUERY,
        "redis_ready": redis_ready,
        "queue_size": get_queue_size(),
    }