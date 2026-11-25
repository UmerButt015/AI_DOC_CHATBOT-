import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
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
    get_temp_state,
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
    if settings.PERSISTENT_DATA_DIR:
        settings.UPLOAD_DIR = os.path.join(settings.PERSISTENT_DATA_DIR, "uploads")
        settings.CHROMA_DIR = os.path.join(settings.PERSISTENT_DATA_DIR, "chroma_db")

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


class UploadJobResponse(BaseModel):
    job_id: str
    status: str
    message: str
    collection_name: str


class UploadJobStatus(BaseModel):
    job_id: str
    status: str
    filename: str
    collection_name: str
    chunks_count: int = 0
    message: str


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


async def process_pdf_job(job_id: str, file_path: str, original_filename: str, collection_name: str):
    """Index an uploaded PDF outside the HTTP request lifecycle."""
    global vectorstore, retriever, rag_chain_with_history

    set_temp_state("upload_job", job_id, {
        "job_id": job_id,
        "status": "processing",
        "filename": original_filename,
        "collection_name": collection_name,
        "chunks_count": 0,
        "message": "PDF indexing in progress.",
    })

    try:
        logger.info("UPLOAD: Job %s indexing %s", job_id, original_filename)
        vectorstore, chunks = await asyncio.to_thread(create_vector_store, file_path, collection_name)
        retriever = await asyncio.to_thread(
            get_retriever,
            vectorstore,
            settings.TOP_K,
            settings.FETCH_K,
            settings.MULTI_QUERY,
            llm,
        )
        rag_chain_with_history = build_rag_chain()

        set_temp_state("upload_job", job_id, {
            "job_id": job_id,
            "status": "completed",
            "filename": original_filename,
            "collection_name": collection_name,
            "chunks_count": len(chunks),
            "message": f"'{original_filename}' processed successfully.",
        })
        logger.info("UPLOAD: Job %s indexed %s chunks", job_id, len(chunks))
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as exc:
        logger.exception("UPLOAD: Job %s failed", job_id)
        set_temp_state("upload_job", job_id, {
            "job_id": job_id,
            "status": "failed",
            "filename": original_filename,
            "collection_name": collection_name,
            "chunks_count": 0,
            "message": f"PDF processing failed: {str(exc)}",
        })
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/upload", response_model=UploadJobResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = Form("default"),
):
    """Save a PDF immediately and process it asynchronously in a background task."""
    logger.info("UPLOAD: Request received")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    original_filename = os.path.basename(file.filename.replace("\\", "/"))
    safe_name = f"{uuid.uuid4().hex}_{original_filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_name)
    job_id = str(uuid.uuid4())
    total_size = 0

    try:
        logger.info("UPLOAD: Saving %s with size limit %s bytes", original_filename, settings.MAX_UPLOAD_BYTES)
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > settings.MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "PDF exceeds the 10 MB upload limit.")
                buffer.write(chunk)

        if total_size == 0:
            raise HTTPException(400, "Uploaded PDF is empty.")

        set_temp_state("upload_job", job_id, {
            "job_id": job_id,
            "status": "queued",
            "filename": original_filename,
            "collection_name": collection_name,
            "chunks_count": 0,
            "message": "PDF upload accepted and queued for indexing.",
        })
        enqueue_job("pdf_upload", {
            "job_id": job_id,
            "filename": original_filename,
            "collection_name": collection_name,
            "status": "queued",
        })

        background_tasks.add_task(
            process_pdf_job,
            job_id,
            file_path,
            original_filename,
            collection_name,
        )

        logger.info("UPLOAD: Job %s queued after saving %s bytes", job_id, total_size)
        return UploadJobResponse(
            job_id=job_id,
            status="queued",
            message="PDF uploaded successfully. It is being indexed in the background.",
            collection_name=collection_name,
        )
    except HTTPException:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    except Exception as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.exception("UPLOAD: File save failed")
        raise HTTPException(500, f"Upload failed: {str(exc)}") from exc


@app.get("/upload/status/{job_id}", response_model=UploadJobStatus)
async def get_upload_status(job_id: str):
    """Return the current PDF indexing status for a background job."""
    status = get_temp_state("upload_job", job_id)
    if status is None:
        raise HTTPException(404, "Upload job not found.")
    return UploadJobStatus(**status)


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