import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import fitz

# =========================
# 환경/기본 설정
# =========================
load_dotenv()
PDF_FOLDER = os.getenv("PDF_FOLDER", "./pdfs")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./vector_store")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 전역 상태(앱 수명주기 동안 1회 로딩)
_db = None
_retriever = None
_chain = None


# =========================
# 기존 로직 (조금만 정리)
# =========================
def extract_text_from_pdfs(pdf_folder: str) -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(pdf_folder):
        print(f"PDF 폴더가 없습니다: {pdf_folder}")
        return docs

    for file in os.listdir(pdf_folder):
        if file.lower().endswith(".pdf"):
            path = os.path.join(pdf_folder, file)
            print(f"Loading: {file}")
            try:
                with fitz.open(path) as pdf:
                    text = ""
                    for page in pdf:
                        text += page.get_text("text")
                    if text.strip():
                        docs.append(Document(
                            page_content=text.strip(),
                            metadata={"source": file}
                        ))
            except Exception as e:
                print(f"Error loading {file}: {e}")

    print(f"\n총 {len(docs)}개의 문서가 로드되었습니다.")
    return docs


def chunk_documents(docs, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    splits = splitter.split_documents(docs)
    print(f"총 {len(splits)}개의 청크가 생성되었습니다.")
    return splits


def build_vector_db(splits, persist_dir: str):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY가 .env에 설정되어 있지 않습니다.")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    db = Chroma.from_documents(
        splits,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    db.persist()
    print(f"Chroma 벡터DB 생성 완료 → {persist_dir}")
    return db


def load_or_create_db(pdf_folder: str, persist_dir: str):
    """
    persist_dir에 기존 DB가 있으면 불러오고, 없으면 새로 생성.
    """
    global _db
    if os.path.isdir(persist_dir) and os.listdir(persist_dir):
        # 기존 DB 로드
        print("기존 Chroma DB 로드 중...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
        _db = Chroma(
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        print("기존 DB 로드 완료")
    else:
        # 새로 생성
        print("DB가 없어 새로 생성합니다...")
        docs = extract_text_from_pdfs(pdf_folder)
        if not docs:
            raise RuntimeError("PDF에서 문서를 로드하지 못해 DB를 생성할 수 없습니다.")
        splits = chunk_documents(docs)
        _db = build_vector_db(splits, persist_dir)
    return _db


def format_docs(docs: List[Document]) -> str:
    merged = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        merged.append(f"[{src}]\n{d.page_content}")
    return "\n\n---\n\n".join(merged)


def build_rag_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "너는 고객지원 RAG 어시스턴트야. 아래 '컨텍스트'만 근거로 한국어로 간결하게 답해. "
         "모르면 모른다고 말하고 추측하지 마. 중요한 수치/조건은 그대로 유지하고, "
         "끝에 간단히 근거 문서명을 나열해."),
        ("human",
         "질문: {question}\n\n컨텍스트:\n{context}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=OPENAI_API_KEY
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask_once(chain, retriever, question: str):
    answer = chain.invoke(question)
    top_docs = retriever.get_relevant_documents(question)
    sources = [d.metadata.get("source", "unknown") for d in top_docs]
    return answer, sources


# =========================
# FastAPI 스키마
# =========================
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


class IngestRequest(BaseModel):
    pdf_folder: Optional[str] = None
    chunk_size: Optional[int] = 800
    chunk_overlap: Optional[int] = 150


class IngestResponse(BaseModel):
    message: str
    num_docs: int
    num_chunks: int


# =========================
# FastAPI 앱
# =========================
app = FastAPI(title="RAG with FastAPI", version="1.0.0")


@app.on_event("startup")
def on_startup():
    """
    서버 시작 시 1) DB 로드/생성 2) retriever/chain 준비
    """
    global _db, _retriever, _chain

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY가 없습니다. .env에 설정하세요.")

    _db = load_or_create_db(PDF_FOLDER, PERSIST_DIR)
    _retriever = _db.as_retriever(search_kwargs={"k": 3})
    _chain = build_rag_chain(_retriever)
    print("RAG 앱 준비 완료")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    global _chain, _retriever
    if _chain is None or _retriever is None:
        raise HTTPException(status_code=503, detail="RAG가 아직 초기화되지 않았습니다.")
    try:
        answer, sources = ask_once(_chain, _retriever, req.question)
        return AskResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))