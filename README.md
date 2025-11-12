# RAG with FastAPI

간단한 **PDF 기반 RAG(검색 증강 생성)** API 서버입니다.  
지정한 폴더의 PDF를 읽어 텍스트로 변환 → 청킹 → 임베딩 → Chroma 벡터 DB에 저장하고, 질의 시 관련 문서를 검색하여 LLM이 답하도록 유도함

## 프로젝트 목적

- RAG 파이프라인의 전 과정을 직접 구축/실행해 보며 학습한다.
- PDF 폴더만 준비하면 바로 질의 응답을 시도할 수 있는 최소 구현체를 제공한다.

---

## 주요 기능

- `extract_text_from_pdfs()` : PDF 텍스트 추출 → `langchain`의 `Document` 리스트로 변환
- `chunk_documents()` : 문서들을 **chunk_size=800**, **chunk_overlap=150** 기준으로 청킹
- `build_vector_db()` : OpenAI 임베딩으로 임베딩 후 **Chroma** 벡터DB에 영구 저장(persist)
- `build_rag_chain()` : **시스템/휴먼 프롬프트**와 LLM(`gpt-4o-mini`)을 결합한 간단한 RAG 체인 구성
- FastAPI 엔드포인트
  - `GET /health` : 헬스 체크
  - `POST /ask` : 질문을 받아 답변 생성(관련 소스 문서명도 함께 반환)

---
