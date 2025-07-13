from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import json

class RAGTool(BaseTool):
    name: str = "rag_search"
    description: str = "문서 데이터베이스에서 관련 정보를 검색하고 답변을 생성합니다."
    vectorstore: Any = None
    embeddings: Any = None
    documents: List[Any] = []
    
    def __init__(self):
        super().__init__()
        self._initialize_rag()
    
    def _initialize_rag(self):
        """RAG 시스템을 초기화합니다."""
        try:
            # 임베딩 모델 초기화
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # 샘플 문서 데이터
            sample_docs = [
                "인공지능(AI)은 인간의 학습능력과 추론능력, 지각능력, 자연언어의 이해능력 등을 컴퓨터 프로그램으로 실현한 기술입니다.",
                "머신러닝은 데이터로부터 패턴을 학습하여 예측이나 분류를 수행하는 AI의 한 분야입니다.",
                "딥러닝은 인공신경망을 기반으로 한 머신러닝 기법으로, 복잡한 패턴을 학습할 수 있습니다.",
                "자연어처리(NLP)는 인간의 언어를 컴퓨터가 이해하고 처리할 수 있도록 하는 기술입니다.",
                "컴퓨터 비전은 컴퓨터가 디지털 이미지나 비디오로부터 의미 있는 정보를 추출하고 이해하는 기술입니다.",
                "강화학습은 에이전트가 환경과 상호작용하며 보상을 최대화하는 방향으로 학습하는 방법입니다.",
                "LangGraph는 LangChain 기반의 복잡한 AI 애플리케이션을 구축하기 위한 프레임워크입니다.",
                "LangChain은 대규모 언어 모델을 활용한 애플리케이션 개발을 위한 프레임워크입니다.",
                "RAG(Retrieval-Augmented Generation)는 외부 지식베이스를 검색하여 더 정확한 답변을 생성하는 기술입니다.",
                "벡터 데이터베이스는 고차원 벡터를 저장하고 유사도 검색을 지원하는 데이터베이스입니다."
            ]
            
            # 문서 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            docs = []
            for i, text in enumerate(sample_docs):
                chunks = text_splitter.split_text(text)
                for chunk in chunks:
                    docs.append(Document(
                        page_content=chunk,
                        metadata={"source": f"doc_{i}", "chunk_id": len(docs)}
                    ))
            
            self.documents = docs
            
            # 벡터 스토어 생성
            if docs:
                # Windows 경로 호환성을 위해 절대 경로 사용
                import tempfile
                temp_dir = tempfile.mkdtemp()
                
                self.vectorstore = Chroma.from_documents(
                    documents=docs,
                    embedding=self.embeddings,
                    persist_directory=temp_dir
                )
                
        except Exception as e:
            print(f"RAG 초기화 오류: {str(e)}")
    
    def _run(self, query: str) -> str:
        """쿼리에 대한 관련 문서를 검색하고 답변을 생성합니다."""
        try:
            if not self.vectorstore:
                return "RAG 시스템이 초기화되지 않았습니다."
            
            # 유사도 검색
            docs = self.vectorstore.similarity_search(query, k=3)
            
            if not docs:
                return "관련된 문서를 찾을 수 없습니다."
            
            # 검색 결과 구성
            result = f"검색 쿼리: {query}\n\n관련 문서:\n"
            for i, doc in enumerate(docs, 1):
                result += f"{i}. {doc.page_content}\n\n"
            
            # 간단한 요약 생성
            all_content = " ".join([doc.page_content for doc in docs])
            summary = f"검색된 {len(docs)}개의 문서에서 관련 정보를 찾았습니다. "
            summary += "위의 문서들을 참고하여 질문에 답변하시기 바랍니다."
            
            return result + summary
            
        except Exception as e:
            return f"RAG 검색 오류: {str(e)}"
    
    def _arun(self, query: str) -> str:
        """비동기 실행 (동기와 동일)"""
        return self._run(query)

class DocumentUploadTool(BaseTool):
    name: str = "document_upload"
    description: str = "새로운 문서를 RAG 시스템에 추가합니다."
    rag_tool: Optional[RAGTool] = None
    
    def __init__(self, rag_tool: RAGTool):
        super().__init__()
        self.rag_tool = rag_tool
    
    def _run(self, document_text: str) -> str:
        """새로운 문서를 RAG 시스템에 추가합니다."""
        try:
            if not self.rag_tool or not self.rag_tool.vectorstore:
                return "RAG 시스템이 초기화되지 않았습니다."
            
            # 문서 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            chunks = text_splitter.split_text(document_text)
            docs = []
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk,
                    metadata={"source": "user_upload", "chunk_id": len(docs)}
                ))
            
            # 벡터 스토어에 추가
            self.rag_tool.vectorstore.add_documents(docs)
            
            return f"문서가 성공적으로 추가되었습니다. {len(chunks)}개의 청크로 분할되었습니다."
            
        except Exception as e:
            return f"문서 업로드 오류: {str(e)}"
    
    def _arun(self, document_text: str) -> str:
        """비동기 실행 (동기와 동일)"""
        return self._run(document_text) 