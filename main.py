#!/usr/bin/env python3
"""
LangGraph AI Agent 데모 시스템
- Tool 2개 (계산기 + RAG 시스템)
- Query Router
- 통합 AI Agent
"""

import asyncio
import os
from dotenv import load_dotenv
from typing import Dict, Any, List
import certifi

# 환경 변수 로드
load_dotenv()

# SSL 인증서 환경변수 설정
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# 도구들 import
from tools.basic_tools import CalculatorTool, TextProcessorTool
from tools.rag_tools import RAGTool, DocumentUploadTool

from agent.langgraph_agent import LangGraphAgent

class AIAgentDemo:
    def __init__(self):
        """AI Agent 데모 시스템을 초기화합니다."""
        self.initialize_tools()
        self.initialize_router()
        self.initialize_agent()
    
    def initialize_tools(self):
        """도구들을 초기화합니다."""
        print("🔧 도구들을 초기화하는 중...")
        
        # 기본 도구들
        self.calculator = CalculatorTool()
        self.text_processor = TextProcessorTool()
        
        # RAG 도구들
        self.rag_tool = RAGTool()
        self.document_upload = DocumentUploadTool(self.rag_tool)
        
        # 도구 리스트
        self.tools: List[Any] = [
            self.calculator,
            self.text_processor,
            self.rag_tool,
            self.document_upload
        ]
        
        print("✅ 도구 초기화 완료")
    
    def initialize_router(self):
        """쿼리 라우터를 초기화합니다."""
        print("🔄 쿼리 라우터를 초기화하는 중...")
        
        # LangGraph Agent가 내부적으로 라우팅을 처리하므로 별도 라우터 불필요
        print("✅ 라우터 초기화 완료 (LangGraph Agent가 내부적으로 처리)")
    
    def initialize_agent(self):
        """LangGraph Agent를 초기화합니다."""
        print("🤖 AI Agent를 초기화하는 중...")
        
        try:
            self.agent = LangGraphAgent(self.tools)
            print("✅ Agent 초기화 완료")
        except Exception as e:
            print(f"❌ LangGraph Agent 초기화 실패: {str(e)}")
            self.agent = None
    
    def test_router(self, query: str) -> Dict[str, Any]:
        """라우터를 테스트합니다."""
        # LangGraph Agent가 내부적으로 라우팅을 처리하므로 더미 결과 반환
        return {
            "tool": "rag_search",
            "reason": "LangGraph Agent가 내부적으로 처리",
            "confidence": 1.0
        }
    
    def test_tool(self, tool_name: str, query: str) -> str:
        """특정 도구를 테스트합니다."""
        tool_map = {
            "calculator": self.calculator,
            "text_processor": self.text_processor,
            "rag_search": self.rag_tool,
            "document_upload": self.document_upload
        }
        
        if tool_name in tool_map:
            try:
                result = tool_map[tool_name].run(query)
                return result
            except Exception as e:
                return f"도구 실행 오류: {str(e)}"
        else:
            return f"알 수 없는 도구: {tool_name}"
      
    def state_graph(self, query: str) -> Dict[str, Any]:
        """StateGraph 기반 Agent를 테스트합니다."""
        if self.agent is None:
            return {"error": "Agent가 초기화되지 않았습니다."}
        
        try:
            result = self.agent.run(query)
            print(f"🔍 StateGraph 실행 결과:")
            print(f"  - 선택된 도구: {result.get('selected_tool', 'N/A')}")
            print(f"  - 도구 결과: {result.get('tool_result', 'N/A')}")
            print(f"  - 최종 응답: {result.get('final_response', 'N/A')}")
            if result.get('error'):
                print(f"  - 오류: {result.get('error')}")
            return result
            
        except Exception as e:
            error_msg = f"StateGraph Agent 실행 오류: {str(e)}"
            return {"error": error_msg}
    
    def run_demo(self):
        """데모를 실행합니다."""
        print("🚀 LangGraph AI Agent 데모")
        
        # 테스트 쿼리들
        test_queries = [
            "2 + 3 * 4 계산해줘",
            "Hello World 텍스트를 분석해줘",
            "인공지능에 대해 알려줘",
            "머신러닝이란 무엇인가요?",
            "LangGraph에 대해 설명해줘"
        ]
        
        # StateGraph Agent 테스트
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 StateGraph 테스트 {i}: {query}")
            self.state_graph(query)
        
        print("✅ 완료!")

def main():
    """메인 함수"""
    try:
        # 데모 실행
        demo = AIAgentDemo()
        demo.run_demo()
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 