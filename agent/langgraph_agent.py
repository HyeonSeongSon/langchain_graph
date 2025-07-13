from typing import Dict, Any, List, TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import json
import asyncio
from dotenv import load_dotenv
import os
import certifi

# 환경 변수 로드
load_dotenv()

# SSL 인증서 환경변수 설정
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

# State 정의
class AgentState(TypedDict):
    query: str
    selected_tool: str
    tool_result: str
    final_response: str
    error: str

class LangGraphAgent:
    def __init__(self, tools: List[Any], llm_model: str = "gpt-3.5-turbo"):
        """LangGraph Agent를 초기화합니다."""
        try:
            self.llm = ChatOpenAI(model=llm_model, temperature=0)
            self.tools = tools
            self.tool_map = {tool.name: tool for tool in tools}
            
            # StateGraph 구성
            self.workflow = self._create_workflow()
            self.app = self.workflow.compile()
            
            print("✅ LangGraph Agent 초기화 완료")
        except Exception as e:
            import traceback
            print(f"❌ LangGraph Agent 초기화 실패: {str(e)}")
            print("상세 에러 정보:")
            traceback.print_exc()
            # 기본값 설정
            self.llm = None
            self.tools = tools
            self.tool_map = {tool.name: tool for tool in tools}
            self.app = None
    
    def _create_workflow(self) -> StateGraph:
        """StateGraph 워크플로우를 생성합니다."""
        
        # 노드 1: 쿼리 라우팅
        def route_query(state: AgentState) -> AgentState:
            """쿼리를 분석하여 적절한 도구를 선택합니다."""
            try:
                query = state["query"]
                query_lower = query.lower()
                
                # 수학 계산 관련
                if any(keyword in query_lower for keyword in ['계산', '+', '-', '*', '/', 'sqrt']):
                    selected_tool = "calculator"
                # 텍스트 처리 관련
                elif any(keyword in query_lower for keyword in ['단어', '문자', '대문자', '소문자', '텍스트', '분석']):
                    selected_tool = "text_processor"
                # 기본적으로 RAG 검색
                else:
                    selected_tool = "rag_search"
                
                return {
                    **state,
                    "selected_tool": selected_tool,
                    "error": ""
                }
            except Exception as e:
                return {
                    **state,
                    "selected_tool": "rag_search",
                    "error": f"라우팅 오류: {str(e)}"
                }
        
        # 노드 2: 도구 실행
        def execute_tool(state: AgentState) -> AgentState:
            """선택된 도구를 실행합니다."""
            try:
                selected_tool = state["selected_tool"]
                query = state["query"]
                
                if selected_tool in self.tool_map:
                    tool = self.tool_map[selected_tool]
                    tool_result = tool.run(query)
                else:
                    tool_result = f"알 수 없는 도구: {selected_tool}"
                
                return {
                    **state,
                    "tool_result": tool_result,
                    "error": ""
                }
            except Exception as e:
                return {
                    **state,
                    "tool_result": f"도구 실행 오류: {str(e)}",
                    "error": f"도구 실행 오류: {str(e)}"
                }
        
        # 노드 3: 응답 생성
        def generate_response(state: AgentState) -> AgentState:
            """LLM을 사용하여 최종 응답을 생성합니다."""
            try:
                query = state["query"]
                tool_name = state["selected_tool"]
                tool_result = state["tool_result"]
                
                # LLM이 없거나 연결 오류가 있는 경우 도구 결과를 직접 반환
                if self.llm is None:
                    final_response = tool_result
                else:
                    try:
                        prompt = ChatPromptTemplate.from_messages([
                            ("system", """당신은 도움이 되는 AI 어시스턴트입니다. 
사용자의 질문에 대해 선택된 도구의 결과를 바탕으로 친절하고 정확한 답변을 제공하세요.

사용된 도구: {tool}
도구 실행 결과: {result}

위 정보를 바탕으로 사용자의 질문에 답변해주세요."""),
                            ("user", "{query}")
                        ])
                        
                        response = self.llm.invoke(
                            prompt.format_messages(
                                tool=tool_name,
                                result=tool_result,
                                query=query
                            )
                        )
                        
                        final_response = str(response.content)
                    except Exception as llm_error:
                        # LLM 호출 실패 시 도구 결과를 직접 반환
                        print(f"⚠️ LLM 호출 실패: {str(llm_error)}")
                        final_response = f"도구 실행 결과: {tool_result}\n\n(LLM 응답 생성에 실패하여 도구 결과를 직접 제공합니다.)"
                
                return {
                    **state,
                    "final_response": final_response,
                    "error": ""
                }
            except Exception as e:
                return {
                    **state,
                    "final_response": f"응답 생성 중 오류가 발생했습니다: {str(e)}",
                    "error": f"응답 생성 오류: {str(e)}"
                }
        
        # StateGraph 구성
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("route", route_query)
        workflow.add_node("execute", execute_tool)
        workflow.add_node("generate", generate_response)
        
        # 엣지 연결
        workflow.set_entry_point("route")
        workflow.add_edge("route", "execute")
        workflow.add_edge("execute", "generate")
        workflow.add_edge("generate", END)
        
        return workflow
    
    def run(self, query: str) -> Dict[str, Any]:
        """Agent를 실행합니다."""
        try:
            if self.app is None:
                return {
                    "error": "Agent가 초기화되지 않았습니다.",
                    "query": query
                }
            
            # 초기 상태 설정
            initial_state = {
                "query": query,
                "selected_tool": "",
                "tool_result": "",
                "final_response": "",
                "error": ""
            }
            
            # 워크플로우 실행
            result = self.app.invoke(initial_state)
            
            return {
                "query": query,
                "selected_tool": result["selected_tool"],
                "tool_result": result["tool_result"],
                "final_response": result["final_response"],
                "error": result.get("error", "")
            }
            
        except Exception as e:
            return {
                "error": f"Agent 실행 오류: {str(e)}",
                "query": query
            }
    
    async def arun(self, query: str) -> Dict[str, Any]:
        """비동기로 Agent를 실행합니다."""
        try:
            if self.app is None:
                return {
                    "error": "Agent가 초기화되지 않았습니다.",
                    "query": query
                }
            
            # 초기 상태 설정
            initial_state = {
                "query": query,
                "selected_tool": "",
                "tool_result": "",
                "final_response": "",
                "error": ""
            }
            
            # 워크플로우 실행 (비동기)
            result = await self.app.ainvoke(initial_state)
            
            return {
                "query": query,
                "selected_tool": result["selected_tool"],
                "tool_result": result["tool_result"],
                "final_response": result["final_response"],
                "error": result.get("error", "")
            }
            
        except Exception as e:
            return {
                "error": f"Agent 실행 오류: {str(e)}",
                "query": query
            } 