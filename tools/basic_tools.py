from typing import Dict, Any
from langchain.tools import BaseTool
import re
import ast

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "수학 계산을 수행합니다. 예: 2 + 3, 10 * 5, sqrt(16)"
    
    def __init__(self):
        super().__init__()
    
    def _run(self, query: str) -> str:
        """수학 표현식을 계산합니다."""
        try:
            # 수학 표현식 추출
            import re
            # 더 정확한 수학 표현식 패턴
            math_pattern = r'(\d+\s*[\+\-\*\/]\s*\d+)'
            math_match = re.search(math_pattern, query)
            
            if not math_match:
                return "오류: 수학 표현식을 찾을 수 없습니다. 예: 2 + 3, 10 * 5"
            
            expression = math_match.group(1).replace(' ', '')  # 공백 제거
            
            # 안전한 수학 표현식만 허용
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return "오류: 허용되지 않는 문자가 포함되어 있습니다."
            
            # 표현식 평가
            result = eval(expression)
            return f"계산 결과: {expression} = {result}"
        except Exception as e:
            return f"계산 오류: {str(e)}"
    
    def _arun(self, query: str) -> str:
        """비동기 실행 (동기와 동일)"""
        return self._run(query)

class TextProcessorTool(BaseTool):
    name: str = "text_processor"
    description: str = "텍스트 처리를 수행합니다. 단어 수 세기, 대소문자 변환 등"
    
    def __init__(self):
        super().__init__()
    
    def _run(self, query: str) -> str:
        """텍스트 처리 작업을 수행합니다."""
        try:
            # 간단한 텍스트 분석
            word_count = len(query.split())
            char_count = len(query)
            upper_count = sum(1 for c in query if c.isupper())
            lower_count = sum(1 for c in query if c.islower())
            
            return f"""텍스트 분석 결과:
- 단어 수: {word_count}
- 문자 수: {char_count}
- 대문자 수: {upper_count}
- 소문자 수: {lower_count}
- 대문자로 변환: {query.upper()}
- 소문자로 변환: {query.lower()}"""
        except Exception as e:
            return f"텍스트 처리 오류: {str(e)}"
    
    def _arun(self, query: str) -> str:
        """비동기 실행 (동기와 동일)"""
        return self._run(query) 