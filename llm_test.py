import os
import certifi

# SSL 인증서 환경변수 설정
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
response = llm.invoke("서울의 날씨는 어때?")
print("🔁 응답 결과:", response.content)