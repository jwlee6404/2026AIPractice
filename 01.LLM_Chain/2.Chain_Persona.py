from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.7)
output_parser = StrOutputParser()

#프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 마케팅 10년차 담당자입니다."),
    ("user", "{question}")
])

# LCEL(|)을 이용해 체인(Chain) 생성
chain = prompt | llm | output_parser

response = chain.invoke({"question": "금융상품을 판매하는 마케팅 문구 3가지 추천해줘"})

print(response)