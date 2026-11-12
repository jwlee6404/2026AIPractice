from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
output_parser = StrOutputParser()

#프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 주어진 컨텍스트를 근거로 질문에 답하는 어시스턴트입니다. "
        "컨텍스트에 없는 내용은 추측하지 말고 '제공된 정보로는 알 수 없습니다.'라고 답하세요. "
        "답변은 한국어로 작성하세요.",
    ),
    (
        "human",
        "[컨텍스트]\n{context}\n\n"
        "[질문]\n{question}",
    )
])

# LCEL(|)을 이용해 체인(Chain) 생성
context_chain = prompt | llm | output_parser

sample_context = (
    "우리 회사의 연차는 입사 1년 차에 15일이 부여되며, "
    "이후 근속 연수에 따라 최대 25일까지 늘어납니다. "
    "연차 신청은 최소 3일 전에 사내 시스템을 통해 해야 합니다."
)
questions = [
    "입사 1년 차 연차는 며칠인가요?",
    "연차는 언제까지 신청해야 하나요?",
    "재택근무 규정은 어떻게 되나요?",  # 컨텍스트에 없는 질문
]

for q in questions:
    answer = context_chain.invoke({"context": sample_context, "question": q})
    print(f"Q: {q}")
    print(f"A: {answer}\n")
