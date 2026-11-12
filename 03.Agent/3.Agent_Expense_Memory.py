import datetime

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

import requests
from dotenv import load_dotenv
import pandas as pd
import io 

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

@tool
def record_expense(date: str, amount: int, description: str) -> str:
    """
    일시, 지출금액, 지출내역을 기록합니다.
    
    Args:
        date (str): 지출 일시 (예: '2026-11-11 14:30')
        amount (int): 지출 금액 (숫자만 입력)
        description (str): 지출 내역 설명 (예: '점심 식사 - 햄버거')
        
    Returns:
        str: 기록 완료 메시지 또는 에러 메시지
    """
    try:
        # 실제 환경에서는 데이터베이스나 파일에 저장하는 로직이 들어갑니다.
        # 여기서는 딕셔너리 형태로 저장하고 출력하는 예시를 보여줍니다.
        expense_record = {
            "date": date,
            "amount": amount,
            "description": description,
            "recorded_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
        # 임시 출력 (실제 적용 시 삭제 및 DB 연동)
        print(f"[DB 저장 시뮬레이션] {expense_record}")
        
        return f"지출 기록 완료: {date}에 '{description}' 명목으로 {amount:,}원이 기록되었습니다."
    
    except Exception as e:
        return f"지출 기록 실패: {str(e)}"


checkpointer = MemorySaver()
agent = create_agent(model=llm, tools=[record_expense], checkpointer=checkpointer)

rt = agent.invoke({"messages" : [{"role" : 'user', 'content' : "나 방금 14시 30분에 햄버거 세트 먹고 15,000원 지출했어. 기록해줘."}]}, config={"configurable": {"thread_id": "user-001"}})
rt = agent.invoke({"messages" : [{"role" : 'user', 'content' : "나 오늘 뭐 먹었지?"}]}, config={"configurable": {"thread_id": "user-001"}})

print(rt['messages'][-1].content)
