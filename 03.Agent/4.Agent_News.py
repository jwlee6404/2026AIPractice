import datetime

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import WebBaseLoader

import requests
from dotenv import load_dotenv
import pandas as pd
import io 
import os

load_dotenv()

client_id = os.getenv("C_ID")
client_secret = os.getenv("C_SCRT")
llm = ChatOpenAI(model="gpt-4o-mini")

@tool
def get_naver_news_search(query: str, display: int = 10, sort : str = 'date'):
    """
    네이버 뉴스 검색을 통해서 사용자가 요청 query에 해당하는 뉴스 본문을 반환한다. 
    Args : 
       input : query :사용자 검색어 ,  display : 한번에 출력하는 뉴스 개수, sort  : 정렬 방법
       output : 해당 기사의 본문 내용 
    """
    API_URL = "https://naverapihub.apigw.ntruss.com/search/v1/news"
    
    headers = {
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
        }
    params = {
        "query":  query,
        "display": display,
        "start": 1,
        "sort":  sort, 
        "format": 'json',
    }
    resp = requests.get(API_URL, headers=headers, params=params, timeout=15)
    loader = WebBaseLoader([x['originallink'] for x in resp.json()['items']][:10])
    docs = loader.load()
    return docs

checkpointer = MemorySaver()
agent = create_agent(model=llm, tools=[get_naver_news_search], checkpointer=checkpointer,
                     system_prompt=""" 
                         규칙 : 
                           1. 사용자가 요청한 뉴스 검색에 대해서는 반드시 도구를 사용해서 대답할 것.
                           2. 도구를 사용하지 못할 경우도 모른다고 할 것.
                           3. 사용자가 요청한 뉴스 검색에 대해서는 제목:, 내용:, 출처:, 날짜: 형식으로 정리해서 대답할 것.
                         """ )

result = agent.invoke({"messages" : [{
                "role" : 'user',
                'content' : "삼성전자 관련 최신 뉴스 5개 정리해서 알려줘"
            }]}, config={"configurable": {"thread_id": "user-007"}})

print(result['messages'][-1].content)