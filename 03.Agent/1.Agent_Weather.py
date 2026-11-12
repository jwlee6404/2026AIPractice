from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
import requests
from dotenv import load_dotenv


load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

@tool
def get_weather(city : str) -> str:
    """
    도시의 이름을 입력받아서 해당 도시의 날씨 정보를 반환하는 함수
    input : 도시의 이름 예) seoul
    return : 해당 도시의 날씨 정보
    """
    return requests.get(f"https://wttr.in/{city}?format=j1").json()



agent = create_agent(model=llm, tools=[get_weather])

rt = agent.invoke({"messages" : [{"role" : 'user', 'content' : "오늘 부산의 날씨 알려줘"}]})

print(rt['messages'][-1].content)
