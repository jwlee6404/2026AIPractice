from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain 


from rag_util import split_by_episode, create_chunked_documents, create_vectorstore


load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

# 텍스트 파일 읽기
with open("./Storage/소설.txt", "r", encoding='utf-8') as f:
    text = f.read()

# 텍스트를 의미단위로 분할 (에피소드 단위)
episode_docs = split_by_episode(text, source_name="소설")

# 의미단위로 분할된 문서를 청크 단위로 생성
chunked_docs = create_chunked_documents(episode_docs)

# Vector DB 생성
vectorstore = create_vectorstore(chunked_docs, persist_directory="./Storage", collection_name="novel")

# chunked_docs insert into vectorstore
vectorstore.add_documents(chunked_docs)

# Vector DB에서 검색을 위한 retriever 생성
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k" : 3})

# Prompt 템플릿 생성
prompt = ChatPromptTemplate.from_template(
    """
너는 한국어 소설 분석용 RAG 어시스턴트다.

반드시 아래 규칙을 지켜라.
1. 답변은 무조건 한국어로 작성한다.
2. 제공된 컨텍스트 범위 안에서만 답한다.
3. 컨텍스트에 없는 내용은 추측하지 말고 "문서에서 확인되지 않습니다."라고 답한다.
4. 감정 변화, 사건 흐름, 장면의 의미를 또렷하게 설명한다.

질문:
{input}

컨텍스트:
{context}
"""
)

document_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, document_chain)

rt = rag_chain.invoke({'input' : "주인공이 화난 이유가 뭐야?"})

# 1. LLM이 생성한 최종 답변만 깔끔하게 출력하기
print("최종 답변:")
print(rt['answer'])

# 2. 답변 생성에 사용된 참조 문서(Context) 내용 확인하기
print("\n--- 참고한 원본 문서(Context) ---")
for i, doc in enumerate(rt['context'], 0):
    print(f"\n[문서 {i}]")
    print(f"내용: {doc.page_content}") 
    print(f"출처(메타데이터): {doc.metadata}")

# 3. 반환된 전체 데이터 구조 확인하기 (디버깅용)
print("\n--- 전체 결과 키(Keys) ---")
print(rt.keys())



