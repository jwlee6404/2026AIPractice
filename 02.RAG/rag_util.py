import re
from typing import List 
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def split_by_episode(raw_text: str, source_name: str) -> List[Document]:
    parts = re.split(r"(\d+화\.\s*[^\n]+)", raw_text)

    documents = []

    # 보통 split 결과는 [앞부분, 제목1, 본문1, 제목2, 본문2, ...] 형태가 됩니다.
    for i in range(1, len(parts), 2):
        episode_title = parts[i].strip()  # 예: "1화. 간판이 하나 더 생겼다"
        episode_body = parts[i + 1].strip() if i + 1 < len(parts) else ""

        # "1화"에서 숫자 부분만 추출합니다.
        match = re.match(r"(\d+)화\.", episode_title)
        episode_number = int(match.group(1)) if match else None

        # 제목과 본문을 다시 합쳐 하나의 에피소드 문서로 만듭니다.
        full_text = f"{episode_title}\n{episode_body}"

        documents.append(
            Document(
                page_content=full_text,
                metadata={
                    "source": source_name,          # 어떤 파일에서 왔는지
                    "episode": episode_number,      # 몇 화인지
                    "episode_title": episode_title, # 에피소드 제목
                },
            )
        )
    return documents

def create_chunked_documents(documents: List[Document], chunk_size: int = 700, chunk_overlap: int = 120) -> List[Document]:
    

    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '. ', ' ', ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunked_docs = splitter.split_documents(documents)

    for idx, doc in enumerate(chunked_docs):
        doc.metadata["chunk_id"] = idx

    return chunked_docs

def create_vectorstore(documents: List[Document], persist_directory: str, collection_name: str):    

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )

    vectorstore.add_documents(documents)

    return vectorstore