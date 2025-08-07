import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from retriever import create_retriever

# API KEY 정보로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("PDF RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")


st.title("HBK's 로컬모델기반 Q/A👀")

# 처음 한번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기위한 용도로 생성
    st.session_state["messages"] = []

# 아무런 파일을 업로드 하지 않았을때
if "chains" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화초기화")

    # 파일 업로더
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    # 모델 선택 메뉴
    selected_model = st.selectbox(
        "LLM 선택", ["ollama", "gpt-4o"], index=0
    )


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메세지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시에 저장(시간이 오래걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return create_retriever(file_path)


# 체인생성
def create_chain(retriever, model_name="ollama"):

    # 프롬프트를 생성합니다.
    if model_name=="ollama":
        # 단계 6: 프롬프트 생성(Create Prompt)
        prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

        # 단계 7: 언어모델(LLM) 생성
        # 모델(LLM) 을 생성합니다.
        # Ollama 모델을 불러옵니다.
        llm = ChatOllama(model="EEVE-Korean-10.8B:latest", temperature=0)

        # 단계 8: 체인(Chain) 생성
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    return chain


# 파일이 업로드 되었을때
if uploaded_file:
    # 파일 업로드 후 retriever 생성 (작업 시간이 오래 걸릴 예정)
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain


# 초기화 버튼이 눌리면
if clear_btn:
    st.session_state["messages"] = []


# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요")

# 경고 메세지를 띄우기 위한 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # ai_answer = chain.invoke({"question": user_input})

        # AI의 답변
        # st.chat_message("assistant").write(ai_answer)

        # 대화기록을 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메세지 출력
        warning_msg.error("파일을 업로드 해주세요.")
