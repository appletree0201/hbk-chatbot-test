import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

st.title("나의 챗GPT")

if "messages" not in st.session_state:
    # 대화 기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []
# 사이드바 생성
    with st.sidebar:
        # 초기화 버튼
        clear_btn = st.button("대화 초기화")

# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)
        # st.write(f"{chat_message.role}: {chat_message.content}")


# for role, message in st.session_state["messages"]:
# st.chat_message(role).write(message)


# 새로운 메세지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

# 체인생성
def create_chain():
    # prompt | llm | out_parser
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", " 당신은 친절한 AI 어시스턴트입니다."),
            ("user", "#Question:\n{question}"),
        ]
    )
    # GPT
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    # 출력파서
    Output_Parser = StrOutputParser()

    # 체인생성
    chain = prompt | llm | Output_Parser
    return chain

# 초기화 버튼이 눌리면
if clear_btn:
    st.session_state["messages"]

# 이전대화기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("질문을 입력하세요")

# 만약에 사용자 입력이 들어오면...
if user_input:
    # st.write(f"사용자 입력: {user_input}")
    st.chat_message("user").write(user_input)
    # 체인생성
    chain = create_chain()
    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
    # ai_answer = chain.invoke({"question": user_input})

        # 빈공간 컨제이터를 만들어서 여기에 토근을 스트리밍한다.
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # st.chat_message("assistant").write(ai_answer)

    # 대화 기록을 저장한다.
    add_message("user", user_input)
    add_message("assistant", ai_answer)
    # ChatMessage(role="user", content=user_input)
    # ChatMessage(role="assistant", content=user_input)
    # st.session_state["messages"].append(("user", user_input))
    # st.session_state["messages"].append(("assistant", user_input))
