import streamlit as st
from dotenv import load_dotenv
import os

# LangChain & OpenAI
from langchain_community.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# .envからAPIキー読み込み
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 専門家の種類とシステムメッセージ
EXPERTS = {
    "金融の専門家": "あなたは金融の専門家です。金融、投資、経済に関する質問に専門的かつ分かりやすく回答してください。",
    "健康・医療の専門家": "あなたは健康・医療の専門家です。健康、医療、栄養に関する質問に専門的かつ分かりやすく回答してください。",
    "IT・テクノロジーの専門家": "あなたはIT・テクノロジーの専門家です。IT、プログラミング、テクノロジーに関する質問に専門的かつ分かりやすく回答してください。"
}

def get_llm_answer(input_text: str, expert_type: str) -> str:
    """入力テキストと専門家タイプを受け取り、LLMの回答を返す"""
    system_message = EXPERTS.get(expert_type, "")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{question}")
    ])
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)
    chain = LLMChain(llm=llm, prompt=prompt)
    result = chain.run({"question": input_text})
    return result

# Streamlit UI
st.title("専門家LLM質問アプリ")
st.markdown("""
### アプリ概要
このアプリは、金融・健康/医療・ITの専門家に質問できるAIチャットです。  
下記の手順でご利用ください。

1. 専門家の種類を選択してください。
2. 質問内容を入力し、「送信」ボタンを押してください。
3. AI専門家からの回答が表示されます。
""")

# ラジオボタンで専門家選択
expert_type = st.radio(
    "専門家の種類を選択してください：",
    list(EXPERTS.keys())
)

# 入力フォーム
input_text = st.text_area("質問内容を入力してください：", height=100)

if st.button("送信"):
    if not input_text.strip():
        st.warning("質問内容を入力してください。")
    else:
        with st.spinner("AI専門家が回答中..."):
            answer = get_llm_answer(input_text, expert_type)
        st.markdown("#### 回答")
        st.write(answer)