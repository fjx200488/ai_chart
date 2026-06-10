import streamlit as st
import re
from openai import OpenAI

# ======================= 动作解析函数 =======================
def parse_output(text: str) -> dict:
    """解析AI输出，提取动作描述并清理文本"""
    action_pattern = r'[\*\[（]([^*\[（）\]]+)[\*\]）]'
    actions = re.findall(action_pattern, text)
    clean_text = re.sub(r'[\*\[（][^*\[（）\]]+[\*\]）]', '', text).strip()
    return {
        "content": clean_text,
        "actions": actions if actions else None,
        "raw": text
    }

# ======================= 界面配置 =======================
st.set_page_config(page_title="角色扮演聊天室", page_icon="🎭")
st.title("🎭 角色扮演聊天室")
st.markdown("自定义一个角色，AI会完全代入该角色与你对话！")

with st.sidebar:
    st.header("⚙️ 配置")
    deepseek_api_key = st.text_input("DeepSeek API Key", type="password",
                                     help="输入你的DeepSeek API Key（从platform.deepseek.com获取）")

    st.header("🎭 角色设定")
    default_role = """你现在是「可莉」，来自游戏《原神》中的西风骑士团。
# 性格：充满好奇、活泼天真、喜欢爆炸物和炸鱼。
# 语言风格：自称“可莉”，说话可爱，常用“哒哒哒”、“蹦蹦炸弹”等。
# 行为：喜欢拉着大哥出去玩，但也记得骑士团的规矩。
# 当前对话对象：你最喜欢的大哥。"""

    role_description = st.text_area("角色形象描述（越详细越像）",
                                    value=default_role,
                                    height=300,
                                    help="请用自然语言描述这个角色的性格、说话方式、背景等。")

    if st.button("清除对话历史"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# 初始化状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ======================= API 调用 =======================
def get_ai_response(api_key: str, role_desc: str, history: list, user_input: str) -> str:
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    system_prompt = (
        "你是一个角色扮演AI。请严格根据下面的角色设定来回答问题，完全融入角色，不要跳出角色。\n"
        f"角色设定如下：\n{role_desc}\n\n"
        "重要：用第一人称扮演该角色，不要解释自己是AI。保持角色一致性。\n"
        "如果角色设定中有语言风格要求（如口头禅、自称），请务必遵守。"
    )
    messages = [
        {"role": "system", "content": system_prompt}
    ] + history + [
        {"role": "user", "content": user_input}
    ]
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.8
    )
    return response.choices[0].message.content

# ======================= 处理用户输入 =======================
user_input = st.chat_input("说点什么吧...")

if user_input:
    if not deepseek_api_key:
        st.warning("请在侧边栏输入DeepSeek API Key")
        st.stop()
    if not role_description.strip():
        st.warning("请填写角色形象描述")
        st.stop()

    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                raw_answer = get_ai_response(
                    deepseek_api_key,
                    role_description,
                    st.session_state.chat_history[:-1],
                    user_input
                )
                parsed = parse_output(raw_answer)
                if parsed["actions"]:
                    st.caption(f"🎭 {'、'.join(parsed['actions'])}")
                st.markdown(parsed["content"])
                st.session_state.messages.append({"role": "assistant", "content": parsed["content"]})
                st.session_state.chat_history.append({"role": "assistant", "content": parsed["content"]})
            except Exception as e:
                st.error(f"出错了：{e}")
