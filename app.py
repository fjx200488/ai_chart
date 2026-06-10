import streamlit as st
import re
from openai import OpenAI

# ======================= 自定义输出解释器（动作解析） =======================
def parse_output(text: str) -> dict:
    """解析AI输出，提取动作描述并清理文本"""
    # 匹配 *动作* 或 [动作] 或 （动作）
    action_pattern = r'[\*\[（]([^*\[（）\]]+)[\*\]）]'
    actions = re.findall(action_pattern, text)
    clean_text = re.sub(r'[\*\[（][^*\[（）\]]+[\*\]）]', '', text).strip()
    return {
        "content": clean_text,
        "actions": actions if actions else None,
        "raw": text
    }

# ======================= 初始化Streamlit界面 =======================
st.set_page_config(page_title="角色扮演聊天室", page_icon="🎭")
st.title("🎭 角色扮演聊天室")
st.markdown("自定义一个角色，AI会完全代入该角色与你对话！")

# 侧边栏配置
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
        st.session_state.chat_history = []   # 存储对话历史（用于上下文）
        st.rerun()

    st.markdown("---")
    st.markdown("💡 **提示**：你可以随时修改角色设定，之后的新对话将自动采用新角色。")

# 初始化session状态
if "messages" not in st.session_state:
    st.session_state.messages = []          # 用于界面显示的消息列表
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # 用于LLM上下文的对话历史 [{"role": "user", "content": ...}, ...]

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ======================= 调用DeepSeek API =======================
def get_ai_response(api_key: str, role_desc: str, history: list, user_input: str) -> str:
    """发送请求到DeepSeek API，返回AI回复的原始文本"""
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    
    # 构建系统提示词（角色设定）
    system_prompt = (
        "你是一个角色扮演AI。请严格根据下面的角色设定来回答问题，完全融入角色，不要跳出角色。\n"
        f"角色设定如下：\n{role_desc}\n\n"
        "重要：用第一人称扮演该角色，不要解释自己是AI。保持角色一致性。\n"
        "如果角色设定中有语言风格要求（如口头禅、自称），请务必遵守。"
    )
    
    # 构建消息列表：[system] + 历史对话 + 当前用户输入
    messages = [
        {"role": "system", "content": system_prompt}
    ] + history + [
        {"role": "user", "content": user_input}
    ]
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.8,
        stream=False
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

    # 添加用户消息到界面和记忆
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 调用AI
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 获取AI原始回复
                raw_answer = get_ai_response(
                    deepseek_api_key,
                    role_description,
                    st.session_state.chat_history[:-1],  # 去掉最后一条（刚添加的用户输入，因为 get_ai_response 会再加一次）
                    user_input
                )
                # 解析动作和文本
                parsed = parse_output(raw_answer)
                answer_content = parsed["content"]
                if parsed["actions"]:
                    action_text = "、".join(parsed["actions"])
                    st.caption(f"🎭 {action_text}")
                st.markdown(answer_content)
                
                # 保存AI回复
                st.session_state.messages.append({"role": "assistant", "content": answer_content})
                st.session_state.chat_history.append({"role": "assistant", "content": answer_content})
                
            except Exception as e:
                st.error(f"出错了：{e}")
                st.info("请检查API Key是否正确，或稍后重试。")
