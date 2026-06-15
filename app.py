import streamlit as st
from langchain_openai import ChatOpenAI
from typing import Dict, Any
import re

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import BaseOutputParser

from langchain_core.runnables import RunnablePassthrough
from langchain_classic.memory import ConversationBufferMemory



# ======================= 自定义输出解释器 =======================
class RoleplayOutputParser(BaseOutputParser):
    """解析AI输出，可提取动作、表情并美化显示"""

    def parse(self, text: str) -> Dict[str, str]:
        """
        解析LLM输出，分离出说话内容和动作描述（如果有）
        简单实现：检查是否包含类似 *动作* 或 （表情） 的标记
        """
        # 匹配常见的动作描述模式：*挥手*、[笑]、（开心地）
        action_pattern = r'[\*\[（]([^*\[（）\]]+)[\*\]）]'
        actions = re.findall(action_pattern, text)

        # 清理文本：去掉动作标记，保留纯文本
        clean_text = re.sub(r'[\*\[（][^*\[（）\]]+[\*\]）]', '', text).strip()

        return {
            "content": clean_text,
            "actions": actions if actions else None,
            "raw": text
        }

    @property
    def _type(self) -> str:
        return "roleplay_parser"


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
    # 默认提供一个示例角色卡（可莉）
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
        if "memory" in st.session_state:
            st.session_state.memory.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("💡 **提示**：你可以随时修改角色设定，之后的新对话将自动采用新角色。")

# 初始化session状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    # 使用ConversationBufferMemory存储历史
    st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="history")
if "parser" not in st.session_state:
    st.session_state.parser = RoleplayOutputParser()

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ======================= 构建Chain（带提示词模板 + 输出解释器） =======================
def get_roleplay_chain(api_key: str, role_desc: str, memory: ConversationBufferMemory):
    """创建角色扮演链：提示词模板 -> LLM -> 输出解释器"""
    # 提示词模板：包含角色设定、对话历史、用户输入
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个角色扮演AI。请严格根据下面的角色设定来回答问题，完全融入角色，不要跳出角色。\n"
         "角色设定如下：\n{role_desc}\n\n"
         "重要：用第一人称扮演该角色，不要解释自己是AI。保持角色一致性。\n"
         "如果角色设定中有语言风格要求（如口头禅、自称），请务必遵守。"
         ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    # 初始化DeepSeek模型（兼容OpenAI接口）
    llm = ChatOpenAI(
        model="deepseek-chat",
        temperature=0.8,  # 稍高温度让角色更生动
        openai_api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )

    # 构建链：prompt | llm | 输出解释器
    chain = prompt | llm | st.session_state.parser
    return chain


# ======================= 处理用户输入 =======================
user_input = st.chat_input("说点什么吧...")

if user_input:
    if not deepseek_api_key:
        st.warning("请在侧边栏输入DeepSeek API Key")
        st.stop()
    if not role_description.strip():
        st.warning("请填写角色形象描述")
        st.stop()

    # 添加用户消息到界面
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 调用链
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                # 获取链（每次重新构建以应用最新的角色描述）
                chain = get_roleplay_chain(deepseek_api_key, role_description, st.session_state.memory)

                # 调用链，传入输入和记忆中的历史
                # 注意：memory.load_memory_variables({}) 返回包含"history"的字典
                history = st.session_state.memory.load_memory_variables({})["history"]

                result = chain.invoke({
                    "role_desc": role_description,
                    "history": history,
                    "input": user_input
                })

                # result 是 RoleplayOutputParser 返回的字典
                answer_content = result["content"]
                if result["actions"]:
                    # 如果有动作描述，额外显示在斜体
                    action_text = "、".join(result["actions"])
                    st.caption(f"🎭 {action_text}")

                st.markdown(answer_content)

                # 保存AI回复到记忆和界面
                st.session_state.messages.append({"role": "assistant", "content": answer_content})
                # 更新memory：保存用户输入和AI输出
                st.session_state.memory.save_context({"input": user_input}, {"output": answer_content})

            except Exception as e:
                st.error(f"出错了：{e}")
                st.info("请检查API Key是否正确，或稍后重试。")

#  终端运行：streamlit run "D:\Pythonstudy\AI助手\AI2\app.py"

