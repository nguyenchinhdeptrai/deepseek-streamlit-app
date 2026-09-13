import os
import json
import uuid
from datetime import datetime

import streamlit as st
from openai import OpenAI

#cấu hình trang web streamlit
st.set_page_config(page_title="DeepSeek Project Assistant", layout="wide")

#hiển thị tiêu đề lớn ở trang web + công tắc "Chế độ ẩn danh" ở góc trên bên phải
header_left, header_right = st.columns([4.6, 1.2], vertical_alignment="center")
with header_left:
    st.title("🤖 DeepSeek AI - Project Assistant")
with header_right:
    is_incognito = st.toggle(
        "🕵️ Ẩn danh",
        key="incognito_mode",
        help="Bật để trò chuyện ở chế độ ẩn danh: không lưu lịch sử đoạn chat này.",
    )
if is_incognito:
    st.caption("🕵️ Đang ở chế độ ẩn danh — đoạn chat này sẽ không được lưu lại.")

# --- CSS: thanh nhập liệu dạng "viên thuốc" (pill) giống Gemini/ChatGPT ---
# Dùng biến CSS của Streamlit (--background-color, --secondary-background-color, ...)
# nên màu sắc tự đổi theo theme Light/Dark mà người dùng chọn (kể cả "Use system setting").
st.markdown(
    """
    <style>
    /* Khoảng đệm dưới cùng để thanh composer không dính sát mép trình duyệt */
    div[data-testid="stBottomBlockContainer"] {
        padding-bottom: 1.6rem;
    }

    /* Khung "viên thuốc" bao quanh toàn bộ composer (nút +, ô nhập, model, mic) */
    .st-key-composer_bar {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 30px;
        padding: 6px 12px;
        margin: 0 auto;
        max-width: 900px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.12);
    }
    .st-key-composer_bar [data-testid="stHorizontalBlock"] {
        align-items: center;
        gap: 0.4rem;
    }

    /* Ô nhập chat: bỏ khung/nền mặc định để hoà vào viên thuốc */
    .st-key-composer_bar [data-testid="stChatInput"] {
        background: transparent !important;
        border: none !important;
    }
    .st-key-composer_bar [data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    .st-key-composer_bar [data-testid="stChatInputSubmitButton"] {
        background: var(--primary-color) !important;
        border-radius: 50% !important;
    }

    /* Nút "+" đính kèm file (mở popover upload) & nút mic (decorative) -> hình tròn */
    .st-key-attach_popover div[data-testid="stPopover"] > button,
    .st-key-mic_btn button {
        border-radius: 50% !important;
        width: 2.3rem;
        height: 2.3rem;
        padding: 0 !important;
        border: none !important;
        background: var(--background-color) !important;
    }
    .st-key-attach_popover div[data-testid="stPopover"] > button:hover,
    .st-key-mic_btn button:hover {
        background: var(--primary-color) !important;
        color: white !important;
    }

    /* Nút mở popover Model + Chế độ suy luận -> bo tròn thành 1 "chip" như ảnh mẫu */
    .st-key-model_popover div[data-testid="stPopover"] > button {
        background: var(--background-color);
        border-radius: 20px !important;
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        min-height: 2.3rem;
        padding: 0.2rem 1rem !important;
        white-space: nowrap;
    }
    .st-key-model_popover div[data-testid="stPopover"] > button:hover {
        border-color: var(--primary-color) !important;
        color: var(--primary-color) !important;
    }

    /* Popover đính kèm file: rộng rãi hơn mặc định để chứa khu vực kéo-thả file */
    div[data-testid="stPopoverBody"] {
        min-width: 260px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 0. Lưu trữ lịch sử chat trên đĩa (giống ChatGPT/Gemini) ---
# File JSON lưu tất cả các đoạn chat, đặt cạnh app.py để không phụ thuộc vào
# thư mục làm việc hiện tại (cwd) khi chạy `streamlit run`.
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.json")

# Các model DeepSeek hiện hành (deepseek-chat / deepseek-reasoner đã bị khai tử,
# hiện tại DeepSeek dùng deepseek-flash làm model chính, hỗ trợ cả chế độ
# suy luận lẫn trả lời nhanh).
MODEL_OPTIONS = {
    "deepseek-flash": "DeepSeek-V4.1-Flash — model mới nhất, nhanh, hỗ trợ suy luận (thinking mode)",
    "deepseek-v4-pro": "DeepSeek-V4-Pro — hiện được DeepSeek tự động route sang V4.1-Flash",
}


def load_conversations() -> dict:
    """Đọc toàn bộ lịch sử các đoạn chat đã lưu từ file JSON."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # File hỏng/không đọc được -> coi như chưa có lịch sử
            return {}
    return {}


def save_conversations(conversations: dict) -> None:
    """Ghi toàn bộ lịch sử các đoạn chat xuống file JSON."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Không thể lưu lịch sử chat: {e}")


def make_title(first_message: str) -> str:
    """Tạo tiêu đề ngắn gọn cho đoạn chat dựa trên tin nhắn đầu tiên."""
    title = " ".join(first_message.strip().split())
    if len(title) > 40:
        title = title[:40].rstrip() + "..."
    return title or "Đoạn chat mới"


try:
    # st.secrets sẽ báo lỗi nếu máy không có file secrets.toml nào cả (kể cả khi
    # dùng biến môi trường) -> bọc try/except để không bị crash trong trường hợp đó.
    api_key = st.secrets.get("DEEPSEEK_API_KEY")
except Exception:
    api_key = None
api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

if not api_key:
    st.warning("⚠️ Vui lòng nhập DeepSeek API Key trong `secrets.toml` hoặc biến môi trường `DEEPSEEK_API_KEY` để tiếp tục.")
    st.stop()

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# --- 1. Khởi tạo state cho hệ thống đa đoạn chat ---
if "conversations" not in st.session_state:
    # {conv_id: {"title": str, "created_at": str, "messages": [...], "summary": str|None}}
    st.session_state.conversations = load_conversations()

if "current_conv_id" not in st.session_state:
    # Nếu đã có lịch sử, mở đoạn chat gần nhất; nếu chưa có, tạo id cho một
    # đoạn chat mới (chưa lưu xuống đĩa cho tới khi có tin nhắn đầu tiên).
    if st.session_state.conversations:
        latest_id = max(
            st.session_state.conversations,
            key=lambda cid: st.session_state.conversations[cid].get("created_at", ""),
        )
        st.session_state.current_conv_id = latest_id
    else:
        st.session_state.current_conv_id = str(uuid.uuid4())

current_conv_id = st.session_state.current_conv_id
current_messages = st.session_state.conversations.get(current_conv_id, {}).get("messages", [])

# Model & chế độ suy luận: các widget thực sự nằm trong popover ở thanh chat phía
# dưới (composer_bar), nhưng giá trị được lưu trong session_state ngay từ đầu nên
# các phần code chạy trước đó (ví dụ nút "Tạo tóm tắt" trong sidebar) vẫn đọc được
# lựa chọn hiện tại.
st.session_state.setdefault("model_select", list(MODEL_OPTIONS.keys())[0])
st.session_state.setdefault("thinking_choice", "⚙️ Mặc định")
st.session_state.setdefault("reasoning_effort", "high")
selected_model = st.session_state["model_select"]


def build_extra_body():
    """Tạo tham số extra_body để bật/tắt/điều chỉnh thinking mode."""
    choice = st.session_state.get("thinking_choice", "⚙️ Mặc định")
    if choice == "🧠 Bật suy luận sâu":
        body = {"thinking": {"type": "enabled"}}
        effort = st.session_state.get("reasoning_effort")
        if effort:
            body["reasoning_effort"] = effort
        return body
    elif choice == "⚡ Tắt (trả lời nhanh)":
        return {"thinking": {"type": "disabled"}}
    return {}


# --- 2. Cấu hình ở Thanh bên (Sidebar) ---
with st.sidebar:
    # st.header("⚙️ Cấu hình")
    # api_key = st.text_input("Nhập DeepSeek API Key:", type="password")

    # --- Model AI của DeepSeek hiện đang dùng ---
    # (Lựa chọn model + chế độ suy luận đã được chuyển xuống thanh chat phía dưới,
    # bấm vào chip cạnh ô nhập tin nhắn để đổi)
    st.caption(f"🧠 Model đang dùng: **{selected_model}**")
    st.caption(MODEL_OPTIONS[selected_model])

    st.divider()

    # --- Danh sách các đoạn chat (giống ChatGPT/Gemini) ---
    st.header("💬 Các đoạn chat")

    if st.button("➕ Đoạn chat mới", use_container_width=True):
        st.session_state.current_conv_id = str(uuid.uuid4())
        st.rerun()

    # Sắp xếp các đoạn chat theo thời gian tạo, mới nhất lên đầu
    sorted_convs = sorted(
        st.session_state.conversations.items(),
        key=lambda item: item[1].get("created_at", ""),
        reverse=True,
    )

    for conv_id, conv in sorted_convs:
        col_select, col_delete = st.columns([5, 1])
        with col_select:
            label = conv.get("title", "Đoạn chat mới")
            is_current = conv_id == st.session_state.current_conv_id
            if st.button(
                ("📌 " if is_current else "") + label,
                key=f"select_{conv_id}",
                use_container_width=True,
            ):
                st.session_state.current_conv_id = conv_id
                st.rerun()
        with col_delete:
            if st.button("🗑️", key=f"delete_{conv_id}"):
                del st.session_state.conversations[conv_id]
                save_conversations(st.session_state.conversations)
                if st.session_state.current_conv_id == conv_id:
                    st.session_state.current_conv_id = str(uuid.uuid4())
                st.rerun()

    st.divider()

    # --- Tóm tắt lịch sử đoạn chat hiện tại ---
    st.header("📝 Tóm tắt đoạn chat")

    if is_incognito:
        st.caption("Chế độ ẩn danh không lưu lịch sử nên không có nội dung để tóm tắt.")
    elif not current_messages:
        st.caption("Đoạn chat này chưa có nội dung để tóm tắt.")
    else:
        summary_source = st.radio(
            "Chọn nguồn nội dung tóm tắt:",
            ["🧑 Tôi tự chọn nội dung", "🤖 AI tự tóm tắt toàn bộ"],
            key=f"summary_source_{current_conv_id}",
        )

        selected_indices = []
        if summary_source == "🧑 Tôi tự chọn nội dung":
            st.caption("Chọn các tin nhắn muốn đưa vào bản tóm tắt:")
            for i, msg in enumerate(current_messages):
                role_label = "🧑 Bạn" if msg["role"] == "user" else "🤖 AI"
                preview = msg["content"].replace("\n", " ").strip()
                if len(preview) > 60:
                    preview = preview[:60] + "..."
                checked = st.checkbox(
                    f"{role_label}: {preview}",
                    key=f"sum_chk_{current_conv_id}_{i}",
                )
                if checked:
                    selected_indices.append(i)

        if st.button("📝 Tạo tóm tắt", use_container_width=True):
            if summary_source == "🧑 Tôi tự chọn nội dung" and not selected_indices:
                st.warning("Vui lòng chọn ít nhất 1 tin nhắn để tóm tắt.")
            else:
                if summary_source == "🧑 Tôi tự chọn nội dung":
                    picked_messages = [current_messages[i] for i in selected_indices]
                else:
                    picked_messages = current_messages

                content_to_summarize = "\n\n".join(
                    f"[{m['role']}]: {m['content']}" for m in picked_messages
                )

                with st.spinner("Đang tóm tắt..."):
                    try:
                        summary_response = client.chat.completions.create(
                            model=selected_model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Bạn là trợ lý tóm tắt hội thoại. Hãy tóm tắt nội dung sau một cách ngắn gọn, đầy đủ ý chính, có thể dùng gạch đầu dòng nếu cần.",
                                },
                                {
                                    "role": "user",
                                    "content": f"Hãy tóm tắt đoạn hội thoại sau:\n\n{content_to_summarize}",
                                },
                            ],
                            temperature=0.3,
                            extra_body=build_extra_body(),
                        )
                        summary_text = summary_response.choices[0].message.content
                        st.session_state.conversations[current_conv_id]["summary"] = summary_text
                        save_conversations(st.session_state.conversations)
                        st.success("Đã tạo tóm tắt! Xem ở đầu khung chat.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi khi tóm tắt: {e}")

# # Kiếm tra API Key
# if not api_key:
#     st.warning("⚠️ Vui lòng nhập API Key ở thanh bên phải để tiếp tục.")
#     st.stop()

# --- 3. Hiển thị bản tóm tắt (nếu có) của đoạn chat hiện tại ---
if not is_incognito:
    existing_summary = st.session_state.conversations.get(current_conv_id, {}).get("summary")
    if existing_summary:
        with st.expander("🧾 Tóm tắt đoạn chat này", expanded=False):
            st.markdown(existing_summary)

# Hiển thị lịch sử trò chuyện của đoạn chat hiện tại (nếu không ở chế độ ẩn danh)
if not is_incognito:
    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# --- 4. Thanh composer dạng "viên thuốc": nút + (đính kèm file), ô nhập, model, mic ---
with st.container(key="composer_bar"):
    col_attach, col_input, col_model, col_mic = st.columns(
        [0.7, 6, 2.2, 0.7], vertical_alignment="center"
    )

    with col_attach:
        with st.container(key="attach_popover"):
            with st.popover("➕", use_container_width=True, help="Đính kèm file từ máy tính của bạn"):
                st.caption("📎 Đính kèm file từ máy tính của bạn")
                st.file_uploader(
                    "Chọn file đính kèm gửi cho AI:",
                    key="uploaded_file",
                    label_visibility="collapsed",
                )

    with col_input:
        prompt = st.chat_input("Nhập câu hỏi hoặc yêu cầu...")

    with col_model:
        with st.container(key="model_popover"):
            with st.popover(f"🧠 {selected_model}", use_container_width=True, help="Chọn model & chế độ suy luận"):
                st.selectbox(
                    "Model DeepSeek:",
                    options=list(MODEL_OPTIONS.keys()),
                    key="model_select",
                )
                st.caption(MODEL_OPTIONS[st.session_state["model_select"]])

                st.divider()

                # --- Chế độ suy luận sâu (Thinking mode), tích hợp ngay dưới model ---
                st.radio(
                    "Chế độ suy luận:",
                    ["⚙️ Mặc định", "🧠 Bật suy luận sâu", "⚡ Tắt (trả lời nhanh)"],
                    key="thinking_choice",
                )
                if st.session_state["thinking_choice"] == "🧠 Bật suy luận sâu":
                    st.select_slider(
                        "Mức độ suy luận:",
                        options=["low", "medium", "high"],
                        value=st.session_state.get("reasoning_effort", "high"),
                        key="reasoning_effort",
                    )

    with col_mic:
        with st.container(key="mic_btn"):
            st.button(
                "🎤",
                use_container_width=True,
                disabled=True,
                help="Nhập bằng giọng nói (sắp ra mắt)",
            )

uploaded_file = st.session_state.get("uploaded_file")
if uploaded_file is not None:
    st.caption(f"📎 Đã đính kèm: `{uploaded_file.name}`")

# --- 5. Xử lý tin nhắn người dùng ---
if prompt:
    context_prompt = prompt
    file_attached = False

    # Đọc nội dung file người dùng vừa tải lên từ máy tính của họ (nếu có)
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.getvalue().decode("utf-8")
            context_prompt = f"Dưới đây là nội dung file `{uploaded_file.name}` tôi đính kèm:\n```\n{file_content}\n```\n\nYêu cầu từ tôi: {prompt}"
            file_attached = True
        except UnicodeDecodeError:
            st.warning(
                f"Không thể đọc nội dung file `{uploaded_file.name}` (có thể không phải file văn bản). "
                "Tin nhắn sẽ được gửi mà không kèm nội dung file."
            )
        except Exception as e:
            st.error(f"Không thể đọc file: {e}")

    # Hiển thị câu hỏi lên màn hình
    with st.chat_message("user"):
        if file_attached:
            st.info(f"📄 **Đã đính kèm file:** `{uploaded_file.name}`")
        st.markdown(prompt)

    # Chuẩn bị dữ liệu gửi lên API DeepSeek
    system_msg = {"role": "system", "content": "Bạn là trợ lý lập trình chuyên nghiệp. Hãy trả lời chính xác, ngắn gọn."}

    if is_incognito:
        api_messages = [system_msg, {"role": "user", "content": context_prompt}]
    else:
        # Nếu đây là tin nhắn đầu tiên của đoạn chat -> tự động tạo mục lịch sử mới
        if current_conv_id not in st.session_state.conversations:
            st.session_state.conversations[current_conv_id] = {
                "title": make_title(prompt),
                "created_at": datetime.now().isoformat(),
                "messages": [],
                "summary": None,
            }

        st.session_state.conversations[current_conv_id]["messages"].append(
            {"role": "user", "content": context_prompt}
        )
        api_messages = [system_msg] + st.session_state.conversations[current_conv_id]["messages"]

    # Gọi API DeepSeek và phản hồi
    with st.chat_message("assistant"):
        with st.spinner("DeepSeek đang xử lý..."):
            try:
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=api_messages,
                    temperature=0.3,
                    extra_body=build_extra_body(),
                )
                ai_message = response.choices[0].message
                ai_reply = ai_message.content

                # Nếu thinking mode được bật, DeepSeek trả về thêm phần suy luận
                # trong `reasoning_content` — hiển thị riêng, không lưu vào lịch sử
                # (để tránh gửi lại phần suy luận cũ vào các lượt hỏi sau).
                reasoning_content = getattr(ai_message, "reasoning_content", None)
                if reasoning_content:
                    with st.expander("🧠 Quá trình suy luận của AI", expanded=False):
                        st.markdown(reasoning_content)

                st.markdown(ai_reply)

                if not is_incognito:
                    st.session_state.conversations[current_conv_id]["messages"].append(
                        {"role": "assistant", "content": ai_reply}
                    )
                    # Lưu lịch sử xuống đĩa ngay sau mỗi lượt trao đổi
                    save_conversations(st.session_state.conversations)
            except Exception as e:
                st.error(f"Lỗi kết nối API: {e}")