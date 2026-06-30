"""Streamlit UI — Ukrainian Literary Whining Generator."""

import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Генератор ниття", page_icon="📜", layout="centered")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.markdown("# 📜 Генератор ниття у стилі класичної української літератури")
st.markdown("Ний як класики")

with st.form("form"):
    text = st.text_area(
        "Ваше нарікання:",
        placeholder="Наприклад: У мене був жахливий день на роботі...",
        height=120,
        max_chars=1000,
    )
    submitted = st.form_submit_button("🪄 Перетворити", type="primary", use_container_width=True)

if submitted:
    if not text.strip():
        st.error("Введіть текст.")
    else:
        with st.spinner("Муза кличе..."):
            try:
                resp = requests.post(
                    f"{API_URL}/generate",
                    json={"text": text.strip()},
                    headers={"X-Session-Id": st.session_state.session_id},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    is_fallback = data.get("is_fallback", False)
                    st.markdown("**Результат:**")
                    if is_fallback:
                        st.warning("Модель тимчасово недоступна — показано запасне повідомлення.")
                    st.markdown(
                        f"""<div style="background:#fdf6e3;border-left:4px solid #8b6914;
                        padding:1rem 1.5rem;border-radius:4px;font-family:Georgia,serif;
                        font-size:1.1rem;line-height:1.8;color:#2c1810">
                        {data['output_text']}</div>""",
                        unsafe_allow_html=True,
                    )
                    st.code(data["output_text"], language=None)
                    st.caption(f"⏱ {data['latency_ms']} мс · {data['model_version']}")
                elif resp.status_code == 429:
                    st.warning("Занадто багато запитів. Спробуйте пізніше.")
                else:
                    st.error(f"Помилка: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error("API недоступне. Переконайтеся, що сервер запущено.")
            except requests.exceptions.Timeout:
                st.error("Модель думає надто довго. Спробуйте ще раз.")

with st.expander("📖 Приклади"):
    try:
        for ex in requests.get(f"{API_URL}/examples", timeout=3).json():
            st.markdown(f"**Сучасне:** {ex['input']}")
            st.markdown(
                f"""<div style="background:#fdf6e3;border-left:3px solid #8b6914;
                padding:.6rem 1rem;border-radius:3px;font-family:Georgia,serif">
                {ex['output']}</div>""",
                unsafe_allow_html=True,
            )
            st.markdown("---")
    except Exception:
        st.info("Запустіть API щоб побачити приклади.")

st.markdown(
    """<hr><small style="color:#aaa">
    ⚠️ Ваш текст може бути збережений для покращення моделі.
    Якщо вам важко — зателефонуйте: <b>0-800-100-102</b> (безкоштовно, цілодобово).
    </small>""",
    unsafe_allow_html=True,
)
