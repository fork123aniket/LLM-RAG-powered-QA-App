import streamlit as st
from src.qa_agent import Ray_LLM_QA
import base64
import os


def add_extra_space(times: int):
    for _ in range(times):
        st.write('\n')

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/jpg;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def load_model():
    path, config_path = os.environ["MODEL_PATH"], os.environ["CONFIG_PATH"]
    DETA_KEY_VALUE, DETA_BASE = os.environ["DETA_KEY"], os.environ["DETA_BASE"]
    LLMAgent = Ray_LLM_QA(DETA_KEY_VALUE, DETA_BASE, "thenlper/gte-base", 1, False, None, True, path, config_path)
    if 'model' not in st.session_state:
        st.session_state.model = LLMAgent


def main():
    set_png_as_page_bg(os.environ["IMAGE_PATH"])
    st.title("LLM RAG-powered QA App")
    add_extra_space(6)
    if 'load_val' not in st.session_state:
        st.session_state.load_val = 0
    if st.session_state.load_val == 0:
        load_model()
        st.session_state.load_val = 1
    query = st.selectbox(
        'Select the question from the below available questions:',
        ("What is the main purpose of a constructor in object-oriented programming?",
            "What is the difference between '==' and 'is' in Python?",
            "What is the significance of the 'main' method in Java?",
            "How does memory management work in C++?",
            "What are lambda functions in Python and when would you use them?"),
        index=None,
        placeholder="Choose a question...")
    if query is not None:
        with st.spinner("Executing the request..."):
            final_answer = st.session_state.model(query)
            st.success(f"Answer: '{final_answer}'")
    else:
        st.info("Please select a question to be answered...", icon="ℹ️")


if __name__ == '__main__':
    main()
