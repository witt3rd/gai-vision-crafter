import json
import os

from langchain.callbacks.base import BaseCallbackHandler

from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

import snakemd

from config import get_config

#
# Helpers
#

# read the json prompts file
with open(get_config().PROMPTS_FILE) as f:
    prompts = json.load(f)


class StreamHandler(BaseCallbackHandler):
    def __init__(
        self,
        container: DeltaGenerator = None,
        initial_text="",
    ) -> None:
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        if self.container is not None:
            self.container.write(self.text)


def reset() -> None:
    if "llm" in st.session_state:
        del st.session_state["llm"]
    if "memory" in st.session_state:
        del st.session_state["memory"]
    if "basic_description" in st.session_state:
        del st.session_state["basic_description"]
    if "tokens" in st.session_state:
        del st.session_state["tokens"]
    if "cost" in st.session_state:
        del st.session_state["cost"]
    if "job_title" in st.session_state:
        del st.session_state["job_title"]
    if "job_description" in st.session_state:
        del st.session_state["job_description"]
    if "goals_and_objectives" in st.session_state:
        del st.session_state["goals_and_objectives"]
    if "priorities" in st.session_state:
        del st.session_state["priorities"]
    if "skills_and_competencies" in st.session_state:
        del st.session_state["skills_and_competencies"]
    if "performance_standards" in st.session_state:
        del st.session_state["performance_standards"]
    if "transcript" in st.session_state:
        del st.session_state["transcript"]


#
# Streamlit app
#

st.set_page_config(
    page_title="VisionCrafter™",
    page_icon="🔦",
    # layout="wide",
)

if get_config().OPENAI_API_KEY is None:
    st.error(
        "OpenAI API key not found. Please set OPENAI_API_KEY environment variable.",
        icon="🔑",
    )
    st.stop()

#
# UI
#

# Custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
<div class="header-container">
  <img src="./app/static/hero.png" alt="VisionCrafter" class="header-image">
  <div class="header-text">
    <h2>VisionCrafter™</h2>
    <p>Crafting Clarity, Directing Vision</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# st.divider()
# st.write(
#     "[![Star](https://img.shields.io/github/stars/witt3rd/gai-ai2ai.svg?logo=github&style=social)](https://gitHub.com/witt3rd/gai-ai2ai)"
#     + "[![Follow](https://img.shields.io/twitter/follow/dt_public?style=social)](https://www.twitter.com/dt_public)"
# )

tab1, tab2 = st.tabs(["Main", "About"])

with tab1:
    with st.container():
        with st.form("basic_description_form"):
            st.text_area(
                "Basic Description",
                key="basic_description",
                height=100,
                help="This will be used to craft a full, visionary description.",
            )
            st.form_submit_button("Generate", use_container_width=True)

    output = st.empty()

    def is_valid_state(key: str) -> bool:
        return key in st.session_state and len(st.session_state[key].strip()) > 0

    def get_llm(
        container: DeltaGenerator = None,
    ) -> ChatOpenAI:
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            verbose=True,
            openai_api_key=get_config().OPENAI_API_KEY,
            streaming=True,
            callbacks=[StreamHandler(container)],
        )
        return llm

    def call_llm(
        container: DeltaGenerator = None,
    ) -> BaseMessage:
        with st.spinner("Thinking..."):
            llm = get_llm(container)
            memory: ChatMessageHistory = st.session_state["memory"]
            response = llm(memory.messages)
            memory.add_message(AIMessage(content=response.content))
            return response

    if "tokens" not in st.session_state:
        st.session_state["tokens"] = 0

    if "cost" not in st.session_state:
        st.session_state["cost"] = 0.0

    if "memory" not in st.session_state:
        memory = ChatMessageHistory()
        st.session_state["memory"] = memory

        system_message = SystemMessage(content=prompts["System"])
        memory.add_message(system_message)

    if is_valid_state("basic_description") and "job_title" not in st.session_state:
        basic_description = st.session_state["basic_description"]
        msg = HumanMessage(
            content=f"Suggest a job title for the following basic job description: {basic_description}.  Only respond with just the job title and no other text."
        )
        st.session_state["memory"].add_message(msg)
        response = call_llm()  # no output
        st.session_state["job_title"] = response.content

    if is_valid_state("job_title") and "job_description" not in st.session_state:
        job_title = st.session_state["job_title"]
        msg = HumanMessage(
            content=f"Write a formal Job Description that sets the {job_title} up for success."
        )
        st.session_state["memory"].add_message(msg)
        response = call_llm(output)
        output.write()
        st.session_state["job_description"] = response.content

    if (
        is_valid_state("job_description")
        and "goals_and_objectives" not in st.session_state
    ):
        job_title = st.session_state["job_title"]
        msg = HumanMessage(
            content=f"What are the Goals and Objectives for the {job_title}?"
        )
        st.session_state["memory"].add_message(msg)
        response = call_llm(output)
        output.write()
        st.session_state["goals_and_objectives"] = response.content

    if is_valid_state("goals_and_objectives") and "priorities" not in st.session_state:
        job_title = st.session_state["job_title"]
        msg = HumanMessage(content=f"What are the Priorities for the {job_title}?")
        st.session_state["memory"].add_message(msg)
        response = call_llm(output)
        output.write()
        st.session_state["priorities"] = response.content

    if (
        is_valid_state("priorities")
        and "skills_and_competencies" not in st.session_state
    ):
        job_title = st.session_state["job_title"]
        msg = HumanMessage(
            content=f"What are the Skills and Competencies for the {job_title}?"
        )
        st.session_state["memory"].add_message(msg)
        response = call_llm(output)
        output.write()
        st.session_state["skills_and_competencies"] = response.content

    if (
        is_valid_state("skills_and_competencies")
        and "performance_standards" not in st.session_state
    ):
        job_title = st.session_state["job_title"]
        msg = HumanMessage(
            content=f"What are the Performance Standards for the {job_title}?"
        )
        st.session_state["memory"].add_message(msg)
        response = call_llm(output)
        output.write()
        st.session_state["performance_standards"] = response.content

    if is_valid_state("performance_standards") and "transcript" not in st.session_state:
        doc = snakemd.new_doc()
        job_title = st.session_state["job_title"]
        doc.add_heading(f"Job Title: {job_title}", 1)

        basic_description = st.session_state["basic_description"]
        doc.add_heading(f"Initial Description", 2)
        doc.add_raw(basic_description)

        job_description = st.session_state["job_description"]
        doc.add_heading(f"Job Description", 2)
        doc.add_raw(job_description)

        goals_and_objectives = st.session_state["goals_and_objectives"]
        doc.add_heading(f"Goals and Objectives", 2)
        doc.add_raw(goals_and_objectives)

        priorities = st.session_state["priorities"]
        doc.add_heading(f"Priorities", 2)
        doc.add_raw(priorities)

        skills_and_competencies = st.session_state["skills_and_competencies"]
        doc.add_heading(f"Skills and Competencies", 2)
        doc.add_raw(skills_and_competencies)

        performance_standards = st.session_state["performance_standards"]
        doc.add_heading(f"Performance Standards", 2)
        doc.add_raw(performance_standards)

        # ensure the output dir exists
        os.makedirs(get_config().JOBS_DIR, exist_ok=True)
        filename = os.path.join(get_config().JOBS_DIR, job_title)
        doc.dump(filename)
        st.session_state["transcript"] = filename + ".md"

        output.empty()
        output.markdown(str(doc))

    st.button(
        "Reset",
        on_click=reset,
        help="Reset the app.",
        type="primary",
    )

with tab2:
    with open("README.md", "r") as f:
        readme = f.read()
    readme = readme.replace("static/", "./app/static/")
    st.markdown(readme, unsafe_allow_html=True)
