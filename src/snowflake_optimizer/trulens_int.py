import streamlit
from openai import OpenAI, AzureOpenAI
from trulens.apps.custom import instrument, TruCustomApp
from trulens.core import Feedback
from trulens.dashboard import streamlit as trulens_st
from trulens.providers.openai import OpenAI as fOpenAI
from trulens.providers.openai.endpoint import OpenAIEndpoint


def get_trulens_app(app, feedbacks):
    return TruCustomApp(
        app=app,
        app_name="ChatApp",
        app_version="v1",
        feedbacks=feedbacks,
    )


class ChatApp:

    def __init__(self, llm_client: OpenAI, llm_model, evaluator_llm_client: fOpenAI = None,
                 evaluator_model: str = None):
        self.__llm_client = llm_client
        self.__llm_model = llm_model
        # self.__provider = fOpenAI(model_engine=eval_model, api_key=api_key, base_url=base_url)
        self.__evaluator_llm_client = evaluator_llm_client
        self.__evaluator_model = evaluator_model

    def get_feedbacks(self):
        # Define a relevance feedback function
        f_relevance = Feedback(self.__evaluator_llm_client.relevance).on_input_output()
        f_correctness = Feedback(self.__evaluator_llm_client.correctness).on_input_output()

        return [f_relevance,
                f_correctness
                ]

    @instrument
    def chat(self, user_input, system_prompt=None, **chat_args):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})
        response = self.__llm_client.chat.completions.create(
            model=self.__llm_model,
            messages=messages,
            **chat_args
        )
        return response.choices[0].message.content

    @staticmethod
    def publish_records(app_id):
        # Retrieve and display feedback results
        session = streamlit.session_state['tru_session']
        records = session.get_records_and_feedback(app_ids=[app_id])
        if records:
            record = records[-1]
            trulens_st.trulens_feedback(record=record)
            trulens_st.trulens_trace(record=record)
