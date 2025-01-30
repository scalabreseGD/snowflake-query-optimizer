import re
from typing import Any, Dict, Literal

import opik
from openai import OpenAI, AzureOpenAI
from opik.integrations.openai import track_openai


class BaseAIClient:

    def __init__(self, client, model_name):
        opik.configure(use_local=True,
                       url='http://localhost:5173/api',
                       workspace='default',
                       force=True
                       )

        self.client = track_openai(client)
        self.model_name = model_name

    @classmethod
    def build_from_api_type(cls, api_type: Literal['openai', 'azure', 'deepseek'], confs):
        if api_type == 'openai':
            return OpenAIClient(confs)
        elif api_type == 'azure':
            return AzureOpenAIClient(confs)
        elif api_type == 'deepseek':
            return DeepSeekR1AIClient(confs)
        else:
            raise ValueError(f'API type {api_type} not supported')

    def invoke(self, messages, **chat_kwargs):
        return self.client.chat.completions.create(model=self.model_name, messages=messages, **chat_kwargs)


class AzureOpenAIClient(BaseAIClient):
    def __init__(self, confs: Dict[str, Any]):
        api_key = confs['API_KEY']
        api_version = confs['API_VERSION']
        api_endpoint = confs['API_ENDPOINT']
        super().__init__(client=AzureOpenAI(azure_endpoint=api_endpoint,
                                            api_key=api_key,
                                            api_version=api_version,
                                            ),
                         model_name=confs['DEPLOYMENT_NAME'])


class OpenAIClient(BaseAIClient):
    def __init__(self, confs: Dict[str, Any]):
        api_key = confs['API_KEY']
        base_url = confs['API_ENDPOINT']
        model_name = confs['MODEL_NAME']
        super().__init__(client=OpenAI(api_key=api_key, base_url=base_url), model_name=model_name)


class DeepSeekR1AIClient(OpenAIClient):
    __pattern = r"<think>.*?</think>\s*"  # `\s*` removes any trailing whitespace or newlines

    def __init__(self, confs: Dict[str, Any]):
        super().__init__(confs)

    def invoke(self, messages, **chat_kwargs):
        response = super().invoke(messages, **chat_kwargs)
        for choice_idx in range(len(response.choices)):
            raw_content = response.choices[choice_idx].message.content
            content_wo_thinking = re.sub(self.__pattern, '', raw_content, flags=re.DOTALL)
            response.choices[choice_idx].message.content = content_wo_thinking
        return response
