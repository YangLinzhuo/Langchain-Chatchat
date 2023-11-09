from server.model_workers.base import ApiModelWorker
from fastchat import conversation as conv
from configs.model_config import TEMPERATURE
from server.utils import get_model_worker_config, get_httpx_client
import sys
import json
from typing import List, Literal, Dict
from configs import MS_SERVER


from pydantic import BaseModel
from typing import Optional


class Parameters(BaseModel):
    # mode: Optional[int] = 0
    do_sample: bool = True
    repetition_penalty: Optional[float] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_new_tokens: Optional[int] = None
    return_full_text: bool = True



class ClientRequest(BaseModel):
    # Prompt
    inputs: str
    # Generation parameters
    parameters: Optional[Parameters] = None
    # Whether to stream output tokens
    stream: bool = False


def request_mindspore_api(
    messages: List[Dict[str, str]],
    temperature: float = TEMPERATURE,
    model_name: str = "mindspore-api",
    version: str = None,
):
    INF_URL = f'http://{MS_SERVER["host"]}:{MS_SERVER["port"]}'
    config = get_model_worker_config(model_name)
    version = version or config.get("version")
    model_type = config.get("model_type")

    # TODO: replace model_type with configurable parameter
    url = f"{INF_URL}/models/{model_type}"

    stream = True
    parameters = Parameters(
        do_sample=False,
        repetition_penalty=1.0,
        temperature=temperature,
        top_k=3,
        top_p=1,
        max_new_tokens=512,
        return_full_text=False,
    )

    # The last message is the newest user input
    inputs = messages[-1]
    role = inputs['role']
    content = inputs['content']

    payload = ClientRequest(
        inputs=content,
        stream=True,
        parameters=parameters
    )

    headers = {
        "user-agent": "mindspore_serving/1.0"
    }
    cookies = None
    timeout = 30000

    with get_httpx_client() as client:
        # with client.stream("POST", url, headers=headers, json=payload) as response:
        #     for line in response.a:
        response = client.post(
            url,
            json=payload.dict(),
            headers=headers,
            cookies=cookies,
            timeout=timeout,
        )
        if stream:
            for byte_payload in response.iter_lines():
                if byte_payload == b"\n":
                    continue
                payload = byte_payload
                print(payload)
                if payload.startswith("data:"):
                    # Decode payload
                    json_payload = json.loads(payload.lstrip("data:").rstrip("/n"))
                    yield json_payload
        else:
            yield response.json()

class MindSporeWorker(ApiModelWorker):
    # BASE_URL = f'http://{MS_SERVER["host"]}:{MS_SERVER["port"]}'
    # SUPPORT_MODELS = ["llama2_7b", "llama2_70b"]

    def __init__(
        self,
        *,
        model_names: List[str] = ["mindspore-api"],
        version: Literal["llama2_7b", "llama2_70b"] = "llama2_7b",
        controller_addr: str,
        worker_addr: str,
        **kwargs
    ):
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 32000) # TODO: for what use?
        super().__init__(**kwargs)
        self.version = version

        # TODO: 确认模板是否需要修改
        self.conv = conv.Conversation(
            name=self.model_names[0],
            system_message="",
            messages=[],
            roles=["user", "assistant"],
            sep="\n### ",
            stop_str="###",
        )
        self.first_newline = True

    def text_postprocess(self, text: str):
        if text == "<0x0A>":
            if self.first_newline:
                self.first_newline = False
                return ""
            return "\n\n"
        return text.replace('\u2581', ' ')

    def generate_stream_gate(self, params):
        messages = self.prompt_to_messages(params["prompt"])
        full_text = ""
        for resp in request_mindspore_api(messages,
                                          temperature=params.get("temperature"),
                                          model_name=self.model_names[0]):
            if 'token' in resp:
                text = self.text_postprocess(resp['token']['text'])
                full_text += text
                yield json.dumps({
                    "error_code": 0,
                    "text": full_text
                },
                    ensure_ascii=False,
                ).encode() + b"\0"
            elif 'generated_text' in resp:
                text = self.text_postprocess(resp['generated_text'])
                full_text += text
                yield json.dumps({
                    "error_code": 0,
                    "text": full_text
                },
                    ensure_ascii=False,
                ).encode() + b"\0"
            else:
                # TODO: Add more error code
                yield json.dumps({
                    "error_code": resp['error_code'],
                    "text": resp['error_msg']
                },
                    ensure_ascii=False,
                ).encode() + b"\0"
