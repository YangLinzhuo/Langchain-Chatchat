from fastchat.conversation import Conversation
from server.model_workers.base import ApiModelWorker, ApiChatParams
from fastchat import conversation as conv
from configs.model_config import TEMPERATURE
from server.utils import get_model_worker_config, get_httpx_client
import json
from typing import List, Dict
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

    # Concat history messages
    content = "<s>"
    for msg in messages:
        role = msg['role']
        if role == "user":
            content += f"<|User|>:{msg['content']}<eoh>\n<|Bot|>:"
        else:   # role == "assistant"
            content += f"{msg['content']}\n"

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
        if stream:
            with client.stream('POST', url, json=payload.dict(), headers=headers,
                               cookies=cookies, timeout=timeout) as response:
                for payload in response.iter_lines():
                    if payload == b"\n":
                        continue
                    if payload.startswith("data:"):
                        # Decode payload
                        json_payload = json.loads(payload.lstrip("data:").rstrip("\n"))
                        yield json_payload
        else:
            response = client.post(
                url,
                json=payload.dict(),
                headers=headers,
                cookies=cookies,
                timeout=timeout,
            )
            yield response.json()


class MindSporeWorker(ApiModelWorker):
    def __init__(
            self,
            *,
            model_names,
            controller_addr: str,
            worker_addr: str,
            **kwargs
    ):
        kwargs.update(model_names=model_names, controller_addr=controller_addr, worker_addr=worker_addr)
        kwargs.setdefault("context_len", 32000)
        super().__init__(**kwargs)

    def make_conv_template(self, conv_template: str = None, model_path: str = None) -> Conversation:
        # TODO: 确认模板是否需要修改
        return conv.Conversation(
            name=self.model_names[0],
            system_message="",
            messages=[],
            roles=["user", "assistant"],
            sep="\n### ",
            stop_str="###",
        )

    def do_chat(self, params: ApiChatParams) -> Dict:
        full_text = ""
        for resp in request_mindspore_api(messages=params.messages,
                                          temperature=params.temperature,
                                          model_name=self.model_names[0]):
            if resp["event"] == "message":
                full_text += resp["data"]
                yield {
                    "error_code": 0,
                    "text": full_text
                }
            else:
                yield {
                    "error_code": 0,
                    "text": "Error Message"
                }
