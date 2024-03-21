from fastchat.conversation import Conversation
from server.model_workers.base import ApiModelWorker, ApiChatParams
from fastchat import conversation as conv
from configs import MS_HISTORY_PROMPT_TEMPLATES, logger, log_verbose
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
    params: ApiChatParams,
    model_name: str = "mindspore-api",
    version: str = None,
):
    messages: List[Dict[str, str]] = params.messages
    temperature: float = params.temperature
    root_url = f'http://{MS_SERVER["host"]}:{MS_SERVER["port"]}'
    config = get_model_worker_config(model_name)
    version = version or config.get("version")
    route = config.get("route")

    url = f"{root_url}/models/{route}"
    stream = config.get("stream", True)
    parameters = Parameters(
        do_sample=False,
        repetition_penalty=1.0,
        temperature=temperature,
        top_k=3,
        top_p=params.top_p,
        max_new_tokens=None,
        return_full_text=False,
    )

    # Concat history messages
    history_prompt = MS_HISTORY_PROMPT_TEMPLATES.get(version, None)
    if history_prompt is None:
        history_prompt = MS_HISTORY_PROMPT_TEMPLATES['default']
    content = history_prompt['instruction']
    for msg in messages:
        role = msg['role']
        if role == "user":
            content += history_prompt['user'].format(msg['content'])
        elif role == "assistant":
            content += history_prompt['assistant'].format(msg['content'])
        else:
            raise ValueError(f"Invalid role value: {role}")
    content += history_prompt['post']

    payload = ClientRequest(
        inputs=content,
        stream=stream,
        parameters=parameters
    )

    headers = {
        "user-agent": "mindspore_serving/1.0"
    }
    cookies = None
    timeout = 30000

    if log_verbose:
        prefix = "MindSporeWorker"
        logger.info(f"{prefix}:data: {payload.dict()}")
        logger.info(f"{prefix}:url: {url}")
        logger.info(f"{prefix}:headers: {headers}")

    with get_httpx_client() as client:
        if stream:
            with client.stream('POST', url, json=payload.dict(), headers=headers,
                               cookies=cookies, timeout=timeout) as response:
                for msg in response.iter_lines():
                    msg_dict = json.loads(msg)
                    yield msg_dict
        else:
            response = client.post(url, json=payload.dict(), headers=headers, cookies=cookies, timeout=timeout)
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
        for resp in request_mindspore_api(params=params, model_name=self.model_names[0]):
            if "event" in resp:     # stream
                full_text += resp["data"][0]["generated_text"]
                yield {
                    "error_code": 0,
                    "text": full_text
                }
            else:
                yield {
                    "error_code": 0,
                    "text": resp["generated_text"]
                }
