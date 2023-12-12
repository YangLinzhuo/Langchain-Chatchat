# MindSpore-Langchain 介绍

本项目基于原 [LangChain-ChatChat](https://github.com/chatchat-space/Langchain-Chatchat) 项目修改而来，添加了对 MindSpore 框架的
适配代码。关于 LangChain 的基本配置和依赖，请参看 README_zh.md 或者 README_en.md。本说明主要介绍支持 MindSpore 框架推理相关的
主要修改点。

LangChain 和 MS-Serving 服务是解耦的两个服务。用户输入 Query 后，LangChain 框架会经过一定处理，生成对应的 Prompt，并向 MS-Serving
服务发送包含 Prompt 的请求。MS-Serving 服务收到请求后，将请求分发给对应后端部署的大模型，获取生成的结果后，通过 Response 返回给
LangChain 框架，最终将结果显示给用户。

![langchain+ms流程](../img/langchain+ms-serving.png)

以下是 LangChain + MS-Serving 知识库详细流程示意图：

![langchain+ms知识库流程](../img/langchain+ms-serving+knowledgebase.png)


# 关于配置

为了便于日后同步原 LangChain-ChatChat 的代码，本仓库 MindSpore 相关的配置全部放在 `configs/mindspore_config.py` 中，
在 `config/__init__.py` 中最后导入该配置，以动态修改原始的配置，尽量减少对配置的侵入式修改。

框架的部分代码中存在直接导入 `config/xx_config.py` 模块中变量的情况，此时 `config/mindspore_config.py` 中的修改可能不会生效，
如果发现修改的配置未生效，可以通过查找 `from config` 开头的代码，看是否存在直接导入 `config/xx_config.py` 的情况，修改对应代码即可。

目前发现的以上情形暂时有以下两处：

- `server/utils.py` 的 `get_prompt_template` 函数
```python
from configs import prompt_config, mindspore_config
    import importlib
    importlib.reload(prompt_config)
    importlib.reload(mindspore_config)
    return prompt_config.PROMPT_TEMPLATES[type].get(name)
```

- `init_databse.py`
```python
# 原来是直接 from configs.model_config import xxx
from configs import NLTK_DATA_PATH, EMBEDDING_MODEL
```


# Bert Embedding 模型

本仓库使用 Bert-base 作为基础的 Embedding 模型，而非原项目中默认使用的 [moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base)。


该模型依赖 [mindformers 套件](https://gitee.com/mindspore/mindformers)，需要按照教程安装 `mindformers` 套件。
在 [HuggingFace](https://huggingface.co/bert-base-chinese) 上下载中文 Bert-base 权重之后，参考
[文档](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/bert.md) 转换成 `ckpt` 格式的权重。

在 `configs/mindspore_config.py` 文件中，修改 `MS_MODEL_MODEL` 字典中 `embed_model` 项下的 `ms-bert-base` 以配置本地权重的路径，
路径只需写到权重文件的上一层文件夹即可。例如权重文件路径为 `/home/bert/bert.ckpt`，那么路径只需要写到 `/home/bert` 即可。

为了和 `mindformers` 套件的配置保持一致，推荐在框架代码根目录下创建 `checkpoint_download/bert` 路径，在该路径中存放权重文件。

```python
global MODEL_PATH
MS_MODEL_PATH = {
    "embed_model": {
        "ms-bert-base": "path to bert base checkpoint"
    },
}
MODEL_PATH["embed_model"].update(MS_MODEL_PATH["embed_model"])

```

在执行 `startup.py` 的目录下，`mindformers` 套件会自动下载 `Bert` 相关的配置文件，存放在 `checkpoint_download/bert` 路径下，名称
为 `bert_base_uncased.yaml`。在某些情况下，可能预设的 `seq_len` 长度不够，可以修改配置文件中 `seq_len` 选项：

```yaml
model:
  model_config:
    # other configs ...
    seq_len: 1024
    # other configs ...
```

在框架中，Embedding 模型相关的代码存放在 `embedding/mindspore` 文件夹下。`embedding/mindspore/__init__.py`
中的 `get_mindspore_embedding` 函数用于导入指定的 Embedding 模型。`embedding/mindspore/bert.py` 中是 Bert Embedding 模型的实现。

`server/knowledge_base/kb_cache/base.py` 中的 `EmbeddingsPool` 添加了处理 `MindSpore` Embedding 模型相关的部分：

```python
class EmbeddingsPool(CachePool):
    def load_embeddings(self, model: str = None, device: str = None) -> Embeddings:
        ...
        if not self.get(key):
            item = ThreadSafeObject(key, pool=self)
            self.set(key, item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                if model == "text-embedding-ada-002":  # openai text-embedding-ada-002
                    ...
                elif 'bge-' in model:
                    ...
                elif model.startswith('ms-'):
                    from embeddings.mindspore import get_mindspore_embedding
                    embeddings = get_mindspore_embedding(model, device)
                else:
                    ...
                item.obj = embeddings
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get(key).obj
```

由于 MindSpore 框架的原因，加载 Bert 模型时需要开启多线程加载，需要将原来框架设定的 `daemon` 参数改为 `False`:

```python
    ...
    if args.api:
        process = Process(
            target=run_api_server,
            name=f"API Server",
            kwargs=dict(started_event=api_started, run_mode=run_mode),
            daemon=False,   # For loading mindspore embedding model
        )
        processes["api"] = process
    ...
```

由于模型初次运行需要编译，因此采用了预加载的方式，在初次运行框架时提前加载网页，减少用户感知：

```python
def run_api_server(started_event: mp.Event = None, run_mode: str = None):
    ...

    print(f"预加载 Embedding 模型")
    embeddings_pool.load_embeddings(EMBEDDING_MODEL, embedding_device())
    print(f"预加载 Embedding 模型完毕")

    ...
```

# 配置 MindSpore Serving 服务

本项目后端大模型基于 [MindSpore Serving](https://gitee.com/mindspore/serving) 仓库的 2.1 分支。

## 环境配置

安装对应版本的 MindSpore 和 MindSpore-lite:

- [MindSpore 2.2.10 Python 3.9 版本](https://repo.mindspore.cn/mindspore/mindspore/version/202311/20231130/r2.2.10_20231130221514_da8ea6513d0a23b41149e846c3a38891e93c3919/unified/aarch64/mindspore-2.2.10-cp39-cp39-linux_aarch64.whl)
- [MindSpore-lite 2.2.10 Python 3.9 版本](https://repo.mindspore.cn/mindspore/mindspore/version/202311/20231130/r2.2.10_20231130221514_da8ea6513d0a23b41149e846c3a38891e93c3919/lite/linux_aarch64/cloud_fusion/python39/mindspore_lite-2.2.10-cp39-cp39-linux_aarch64.whl)

## 导出 `MindIR` 模型

MindSpore Serving 服务需要使用 `MindIR` 格式的模型。需要使用 [mindformers 套件](https://gitee.com/mindspore/mindformers)
的 `ft-predict-opt` 分支。参考对应模型的教程导出 `mindir` 格式的模型文件。

- 下载 HuggingFace 权重文件
- 转换权重为 `ckpt` 文件：使用 `mindformers` 套件中的转换脚本转换，具体可以参考每个模型各自的详细文档
- 导出 `mindir` 格式文件，单卡只支持 `batch_size=1`

导出时文件配置可以参考 `server/model_workers/internlm_config/run_internlm_20b_910b_1p.yaml`，注意修改以下内容：

```yaml
...
load_checkpoint: "/home/ckpt/internlm.ckpt"  # 设置为本地权重路径
...

infer:
  prefill_model_path: "/home/internlm-mindir/prefill.mindir"   # 导出文件本地路径
  increment_model_path: "/home/internlm-mindir/prefill.mindir" # 导出文件本地路径
  infer_seq_length: 4096
  model_type: mindir
...
# model config
model:
  model_config:
    ...
    checkpoint_name_or_path: "/home/internlm/internlm.ckpt" # 修改为本地权重路径
    ...
processor:
  return_tensors: ms
  tokenizer:
    ...
    type: InternLMTokenizer
    vocab_file: '/home/internlm/tokenizer.model' # 本地 tokenizer 路径
  type: LlamaProcessor
```

## 修改配置

启动 MindSpore Serving 之前，需要配置 `mindspore-lite` 相关设置。`server/model_workers/internlm_config` 下的
`internlm_lite_full.ini` 和 `internlm_lite_inc.ini` 用于 `mindspore-lite` 模型启动配置。


使用 `serving_config.py` 替换 `serving` 仓库下 `config/serving_config.py` 的文件，修改对应 `mindir` 文件路径：

```python
MINDIR_ROOT = "/path/to/mindir/directory"
prefill_model_path = [
    f"{MINDIR_ROOT}/prefill_graph.mindir"
]
decode_model_path = [
    f"{MINDIR_ROOT}/path/to/inc_graph.mindir"
]
argmax_model = ["/path/to/argmax.mindir"]
topk_model = ["/path/to/topk.mindir"]
```

其中 `argmax_model` 和 `topk_model` 需要运行 `serving` 仓库中的 `post_sampling_model.py` 脚本生成，并修改为生成的文件路径。

修改对应 `ini` 文件路径，`ini` 文件可以使用 `server/model_workers/internlm_config` 下的 `ini` 文件：

```python
ctx_path = '/path/to/xx_lite_full.ini'     # 填写 xx_lite_full.ini 路径
inc_path = [
    '/path/to/xx_lite_inc.ini',            # 填写 xx_lite_inc.ini 路径
]

post_model_ini = '/path/to/config.ini'          # 填写 config.ini 路径
tokenizer_path = '/path/to/tokenizer.model'     # 填写 tokenizer.model 路径
```


## 启动 MindSpore Serving 服务

克隆 MindSpore Serving 仓库后，进入 `serving` 目录，将当前路径添加到 `PYTHONPATH` 环境变量：

```shell
export PYTHONPATH=${PYTHONPATH}:$(pwd)
```


通过 `start_agent.py` 启动 agent 服务，加载模型需要花费较长时间，请耐心等待。推荐使用以下命令后台运行：

```shell
nohup python start_agent.py > agent.log 2>&1 &
```

通过 `client/server_app_post.py` 启动后台服务。server 的端口配置在 `config/serving_config.py` 文件中：

```python
SERVER_APP_HOST = '0.0.0.0'
SERVER_APP_PORT = 9889
device = 6  # 用于加载模型的昇腾芯片编号
```

# 启动 LangChain-ChatChat 服务

## 安装依赖

vllm 依赖会安装依赖 CUDA 的 `torch`，可以在 `requirements.txt` 文件中注释掉，不影响使用。

然后使用以下命令安装依赖：

```shell
pip install -r requirements.txt
pip install -r requirements_api.txt
pip install -r requirements_lite.txt
pip install -r requirements_webui.txt
```

## 服务配置

可以配置 `configs/mindspore_config.py` 文件，在 `MS_ONLINE_LLM_MODEL` 项目下
添加 `mindspore-api` 项，可以配置 `model_type` 选项来选择后端模型，目前暂时支持 `InternLM-20B`。

```python
global LLM_MODELS
# LLM 名称
MS_LLM_MODEL = "mindspore-api"
LLM_MODELS = [MS_LLM_MODEL]

# LLM 运行设备。设为"auto"会自动检测，也可手动设定为"ascend","cuda","mps","cpu"其中之一。
# 对于调用远端 API 运行的模型，该设置无效
MS_LLM_DEVICE = "auto"
LLM_DEVICE = MS_LLM_DEVICE

global ONLINE_LLM_MODEL
MS_ONLINE_LLM_MODEL = {
    "mindspore-api": {
        "version": "InternLM-20B",
        "api_key": "EMPTY",
        "secret_key": "",
        "provider": "MindSporeWorker",
        "model_type": "internlm"
    },
}
ONLINE_LLM_MODEL.update(MS_ONLINE_LLM_MODEL)
```

MindSpore Serving 服务的 `ip` 和端口地址在 `configs/mindspore_config.py` 中的 `MS_SERVER` 中配置，`ip` 和端口号为
启动 MindSpore Serving 服务时配置的 `ip` 和端口号。

示例如下：

```python
MS_SERVER = {
    "host": "0.0.0.0",
    "port": 1234
}
```

## 启动服务

使用 `python startup.py -a -n mindspore-api` 即可启动基于 MindSpore Serving 后端的 LangChain-ChatChat 框架。

启动前有几点需要注意：

- 关闭网络代理，否则可能路由到本地 `ip`
- 网页如果无法访问，可能需要关闭防火墙（可选）：`systemctl stop firewalld`
- 设置环境变量：`export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`

启动之后，访问服务器对应的 `{ip}:{port}` 即可进入到对话页面，这里的 `ip` 是服务器连接的 `ip`，不是本地 `0.0.0.0`。


终止服务后，可以用以下命令关闭所有进程：

```shell
pkill -f -9 startup.py
pkill -f -9 webui.py
pkill -f -9 multiprocessing.spawn
```
