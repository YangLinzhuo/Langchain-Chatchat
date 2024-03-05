# MindSpore-Langchain 介绍

本项目基于原 [LangChain-ChatChat](https://github.com/chatchat-space/Langchain-Chatchat) 项目修改而来，添加了对 MindSpore 框架的
适配代码。关于 LangChain 的基本配置和依赖，请参看 README_zh.md 或者 README_en.md。本文档主要介绍适配 MindSpore 框架推理相关的
内容。

LangChain 和 MS-Serving 服务是解耦的两个服务。用户输入 Query 后，LangChain 框架会经过一定处理，生成对应的 Prompt，并向 MS-Serving
服务发送包含 Prompt 的请求。MS-Serving 服务收到请求后，将请求分发给对应后端部署的大模型，获取生成的结果后，通过 Response 返回给
LangChain 框架，最终将结果显示给用户。

![langchain+ms流程](../img/langchain+ms-serving.png)

以下是 LangChain + MS-Serving 知识库详细流程示意图：

![langchain+ms知识库流程](../img/langchain+ms-serving+knowledgebase.png)


# 关于配置

为了便于日后同步原 LangChain-ChatChat 仓库的代码，本仓库 MindSpore 相关的配置全部放在 `configs/mindspore_config.py` 中，
注意需要在 `config/__init__.py` 中最后导入该配置，以动态修改原始的配置，尽量减少对原配置的侵入式修改。

框架的部分代码中存在直接导入 `config/xx_config.py` 模块中变量的情况，此时 `config/mindspore_config.py` 中的修改可能不会生效，
如果发现修改的配置未生效，可以通过查找 `from config` 开头的代码，看是否存在直接导入 `config/xx_config.py` 的代码，修改对应代码即可。

目前发现的以上情形暂时有以下两处：

- `server/utils.py` 的 `get_prompt_template` 函数，需要额外导入 `mindspore_config` 模块：
```python
    from configs import prompt_config, mindspore_config
    import importlib
    importlib.reload(prompt_config)
    importlib.reload(mindspore_config)
    return prompt_config.PROMPT_TEMPLATES[type].get(name)
```

- `init_databse.py`，需要修改导入的方式：
```python
# 原来是直接 from configs.model_config import xxx
from configs import NLTK_DATA_PATH, EMBEDDING_MODEL
```

其他修改：

- `startup.py`: 预加载、`run_api_server` 进程启动参数 `daemon=False`
- `config\__init__.py`：导入 `mindspore_config`
- `server/knowledge_base/kb_cache/base.py`：处理 `mindspore` 框架 Embedding 模型导入
- `server/knowledge_base/utils.py`：`make_text_splitter` 函数添加处理 `mindformers` tokenizer 分支

# 环境配置

镜像及软件包文件请在 `issue` 中提供邮箱地址，将分享链接通过邮箱发送给您。

- 解压文件后，解压 `zip` 包后，导入其中带有 `tar` 后缀的镜像包，例如：
```shell
docker load < mindspore_serving.tar
```
- 启动导入的镜像：
```shell
docker run -it --ipc=host \
--net=host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /var/log/npu/:/usr/slog \
mindspore_serving:{tag} \
/bin/bash
```
注意需要添加 `--net=host`，以支持外部访问容器的端口。`--device` 指定加载的昇腾设备编号，默认全部加载，可以根据情况选择仅加载部分设备。
`mindspore_serving:{tag}` 为导入的镜像名称及 `tag`，如果实际有差异，请替换为实际的容器镜像名称及 `tag`。

- 进入容器后，安装软件包中提供的 `mindspore` 和 `mindspore-lite` 的 `whl` 包：
```shell
pip install mindspore-2.2.10-cp39-cp39-linux_aarch64.whl
pip install mindspore_lite-2.2.10-cp39-cp39-linux_aarch64.whl
```

- 检查是否安装了 `mindspore-serving` 模块，如果已安装，请卸载原镜像中的包。
- `mindfomers` 和 `serving` 文件夹是后续使用的 `mindformers` 套件和 `serving` 套件的对应分支代码，可以直接使用

# Bert Embedding 模型配置

本仓库使用支持 Bert-Base 和 BGE 作为基础的 Embedding 模型，而非原项目中默认使用的 [moka-ai/m3e-base](https://huggingface.co/moka-ai/m3e-base)。

## Bert-Base 模型配置
该模型依赖 [mindformers 套件](https://gitee.com/mindspore/mindformers)，需要按照教程安装 `mindformers` 套件。
在 [HuggingFace](https://huggingface.co/bert-base-chinese) 上下载中文 Bert-base 权重以及对应的 `vocab.txt` 文件，参考
[文档](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/bert.md) 转换成 `ckpt` 格式的权重，注意 `--layers 12` 参数不能少。

转换命令：

```shell
python convert_weight.py --layers 12 --torch_path path/to/pytorch_mode.bin --mindspore_path ./ms_bert_base.ckpt
```

在 `configs/mindspore_config.py` 文件中，修改 `MS_MODEL_MODEL` 字典中 `embed_model` 项下的 `ms-bert-base` 以配置本地权重的路径，
路径只需写到权重文件的上一层文件夹即可。例如权重文件路径为 `/home/bert/bert.ckpt`，那么路径只需要写到 `/home/bert` 即可。

为了和 `mindformers` 套件的配置保持一致，推荐在框架代码根目录下创建 `checkpoint_download/bert` 路径，在该路径中存放权重文件。

```python
global MODEL_PATH
MS_MODEL_PATH = {
    "embed_model": {
        "ms-bert-base": "checkpoint_download/bert",
        "ms-bge": "checkpoint_download/bge",
    },
}
MODEL_PATH["embed_model"].update(MS_MODEL_PATH["embed_model"])

```

在执行 `startup.py` 的目录下，`mindformers` 套件会自动下载 `Bert` 相关的配置文件，存放在 `checkpoint_download/bert` 路径下，名称
为 `bert_base_uncased.yaml`（本仓库中在 `checkpoint_download` 文件夹下预置了 Bert 配置，可以下载权重后放入其中直接使用）。在某些情况下，可能预设的 `seq_len` 长度不够，可以修改配置文件中 `seq_len` 选项：

```yaml
model:
  model_config:
    # other configs ...
    seq_len: 512 # 最大不超过 512
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

如果不需要用到知识库功能，可以将这部分代码注释掉。

## BGE 模型配置

BGE 模型配置和上文 Bert-Base 基本一致，修改 `MS_EMBEDDING_MODEL = "ms-bge"` 即可选择使用 BGE 来作为 Embedding 模型。

权重下载地址：https://huggingface.co/BAAI/bge-large-zh

需要注意的是，BGE 模型和 Bert 模型结构稍有不同，需要使用 `checkpoint_download/bge/convert_weight.py` 脚本转换，使用该文件夹下的
`bge.yaml` 作为模型配置文件。注意使用 `convert_weight.py` 转换时需要添加参数 `--layers 24`。

转换命令：

```shell
python convert_weight.py --layers 24 --torch_path path/to/pytorch_mode.bin --mindspore_path ./ms_bge.ckpt
```

# 配置 MindSpore Serving 服务

本项目后端大模型基于 [MindSpore Serving](https://gitee.com/mindspore/serving) 仓库的 2.1 分支。

当前暂时支持 `LLaMA2-70B` 和 `InternLM-20B`，后续会提供其他模型的支持。下面以 `InternLM-20B` 为例。

## 导出 `MindIR` 模型

以 `InternLM-20B` 模型为例。

MindSpore Serving 服务需要使用 `MindIR` 格式的模型。需要使用 [mindformers 套件](https://gitee.com/mindspore/mindformers)
的 `ft-predict-opt` 分支。参考对应模型的教程导出 `mindir` 格式的模型文件。

- 下载 [HuggingFace 权重文件](https://huggingface.co/internlm/internlm-chat-20b/tree/v1.0.2)，注意是 v1.0.2 版本，其他的小文件也需要下载，在转换权重时都需要。
- 转换权重为 `ckpt` 文件：使用 `mindformers` 套件中的转换脚本转换，路径在 `research/internlm/convert_weight.py`。
```shell
python convert_weight.py --torch_ckpt_dir /path/to/torch/checkpoint --mindspore_ckpt_path /path/to/save/ckpt
```
`mindspore_ckpt_path` 需要提供文件名，不能是文件夹名称。
- 导出 `mindir` 格式文件，注意单卡只支持 `batch_size=1`，导出时文件配置可以参考 `server/model_workers/internlm_config/run_internlm_20b_910b_1p.yaml`，注意修改以下内容：

```yaml
...
load_checkpoint: "/home/ckpt/internlm.ckpt"  # 设置为本地权重路径
...

infer:
  prefill_model_path: "/home/internlm-mindir/prefill.mindir"   # 导出文件本地路径
  increment_model_path: "/home/internlm-mindir/inc.mindir" # 导出文件本地路径
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

- 配置文件修改完成后，将 `yaml` 配置文件和转换后的权重文件、以及 `tokenizer` 模型文件放在相同文件夹下，使用 `mindformers` 中的导出脚本导出 `mindir` 模型，文件路径在 `mindformers/tools/export.py`：
```shell
python export.py --model_dir /path/to/ckpt/dir/
```

导出大约需要 30~40 分钟，请耐心等待。

## 修改 MS-Serving 配置

启动 MS-Serving 之前，需要配置 `mindspore-lite` 相关设置。如果前文的导出模型成功，在目标文件夹下应该能够看到带有 `prefill` 和 `inc` 名称的 `.mindir` 文件。

在本仓库的 `server/model_workers/internlm_config` 文件夹下，准备了 `*.ini` 和 `*.yaml` 文件用于 `mindspore-lite` 调用导出的 MindIR 模型，`serving_config.py` 用于配置 MS-Serving 服务。


使用 `server/model_workers/internlm_config/serving_config.py` 替换 `serving` 仓库下 `config/serving_config.py` 的文件，修改对应 `mindir` 文件路径：

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

其中 `argmax_model` 和 `topk_model` 需要运行 `serving` 仓库中的 `post_sampling_model.py` 脚本生成，并修改为生成的文件路径，默认在当前路径下的 `extends` 文件夹中。

修改对应 `ini` 文件路径，`ini` 文件可以使用 `server/model_workers/internlm_config` 下的 `ini` 文件：

```python
ctx_path = '/path/to/xx_lite_full.ini'     # 填写 xx_lite_full.ini 路径
inc_path = [
    '/path/to/xx_lite_inc.ini',            # 填写 xx_lite_inc.ini 路径
]

post_model_ini = '/path/to/config.ini'          # 填写 config.ini 路径
tokenizer_path = '/path/to/tokenizer.model'     # 填写 tokenizer.model 路径
```

带有 `prefill` 和 `inc` 的配置文件对应 `ctx_path` 和 `inc_path`，`config.ini` 对应 `post_model_ini` 配置，`tokenizer_path` 是对应模型的 tokenizer 文件路径。


## 启动 MindSpore Serving 服务

在 `serving` 仓库代码的根目录，将当前路径添加到 `PYTHONPATH` 环境变量：

```shell
export PYTHONPATH=$(pwd):${PYTHONPATH}
```

注意如果镜像中原来安装了 `mindspore-serving`，请先卸载，否则修改的配置可能不生效。

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

端口号和设备可以根据情况修改。后续启动 LangChain 框架时填写的端口号需要和该设置对应。

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

## 生成向量库

首先设置环境变量，将 `mindformers` 代码的根路径添加到 `PYTHONPATH` 环境变量中：

```shell
export PYTHONPATH=/path/to/mindformers:${PYTHONPATH}
```

框架启动前，需要使用 `python init_database.py -r` 命令生成向量库，需要使用到 Embedding 模型以及后端大模型的 tokenizer。

相关配置如下：

```python
MS_MODEL_PATH = {
    "embed_model": {
        "ms-bert-base": "checkpoint_download/bert",
        "ms-bge": "checkpoint_download/bge",
    },
}

# 选用的 Embedding 名称
MS_EMBEDDING_MODEL = "ms-bge"

MS_ONLINE_LLM_MODEL = {
    "mindspore-api": {
        ...
        "model_path": "checkpoint_download/internlm/"    # used for load corresponding tokenizer
    },
}
```

其中 `MS_ONLINE_LLM_MODEL` 中配置的 `model_path` 用于切分文档时使用，建议填写后端大模型对应的配置文件及 tokenizer.model 所在文件夹
路径，提升文档切分的效果。可以直接使用转换 `MindIR` 格式的权重时所使用的配置文件及 tokenizer 文件。

修改 `configs/kb_config.py` 中的分词器配置：

```python
# TextSplitter配置项，如果你不明白其中的含义，就不要修改。
text_splitter_dict = {
    "ChineseRecursiveTextSplitter": {
        "source": "mindformers",   # <---- modify here
        "tokenizer_name_or_path": "",
    },
    ...
}
```

修改 `source` 为 `mindformers` 使用对应后端大模型的 tokenizer。否则框架会使用默认的分词器，导致知识库匹配效果较差。

## 服务配置

使用 `python copy_config_example.py` 生成 `configs` 下的配置文件。

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
        "model_type": "internlm",
        "model_path": "checkpoint_download/internlm/",   # used for load corresponding tokenizer
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
    "port": 9889
}
```

## 启动服务

启动前有几点需要注意：

- 关闭网络代理，否则可能无法路由到本地 `ip`
- 网页如果无法访问，可能需要关闭防火墙（可选）：`systemctl stop firewalld`
- 设置环境变量：`export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`

首先执行 `python init_database.py -r` 初始化数据库（执行前需要先配置好 Bert）。 

然后使用 `python startup.py -a -n mindspore-api` 即可启动基于 MS-Serving 后端的 LangChain-ChatChat 框架。

启动之后，访问服务器对应的 `{ip}:{port}` 即可进入到对话页面，这里的 `ip` 是服务器连接的 `ip`，不是本地 `0.0.0.0`。


终止服务后，可以用以下命令关闭所有进程：

```shell
pkill -f -9 startup.py
pkill -f -9 webui.py
pkill -f -9 multiprocessing.spawn
```
