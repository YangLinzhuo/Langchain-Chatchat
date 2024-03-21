
# 环境准备

1. 镜像（包含 cann 包，mindspore, mindspore lite）
   - 链接：https://pan.baidu.com/s/1KKaLgQOxCQaKBx4-8Fhctw
   - 密码：W6z7
2. mindformers 源码：`git clone git@gitee.com:mindspore/mindformers.git -b dev` (commit id: 31804b268d03b2f3f831a5b7e6fb272c1d963fb3)
3. serving 源码：`git clone git@gitee.com:mindspore/serving.git -b dev` (commit id: 5bbae09bec83a94dcd7ce98f451d8b332f8cfc30)

# Baichuan2-13B 容器部署

## 基于 mindformers 导出推理模型 （`cd /path/mindformers`）

**step 1. 模型转换（PyTorch -> MindSpore）**
```shell
python ./research/baichuan/convert_weight.py --torch_ckpt_dir TORCH_CKPT_DIR --mindspore_ckpt_path MS_CKPT_NAME

# 参数说明
TORCH_CKPT_DIR: huggingface 权重保存目录路径
MS_CKPT_NAME: 权重保存文件名，保存为 TORCH_CKPT_DIR/OUTPUT_NAME，可自定义保存路径
```

**step 2. MindIR 导出（适用于 MindSpore Lite 推理）**

（1）修改配置文件 `research/baichuan2/export_baichuan2_13b.yaml` 中如下参数：

```yaml
seq_length: 4096
use_kvcache_op: False
use_paged_attention: True
checkpoint_name_or_path: "/path/to/baichuan2_13b.ckpt" # 导出任务这里必填" # 第一步转换的 ckpt 文件路径
```

（2）执行 `export.py`，完成模型导出：

```shell
cd research/baichuan2/
python run_baichuan2.py \
    --config export_baichuan2_13b.yaml \
    --run_mode export \
    --use_parallel False \
    --device_id 0
```

说明：导出的模型会存放在当前目录的 `output` 文件夹下。

## 基于 serving 模型服务化部署（`cd /path/serving`）

**step 1. 创建如下 3 个推理配置文件**

（1）按照如下内容，创建全量配置文件：`baichuan2_13b_prefill.cfg`

```conf
[ascend_context]
provider=ge

[ge_session_options]
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype

[graph_kernel_param]
opt_level=2
enable_cce_lib=true
disable_cce_lib_ops=MatMul
disable_cluster_ops=MatMul,Reshape

[ge_graph_options]
ge.inputShape=batch_valid_length:1;slot_mapping:-1;tokens:1,-1
ge.dynamicDims=1024,1024;1536,1536;4096,4096
ge.dynamicNodeType=1
```

（2）按照如下内容，创建增量配置文件：baichuan2_13b_inc.cfg

```conf
[ascend_context]
provider=ge

[ge_session_options]
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.event=notify
ge.exec.staticMemoryPolicy=2
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype

[graph_kernel_param]
opt_level=2
enable_cce_lib=true
disable_cce_lib_ops=MatMul
disable_cluster_ops=MatMul,Reshape

[ge_graph_options]
ge.inputShape=batch_valid_length:16;block_tables:16,32;slot_mapping:16;tokens:16,1
```

block_tables 参数说明：

- 16：代表 batch_size
- 32：代表 max_seq_len/block_size

（3）按照如下内容，创建 serving 后处理配置文件：`config.ini`

```ini
[ascend_context]
provider=ge

[ge_session_options]
ge.externalWeight=1
ge.exec.atomicCleanPolicy=1
ge.exec.staticMemoryPolicy=2
ge.exec.formatMode=1
ge.exec.precision_mode=must_keep_origin_dtype
```

**step 2. serving 后处理模型生成**

`tools/post_sampling_model.py` 第 51 行修改为：

```python
input_ids_dyn = Tensor(shape=[None, None, None], dtype=mstype.float32)
```

执行 `python tools/post_sampling_model.py --output_dir ./target_dir`

- `output_dir`：后处理模型生成的目录，生成 `argmax,mindir` 与 `topk.mindir`


**step 3. 服务化部署**

（1）修改 `serving/configs/baichuan2_13b_pa.yaml` 配置文件：

```yaml
model_path:
    prefill_model: ["/path/baichuan2/mindir_full_checkpoint/rank_0_graph.mindir"]   # 导出的全量模型
    decode_model: ["/path/baichuan2/mindir_inc_checkpoint/rank_0_graph.mindir"]     # 导出的增量模型
    argmax_model: "/path/post_process/argmax.mindir"                                # 上一步生成的后处理模型
    topk_model: "/path/post_process/topk.mindir"                                    # 上一步生成的后处理模型
    prefill_ini : ['/path/config_ini/baichuan2_13b_prefill.cfg']                    # 前文配置的全量模型配置
    decode_ini: [/path/config_ini/baichuan2_13b_inc.cfg']                           # 前文配置的增量模型配置
    post_model_ini: '/path/to/config.ini'                                           # 前文配置的后处理模型配置

model_config:
    model_name: 'baichuan2pa'
    max_generate_length: 4096
    end_token: 2
    seq_length: [1024,1536,4096]
    vocab_size: 125696
    prefill_batch_size: [1]
    decode_batch_size: [16]  # [16]
    zactivate_len: [4096]   # PA 场景不需要关注
    model_type: 'dyn'
    seq_type: 'static'
    batch_waiting_time: 0.0
    decode_batch_waiting_time: 0.0
    batching_strategy: 'continuous'
    current_index: False
    page_attention: True
    model_dtype: "DataType.FLOAT32"
    pad_token_id: 0

serving_config:
    agent_ports: [61166]    # 如果端口被占用，可配置其他端口，使用 lsof -i:<port number> 查询
    start_device_id: 0
    server_ip: '0.0.0.0'
    server_port: 61170      # 如果端口被占用，可配置其他端口，使用 lsof -i:<port number> 查询

pa_config:
    num_blocks: 512
    block_size: 16
    decode_seq_length: 4096

tokenizer:
    type: Baichuan2Tokenizer
    vocab_file: '/path//to/tokenizer.model'     # 填写 Baichuan2 模型的 tokenizer.model 路径

basic_inputs:
    type: LlamaBasicInputs

extra_inputs:
    type: LlamaExtraInputs

warmup_inputs:
    type: LlamaWarmupInputs
```

（2）启动服务

```shell
python examples/start.py --config configs/baichuan2_13b_pa.yaml > start.log 2>&1 &
```

`start.log` 如下日志表示启动成功：

```log
----agents are ready----
----starting server----
----server is ready----
```


**step 4. 服务调用**

非流式返回：

```shell
curl 127.0.0.1:61170/models/llama2/generate -X POST \
    -d '{"inputs":"Hello?","parameters":{"max_new_tokens":1000, "do_sample":"False", "top_p":"0.8", "top_k":"1", "return_full_text":"True"}}' \
    -H 'Content-Type: application/json'
```

流式返回：

```shell
curl 127.0.0.1:61170/models/llama2/generate_stream -X POST \
    -d '{"inputs":"Hello?","parameters":{"max_new_tokens":1000, "do_sample":"False", "top_p":"0.8", "top_k":"1", "return_full_text":"True"}, "stream":"True"}' \
    -H 'Content-Type: application/json'
```
