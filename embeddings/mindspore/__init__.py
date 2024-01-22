from .bert import BertForEmbedding, BertEmbedding, BgeEmbedding

from server.utils import get_model_path


MS_DEVICE_MAP = {
    "cpu": "CPU",
    "cuda": "GPU",
    "ascend": "Ascend"
}


def get_mindspore_embedding(model: str, device: str):
    if device.lower() not in ("cpu", "ascend", "cuda"):
        raise ValueError(f"Unsupported mindspore device type {device}.")
    device = MS_DEVICE_MAP[device.lower()]
    if model in ("ms-bert-base", "ms-bge"):
        import mindspore
        from mindformers import BertTokenizer
        from configs import EMBEDDING_DEVICE_ID
        mindspore.set_context(mode=mindspore.GRAPH_MODE, device_id=EMBEDDING_DEVICE_ID, device_target=device)
        model_path = get_model_path(model, "embed_model")
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model_name = model
        model = BertForEmbedding.from_pretrained(model_path)
        if model_name == "ms-bert-base":
            embeddings = BertEmbedding(tokenizer, model, padding='max_length',
                                       max_length=model.config.seq_length, batch_size=1)
        else:
            embeddings = BgeEmbedding(tokenizer, model, padding='max_length',
                                      max_length=model.config.seq_length, batch_size=1,
                                      query_instruction="为这个句子生成表示以用于检索相关文章：")
        # Invoke model compilation
        _ = embeddings.embed_query("A")
    else:
        raise ValueError(f"Unsupported mindspore embedding type {model}.")
    return embeddings
