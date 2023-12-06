from .bert import BertForEmbedding, BertEmbedding

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
    if model == "ms-bert-base":
        import mindspore
        from mindformers import BertTokenizer
        from configs import EMBEDDING_DEVICE_ID
        mindspore.set_context(mode=mindspore.GRAPH_MODE, device_id=EMBEDDING_DEVICE_ID, device_target=device)
        tokenizer = BertTokenizer.from_pretrained('bert_base_uncased')
        model = BertForEmbedding.from_pretrained(get_model_path(model, "embed_model"))
        embeddings = BertEmbedding(tokenizer, model, padding='max_length',
                                   max_length=model.config.seq_length, batch_size=1)
        # Invoke model compilation
        result = embeddings.embed_query("A")
        print(f"Query: {result}")
    else:
        raise ValueError(f"Unsupported mindspore embedding type {model}.")
    return embeddings
