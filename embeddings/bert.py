from langchain.embeddings.base import Embeddings
import mindspore
import mindspore.ops as ops

class BertEmbedding(Embeddings):
    def __init__(self, tokenizer, bert_model, padding, max_length, batch_size):
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.padding = padding
        self.max_length = max_length
        self.batch_size = batch_size
        mindspore.set_context(mode=mindspore.GRAPH_MODE, device_id=7, device_target="Ascend")

    def embed_documents(self, texts):
        texts_len = len(texts)
        embeddings = [[] for _ in range(texts_len)]
        i = 0
        while i < texts_len:
            batch_embedding = self.embed_batch(texts[i:i+self.batch_size])
            embeddings[i:i+self.batch_size] = batch_embedding
            i += self.batch_size
        return embeddings

    def embed_batch(self, texts: list):
        tokens = self.tokenizer(texts, return_tensors='ms', max_length=self.max_length, padding=self.padding)
        bert_output = self.bert_model(tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids'])
        embedding_batch = self.postProcess(bert_output, tokens['attention_mask'])
        return embedding_batch

    def embed_query(self, text: str):
        embedding = self.embed_batch([text])
        return embedding[0]

    def postProcess(self, bert_encode, attention_mask):
        # Mask and Mean for bert_embedding
        mask_expand = attention_mask.unsqueeze(-1).expand_as(bert_encode)

        sum_embedding = ops.sum(bert_encode * mask_expand, dim=1)
        sum_mask = mask_expand.sum(axis=1)
        sum_mask = ops.clamp(sum_mask, min=1e-9)
        sent_embedding = sum_embedding / sum_mask
        L2N = ops.L2Normalize(axis=1)
        output = L2N(sent_embedding)
        return output

