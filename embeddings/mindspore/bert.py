from langchain.embeddings.base import Embeddings
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.ops import operations as P

from mindformers.models.bert.bert import BertModel
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from mindformers.models.base_model import BaseModel
from mindformers.mindformer_book import MindFormerBook
from mindformers.models.bert.bert_config import BertConfig


@MindFormerRegister.register(MindFormerModuleType.MODELS)
class BertForEmbedding(BaseModel):
    """
    Bert with dense layer for embedding task.

    Args:
        config (BertConfig): The config of BertForEmbedding
    Returns:
        Tensor, loss, logits.
    """
    _support_list = MindFormerBook.get_model_support_list()['bert']

    def __init__(self, config=BertConfig()):
        super(BertForEmbedding, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, config.use_one_hot_embeddings)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.reshape = P.Reshape()
        self.load_checkpoint(config)

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_ids,
                  label_ids=None):
        """Get Training Loss or Logits"""
        bert_outputs = self.bert(input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)
        logits = bert_outputs[0]

        if self.training:
            logits = self.reshape(logits, (-1, self.num_labels))
            label_ids = self.reshape(label_ids, (-1, ))
            output = self.cross_entropy_loss(logits, label_ids)
        else:
            if label_ids is None:
                output = logits
            else:
                output = (logits, label_ids)

        return output


class BertEmbedding(Embeddings):
    def __init__(self, tokenizer, bert_model, padding, max_length, batch_size):
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.padding = padding
        self.max_length = max_length
        self.batch_size = batch_size

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
        """
        Returns:
            list[np.ndarray]: list of embedding array
        """
        tokens = self.tokenizer(texts, return_tensors='ms', max_length=self.max_length, padding=self.padding)
        bert_output = self.bert_model(tokens['input_ids'], tokens['attention_mask'], tokens['token_type_ids'])
        embedding_batch = self.postprocess(bert_output, tokens['attention_mask'])
        return [item.asnumpy() for item in embedding_batch]

    def embed_query(self, text: str):
        """
        Returns:
            np.ndarray: embedding array
        """
        embedding = self.embed_batch([text])
        return embedding[0]

    def postprocess(self, bert_encode, attention_mask):
        # Mask and Mean for bert_embedding
        mask_expand = attention_mask.unsqueeze(-1).expand_as(bert_encode)

        sum_embedding = ops.sum(bert_encode * mask_expand, dim=1)
        sum_mask = mask_expand.sum(axis=1)
        sum_mask = ops.clamp(sum_mask, min=1e-9)
        sent_embedding = sum_embedding / sum_mask
        L2N = ops.L2Normalize(axis=1)
        output = L2N(sent_embedding)
        return output
