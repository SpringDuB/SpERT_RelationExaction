"""
模型预测器，封装好的，专门用于模型部署 模型的推理
"""
import torch
from transformers import BertTokenizer

from spert import util, prediction
from spert.input_reader import JsonPredictionInputReader
from spert.models import load_model
from spert.sampling import create_eval_sample, collate_fn_padding


class SpertPredictor(object):
    def __init__(self, model_type, model_path, cache_path, tokenizer_path, types_path, max_pairs=100, prop_drop=0.0,
                 size_embedding=25, max_span_size=10, rel_filter_threshold=0.4, cpu=True):
        super(SpertPredictor, self).__init__()

        self.model_type = model_type
        self.model_path = model_path
        self.cache_path = cache_path
        self.tokenizer_path = tokenizer_path
        self.types_path = types_path
        self.lowercase = True
        self.max_pairs = max_pairs
        self.prop_drop = prop_drop
        self.size_embedding = size_embedding
        self.freeze_transformer = False
        self.max_span_size = max_span_size
        self.rel_filter_threshold = rel_filter_threshold

        # Bert对应的tokenizer恢复
        self.tokenizer = BertTokenizer.from_pretrained(
            self.tokenizer_path, do_lower_case=self.lowercase, cache_dir=self.cache_path
        )
        self.input_reader = JsonPredictionInputReader(
            self.types_path, self.tokenizer,
            max_span_size=self.max_span_size,
            spacy_model=None
        )
        self.device = torch.device("cuda" if (torch.cuda.is_available() and not cpu) else "cpu")
        self.model = load_model(self, self.tokenizer, self.input_reader)  # 模型恢复
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text):
        # 1. 解析文本，获取得到document对象
        doc = self.input_reader.parse_tokens(text)

        # 2. 获取实体对象的优选集
        batch = collate_fn_padding([create_eval_sample(doc, max_span_size=self.max_span_size)])
        batch = util.to_device(batch, self.device)

        # 3. 模型预测 run model (forward pass)
        result = self.model(
            encodings=batch['encodings'],
            context_masks=batch['context_masks'],
            entity_masks=batch['entity_masks'],
            entity_sizes=batch['entity_sizes'],
            entity_spans=batch['entity_spans'],
            entity_sample_masks=batch['entity_sample_masks'],
            inference=True
        )
        entity_clf, rel_clf, rels = result

        # 4. 预测结果解析处理
        predictions = prediction.convert_predictions(
            entity_clf, rel_clf, rels,
            batch, self.rel_filter_threshold,
            self.input_reader
        )
        batch_pred_entities, batch_pred_relations = predictions

        # 5. 将处理结果转换为期望输出的结果格式
        result = prediction.parse_predictions([doc], batch_pred_entities, batch_pred_relations)
        return result
