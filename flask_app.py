from flask import Flask, request, jsonify

from args import predict_argparser
from config_reader import _yield_configs_from_file
from spert.spert_predictor import SpertPredictor

app = Flask(__name__)

# 参数解析
arg_parser = predict_argparser()
args, _ = arg_parser.parse_known_args()
run_args, _, _ = next(_yield_configs_from_file(arg_parser, args))

# 预测器的构建
predictor = SpertPredictor(
    model_type=run_args.model_type, model_path=run_args.model_path,
    cache_path=run_args.cache_path, tokenizer_path=run_args.tokenizer_path,
    types_path=run_args.types_path, max_pairs=run_args.max_pairs,
    prop_drop=run_args.prop_drop, size_embedding=run_args.size_embedding,
    max_span_size=run_args.max_span_size, rel_filter_threshold=run_args.rel_filter_threshold
)


@app.route("/")
def index():
    return "命名实体识别和关系抽取接口服务"


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        text = request.args['text']
    else:
        text = request.form['text']
    result = predictor.predict(text)
    return jsonify(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9999)
