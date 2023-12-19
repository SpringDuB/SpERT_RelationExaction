import argparse

from args import train_argparser, eval_argparser, predict_argparser
from config_reader import process_configs
from spert import input_reader
from spert.spert_predictor import SpertPredictor
from spert.spert_trainer import SpERTTrainer


def _train():
    arg_parser = train_argparser()  # 参数解析对象的构建
    process_configs(target=__train, arg_parser=arg_parser)


def __train(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.train(
        train_path=run_args.train_path,  # 训练数据所在文件
        valid_path=run_args.valid_path,  # 验证数据所在文件
        types_path=run_args.types_path,  # 实体识别、关系抽取的标签信息所在文件
        input_reader_cls=input_reader.JsonInputReader  # 数据加载器
    )


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


def __eval(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.eval(
        dataset_path=run_args.dataset_path,
        types_path=run_args.types_path,
        input_reader_cls=input_reader.JsonInputReader
    )


def _predict():
    arg_parser = predict_argparser()
    process_configs(target=__predict, arg_parser=arg_parser)


def __predict(run_args):
    trainer = SpERTTrainer(run_args)
    trainer.predict(
        dataset_path=run_args.dataset_path,
        types_path=run_args.types_path,
        input_reader_cls=input_reader.JsonPredictionInputReader
    )


def local_deploy_predict():
    arg_parser = predict_argparser()
    process_configs(target=__local_deploy_predict, arg_parser=arg_parser, multi_process=False)


def __local_deploy_predict(run_args):
    predictor = SpertPredictor(
        model_type=run_args.model_type, model_path=run_args.model_path,
        cache_path=run_args.cache_path, tokenizer_path=run_args.tokenizer_path,
        types_path=run_args.types_path, max_pairs=run_args.max_pairs,
        prop_drop=run_args.prop_drop, size_embedding=run_args.size_embedding,
        max_span_size=run_args.max_span_size, rel_filter_threshold=run_args.rel_filter_threshold
    )
    while True:
        text = input("请输入一个以空格隔开的token字符串:")
        if text == "q":
            break
        result = predictor.predict(text)
        print(result)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()

    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    elif args.mode == 'predict':
        _predict()
    elif args.mode == 'deploy_predict':
        local_deploy_predict()
    else:
        raise Exception("Mode not in ['train', 'eval', 'predict'], e.g. 'python spert.py train ...'")
