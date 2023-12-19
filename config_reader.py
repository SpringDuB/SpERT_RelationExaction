import copy
import multiprocessing as mp


def process_configs(target, arg_parser, multi_process=False):
    args, _ = arg_parser.parse_known_args()  # 获取运行命令给定的参数

    if multi_process:
        ctx = mp.get_context('spawn')
        for run_args, _run_config, _run_repeat in _yield_configs(arg_parser, args):
            p = ctx.Process(target=target, args=(run_args,))  # 使用run_args参数以进程的方式来运行target方法
            p.start()
            p.join()
    else:
        for run_args, _run_config, _run_repeat in _yield_configs(arg_parser, args):
            target(run_args)


def _read_config(path):
    lines = open(path, "r", encoding="utf-8").readlines()

    runs = []
    run = [1, dict()]
    for line in lines:
        stripped_line = line.strip()

        # continue in case of comment
        if stripped_line.startswith('#'):
            continue

        if not stripped_line:
            if run[1]:
                runs.append(run)

            run = [1, dict()]
            continue

        if stripped_line.startswith('[') and stripped_line.endswith(']'):
            repeat = int(stripped_line[1:-1])
            run[0] = repeat
        else:
            key, value = stripped_line.split('=')
            key, value = (key.strip(), value.strip())
            run[1][key] = value

    if run[1]:
        runs.append(run)

    return runs


def _convert_config(config):
    config_list = []
    for k, v in config.items():
        if v.lower() == 'true':
            config_list.append('--' + k)
        elif v.lower() != 'false':
            config_list.extend(['--' + k] + v.split(' '))

    return config_list


def _yield_configs(arg_parser, args, verbose=True):
    _print = (lambda x: print(x)) if verbose else lambda x: x

    if args.config:
        config = _read_config(args.config)  # 读取config文件内容，直接以"="进行属性分割

        for run_repeat, run_config in config:
            print("-" * 50)
            print("Config:")
            print(run_config)

            args_copy = copy.deepcopy(args)  # 复制命令行参数
            config_list = _convert_config(run_config)  # 将dict类型的config参数转换为命令行参数的格式, eg: --k1 v1 --k2 v2
            run_args = arg_parser.parse_args(config_list, namespace=args_copy)  # 重新解析参数
            run_args_dict = vars(run_args)

            # set boolean values
            for k, v in run_config.items():
                if v.lower() == 'false':
                    run_args_dict[k] = False

            print("Repeat %s times" % run_repeat)
            print("-" * 50)

            for iteration in range(run_repeat):
                _print("Iteration %s" % iteration)
                _print("-" * 50)

                yield run_args, run_config, run_repeat

    else:
        yield args, None, None


def _yield_configs_from_file(arg_parser, args, _print=None):
    if _print is None:
        _print = lambda x: print(x)

    config = _read_config(args.config)  # 读取config文件内容，直接以"="进行属性分割

    for run_repeat, run_config in config:
        print("-" * 50)
        print("Config:")
        print(run_config)

        args_copy = copy.deepcopy(args)  # 复制命令行参数
        config_list = _convert_config(run_config)  # 将dict类型的config参数转换为命令行参数的格式, eg: --k1 v1 --k2 v2
        run_args = arg_parser.parse_args(config_list, namespace=args_copy)  # 重新解析参数
        run_args_dict = vars(run_args)

        # set boolean values
        for k, v in run_config.items():
            if v.lower() == 'false':
                run_args_dict[k] = False

        print("Repeat %s times" % run_repeat)
        print("-" * 50)

        for iteration in range(run_repeat):
            _print("Iteration %s" % iteration)
            _print("-" * 50)

            yield run_args, run_config, run_repeat
