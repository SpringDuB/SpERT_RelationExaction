模型源码地址：

[lavis-nlp/spert: PyTorch code for SpERT: Span-based Entity and Relation Transformer (github.com)](https://github.com/lavis-nlp/spert)

模型结构：

![alt text](http://deepca.cs.hs-rm.de/img/deepca/spert.png)

## Setup
### Requirements
- Required
  - Python 3.5+
  - PyTorch (tested with version 1.4.0)
  - transformers (+sentencepiece, e.g. with 'pip install transformers[sentencepiece]', tested with version 4.1.1)
  - scikit-learn (tested with version 0.24.0)
  - tqdm (tested with version 4.55.1)
  - numpy (tested with version 1.17.4)
- Optional
  - jinja2 (tested with version 2.10.3) - if installed, used to export relation extraction examples
  - tensorboardX (tested with version 1.6) - if installed, used to save training process to tensorboard
  - spacy (tested with version 3.0.1) - if installed, used to tokenize sentences for prediction

## Examples
更改config配置文件中数据集的路径或通过脚本参数的形式更改

(1) :如果已经更改数据集路径，可以直接执行以下代码：

```
python spert.py train --config configs/example_train.conf
```

(2) 使用默认的测试数据集进行测试:
```
python ./spert.py eval --config configs/example_eval.conf
```

(3) 预测请提前安装spacy库，需要下载一个模型. 
```
python ./spert.py predict --config configs/example_predict.conf
```

其他数据集文件：

链接：https://pan.baidu.com/s/1MkYYBi_GX_zKPU-gt3t7pQ?pwd=1472 
提取码：1472

端口调用如下：返回实体和关系以及token字符串

```
[{"entities":[{"end":5,"start":1,"type":"图书作品"},{"end":11,"start":7,"type":"人物"}],"relations":[{"head":0,"tail":1,"type":"作者"}],"tokens":["《","邪","少","兵","王","》","是","冰","火","未","央","写","的","网","络","小","说","连","载","于","旗","峰","天","下"]}]
```
