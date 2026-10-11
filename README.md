# MNIST 数字识别实验精选

这是从本地数字识别研究项目整理出的可公开代码摘选，包含三条相互独立的实验线：

- `mnist_residual_cnn_v9_1.py`：残差卷积网络训练实验。
- `mnist_manual_n_ann_sigmoid_observation.py`：使用 NumPy 手写前向传播、反向传播和训练过程，观察深层 Sigmoid 网络的训练动态。
- `mnist_activation_encoding_experiments.py`、`activation_encoding_core.py`、`activation_encoding_readout.py`：激活函数、冻结编码器与共同输出头实验。`test_activation_encoding.py` 提供合成数据检查。

## 环境

需要 Python 3，以及 `numpy`、`pandas`、`matplotlib`、`tensorflow`、`scikit-learn`。具体版本应按你的 Python 与操作系统选择兼容组合。

## 快速检查

在本目录运行：

```bash
python -B -m unittest -v test_activation_encoding
python -B mnist_activation_encoding_experiments.py --stage all --synthetic-check
```

第二条命令只检查流程，会在 `logs/` 下生成合成数据结果；它不代表 MNIST 实验性能。正式实验需要先准备 MNIST 数据。可用 `--mnist-cache` 指定 `mnist.npz` 的位置，更多选项请运行 `python mnist_activation_encoding_experiments.py --help`。

## 范围与结果解释

本摘选保留了有区别的实验实现，没有包含旧版脚本、备份代码、训练日志、模型权重、测试图片、原始数据集、论文 PDF 和生成的报告。训练会产生的目录已写入 `.gitignore`。

代码用于复现实验方法；这里没有打包历史模型或声称这些脚本已经在新环境中完成正式 MNIST 复现。验证集观察、测试集结果和因果解释应分别处理，不能用测试集反复挑选模型后报告无偏准确率。

