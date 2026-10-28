---
title: MNIST数据投毒实验复现说明
date: 2026-10-26
tags: [数据安全治理, 数据投毒, MNIST]
status: polished
share: false
---

# MNIST数据投毒实验复现说明

## 实验内容

任意投毒从全部训练样本中选取一定比例，将标签随机修改为其他类别，检验整体准确率下降。目标投毒仅把数字3的部分训练标签改为8，检验真实数字3被识别为8的比例。实验不修改图像特征，测试集真值保持不变。

正式实验共65次真实训练，13组配置，每组使用17、29、43、71、101五个配对种子。固定10000个训练样本、2000个独立验证样本和10000个官方测试样本。主模型为196→96→10的MLP，固定训练60轮，另有0→1类别对照和逻辑回归模型对照。

## 运行

使用Python 3.12，在本目录执行：

```powershell
python -m pip install -r requirements.txt
python experiment.py
python analyze.py
```

默认读取 `../model_security_project/data/MNIST/raw/` 中四个未压缩IDX文件。完整实验包已附这些文件，解压后保持目录结构即可；不需要联网下载数据，也不依赖PyTorch。

若数据位于其他目录，可执行：

```powershell
python experiment.py --data-dir '实际MNIST原始文件目录' --out results
python analyze.py --data-dir '实际MNIST原始文件目录' --out results
```

已有results为2026-10-26本机实际运行结果。程序核对源码、数据与关键环境版本指纹后才复用逐次结果；如需完整重新训练，可先把results改名备份。

短流程检查只验证执行过程，不用于正式报告：

```powershell
python experiment.py --smoke --out ../workbench/smoke_results
```

## 统计口径

任意投毒率以全部10000个训练样本为分母。目标投毒率以源类别训练样本为分母，数字3有1022个训练样本；70%设定实际修改715个标签，对应源类69.96%、全训练集7.15%。7%任意投毒修改700个样本，与715个不是严格等预算。

3→8目标攻击成功率以1010个真实数字3测试样本为分母；0→1对照使用980个真实数字0测试样本。非源类准确率使用其余测试类别，3→8实验的分母为8990。

每组报告5次均值和样本标准差，误差线不是置信区间。种子共同改变投毒采样、MLP初始化与训练打乱；不能将波动全部归因于某一个因素。测试集不参与轮数、预算或种子的选择。

## 输出文件

- `experiment.py`：IDX检查、固定抽样、标签污染、模型训练与测试预测。
- `analyze.py`：原始哈希复查、改标计数核验、统计汇总和11幅图。
- `results/runs.csv`：65次原始指标；`paired_runs.csv`增加同模型同种子基线差值。
- `results/summary.csv`、`summary.json`：每组均值与样本标准差，比例以0至1保存。
- `results/runs/*.json`：每次参数、指标和训练/验证历史。
- `results/runs/*.npz`：完整测试预测、混淆矩阵和投毒索引及标签。
- `results/metadata.json`、`execution.log`：实际版本、数据哈希、时间和执行记录。
- `results/audit.json`：65次数据与指标完整性检查。
- `results/figures/`：检测指标题图、数据示例和实验曲线等11幅图。

报告开头的检测题根据原题图计数：TP=4、FP=2、FN=1、TN=3。Accuracy=70.00%，Precision=66.67%，Recall=80.00%，F1=72.73%，FPR=40.00%，TPR=80.00%。这道10样本的二分类题与MNIST十分类实验分开计算。
