# 第一次作业

## 内容

1. 内部威胁检测：依据题图计算 Accuracy、Precision、Recall、F1、FPR、TPR。
2. 数据投毒实验：实现任意标签投毒与目标3→8标签投毒，分析设计思路与影响成功率的因素。

## 阅读入口

- [作业原题](第一次作业.docx)
- [Word实验报告](数据安全治理第一次作业_投毒攻击实验报告.docx)
- [Markdown实验报告](数据安全治理第一次作业_投毒攻击实验报告.md)
- [实验代码与详细复现说明](poisoning_lab/README.md)
- [汇总结果](poisoning_lab/results/summary.csv)
- [运行环境及原始数据哈希](poisoning_lab/results/metadata.json)
- [结果完整性检查](poisoning_lab/results/audit.json)

报告正文保留实验当时的原始标题和运行日期；本仓库按课程提交要求归档在“第一次作业”目录。

## 快速复现

使用 Python 3.12，在当前作业目录执行：

```powershell
cd poisoning_lab
python -m pip install -r requirements.txt
python experiment.py
python analyze.py
```

已有结果来自2026年10月26日的65次实际训练。每组使用5个配对随机种子，结果以均值和样本标准差报告。若要重新训练全部模型，先将 `results` 改名备份。

`model_security_project/data/MNIST/raw/` 附四个未压缩MNIST IDX数据文件，保持目录结构即可使用默认路径。正式测试集包含10000个样本；投毒仅修改训练标签。

## 主要结果

- 干净MLP平均准确率：95.63%。
- 任意投毒：随机改标全部训练样本的60%，平均准确率降至78.66%。
- 目标投毒：将数字3类内70%的标签改为8，实际修改715个样本、占训练集7.15%，平均目标误判率79.58%。

任意投毒率以全训练集为分母，目标投毒率以源类别训练样本为分母，不能直接用相同百分比比较攻击预算。
