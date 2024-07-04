# 5/19
1. FACED 老数据 跑一遍  甘 ok
2. extract_fea.py  检查路径问题 沈 ok
3. wd 调整 甘
4. SEED 完整跑通 去att 沈 ok
5. FACED segatt mslen 甘 segatt 1 5 10 20 25 30  1-6 ok
6. FACED model ablation 王  结果有问题
7. PR-PL对比试验  张  结果有问题

# 5/25
1. FACED  （什么任务）segatt  1-5 甘  ok
2. wd 调整 甘 9分类only  0.015 0.0015 0.00015  mlp  取最好  ok
3. Variations of the convolution design  王  2/9都要先写code
4. 使用之前的seed数据，相同结构 沈
5. model-ablation 无att me de 找bug 王
6. PR-PL normTrain de running-norm lds FACED SEED 张
7. 预处理噪声 FACED 张

# 6/1
1. 30 epochs SEED 原数据 SEED 沈
2. de  mlp  baseline  甘  no
3. best model faced 2/9  甘  half
4. FACED去噪更少  best model  甘  running
5. variation of conv  王
6. FACED去噪  张

# 6/8
1. 检查seed 新老代码 看一下新科写的新代码  甘/新科
2. variation of conv  王  
3. transformer layer  甘
4. k-means 杨
5. 整理论文结构 新科

# 6/23
1. transformer layer  2/9 甘 ok
2. transformer 1 layer 2/9 time 甘 ok
3. DE + MLP 2/9   lessICA FACED 2/9 甘 ok
4. clisa  SEEDV   王
5. ablation  无空域 active  sigmoid  2/9分类(补充之前只跑9分类的部分) 王 ok
6. DE + MLP  SEEDV  甘  ok
7. 预训练wd 加两个 杨  0.0005，0.0001，0.0003 0.00005 is ok 
8. pretrain wd 0.0003 00.0002 杨
9. 模型架构 时空模式 推理模式 std model 新科
10. 混淆矩阵 FACED9 SEEDV SEED no_att DE+MLP std-model 甘/王

|       | FACED9 | SEEDV | SEED |
|:-------|:--------:|-------:|-------:|
| no_att | 王 ok| 王 | 王 |
| DE+MLP | 甘 ok| 甘 ok| 原有（甘） |
| std-model | 甘 ok| 王 ok| 甘 |

9. 训好整体model做分析 FACED 2/9  SEEDV ok
10. 用官方FACED 数据 做一下 2/9 std is ok  官方结果稍高
11. 官方FACED 数据 调2分类best seg_att=7 杨
12. SEED best model 甘 
13. SEED DE model 甘 
14. seed no_att model 王

# 7/21 截止
## -7/7
1. 填表格 甘 恺璇 杨
2. SEED  老代码调整超参 杨
3. 可视化 新科
## 7/7-7/10
1. translayer

## 7/10-7/21
1. 画图写文章
2. 混淆矩阵 SEED
