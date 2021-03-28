#advForNLP

中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer, 基于pytorch，开箱即用。

## 环境
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX

## 中文数据集
我从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万


### 更换自己的数据集
 - 如果用字，按照我数据集的格式来格式化你的数据。  
 - 如果用词，提前分好词，词之间用空格隔开，`python run.py --model TextCNN --word True`  
 - 使用预训练词向量：utils.py的main函数可以提取词表对应的预训练词向量。  


## 效果

- 准确率	precision	Recall	F1
- Baseline	0.8658	0.8648	0.8644
- Pgd	0.8819	 0.8818	  0.8815
- Free	0.8772	0.8765	0.8762
- Fgsm	0.8866	 0.8857	0.8855
 

## 使用说明
```
baseline算法调测命令：
python run.py --model TextCNN  --sgdflag base
pgd算法调测命令：
python run.py --model TextCNN  --sgdflag pgd
free算法调测命令：
python run.py --model TextCNN  --sgdflag free
fgsm算法调测命令：
python run.py --model TextCNN  --sgdflag fgsm

```

### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。  
- commons：通用代码
- cores：核心代码
- dataset：数据集和模型存放路径
- models：模型代码
- trains:训练代码
- runAdv.py 训练入口