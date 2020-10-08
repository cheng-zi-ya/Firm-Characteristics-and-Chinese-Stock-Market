# Firm-Characteristics-and-Chinese-Stock-Market
对一篇论文的复现，感谢该论文的作者！其对国内股票市场的投资实操具有比较大的意义，大家可以参考。如果有帮助，望star哦😀  
  
这里上传的数据不是完整的数据，大家在自己做的时候可以自己整理新的一份A股数据，然后将项目中的流程应用就可以了。  
  
之后会对本项目进行补充，使得更完善一些。 
  
  
## 介绍  

本项目是对论文《Firm Characteristics and Chinese Stock Market》的实现，其主要包含了75个关于公司和股票特征的经典指标的计算，以及在中国市场上的检验，并使用了一些整合因子的方法，指导实际投资。  
  
在本项目的文件中，pdf文件即为对应的论文，而data.csv文件是给出了一个数据样例（是从全部的数据中抽取出了一些公司），而factor_test_monthly.py文件是对于单个因子的回报检验并使用五因子模型对其进行检验，fm.py, pca.py, pls.py和fc.py为对应的四个因子整合的方法。

其中，pls和fc的效果比较好，放到实际中，可以对在A股市场取得不错的收益。在论文中，pls对应的多空组合是 2.60%的月度收益，这样的年化大约是31.2%，可以说是比较不错的效果了。（论文中的测试期: 2003-06至2016-12）.
  
  
## 效果展示  
  
这里放上两张论文中的图片，一张是单因子的效果图，一张是pls组合的效果：  
![image](http://github.com/cheng-zi-ya/Firm-Characteristics-and-Chinese-Stock-Market/raw/master/photos/factors.png)  
  
![image](http://github.com/cheng-zi-ya/Firm-Characteristics-and-Chinese-Stock-Market/raw/master/photos/pls.png)  
  
## 使用教程  
  
factor_test_monthly.py： 对单因子进行测试和评价；
fm.py, pca.py, pls.py和fc.py： 对于因子进行聚合，形成对应的值，测试其的效果。
