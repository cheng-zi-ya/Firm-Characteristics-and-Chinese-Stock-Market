# Firm-Characteristics-and-Chinese-Stock-Market

It is a project that want to accomplish several interesting algorithms to combine features and make predictions. The main reference in the beginning is the paper 'Firm Characteristics and Chinese Stock Market'.

I am still working to add more tools in this project.

  
## Introduction

This project tried several algorithms to combine 75 stock market factors based on companies and financial features. Also, it provided tools to check the performance of stock factors.

Among all files, the pdf file is the corresponding paper, and the data.csv provides a data sample. Moreover, factor_test_monthly.py includes backtesting on single factors. fm.py, pca.py, pls.py and fc.py provides 4 algorithms.

We could find that pls and fc have better performance on Chinese stock market. It could achieved about 2.60% monthly return during 2003-06 ~ 2016-12.

  
## Results 
  
Two results pictures:

![image](https://raw.githubusercontent.com/cheng-zi-ya/Firm-Characteristics-and-Chinese-Stock-Market/main/photos/factors.png)  
  
![image](https://raw.githubusercontent.com/cheng-zi-ya/Firm-Characteristics-and-Chinese-Stock-Market/main/photos/pls.png)  


## Usage
  
factor_test_monthly.py： backtesting and evaluation on single factors.
fm.py, pca.py, pls.py和fc.py： algorithms to combine factors.

