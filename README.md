# c620

## Installation
首先確保你的電腦有安裝[Anaconda](https://www.anaconda.com/products/individual)
以及[git](https://git-scm.com/downloads)


1.強烈建議使用Anaconda建立虛擬環境以避免一些環境問題
在命令提示字元(cmd)輸入
```
conda create -n c620_env python=3.7
```
2.然後激活該虛擬環境
```
conda activate c620_env
```
3.使用git指令下載整個c620專案資料夾
```
git clone https://github.com/skywalker0803r/c620.git
```
4.切換目錄到c620資料夾底下
```
cd c620
```
5.安裝必要套件
```
pip install -r requirements.txt
```
過程中如果沒有報錯至此整個安裝完成.

## Usage

輸入以下指令打開試算網頁
```
streamlit run web.py
```

## 流程圖

![](https://github.com/skywalker0803r/c620/blob/main/img/overview.png)

## 流程圖封裝 調用只須一行
![](https://github.com/skywalker0803r/c620/blob/main/img/.png)



## Demo

web.py
:-------------------------:
![](https://github.com/skywalker0803r/c620/blob/main/gif/web.gif)

## 試算結果自動保存功能

在c620專案目錄底下有一個log資料夾,裡面存放試算完的結果
![](https://github.com/skywalker0803r/c620/blob/main/img/log_dir.png)

以c620_wt_log.xlsx為範例打開來裡面內容如下
![](https://github.com/skywalker0803r/c620/blob/main/img/c620wt_log.png)
