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
https://github.com/skywalker0803r/c620.git
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

輸入以下指令打開ICG試算網頁
```
streamlit run web_icg.py
```
輸入以下指令打開c620_c660_c670(簡稱F)試算網頁
```
streamlit run web_f.py
```

##Demo

ICG|F
:-------------------------:|:-------------------------:
![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/gif/PPO_LunarLander-v2.gif) |  ![](https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/gif/PPO_BipedalWalker-v2.gif)
