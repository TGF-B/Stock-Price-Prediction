# Stock-Price-Prediction
若要分析的数据集是实时更新的，我们把它称作TSA时间序列分析（Time Series Analysis）,经常需要用到TSA的场景之一就是股市。这次我们来分析一下苹果的股价，并用LSTM长短期记忆神经网络算法([Long Short Term Memory](https://www.geeksforgeeks.org/long-short-term-memory-networks-explanation/))来预测它的走向。

## 导入数据并预处理
股市数据可以从[纳斯达克官网](https://www.nasdaq.com/market-activity/stocks/aapl/historical)下载。
我取了近一年的数据。
```python
import pandas as pd
import numpy as np
data=pd.read_csv("数据下载路径")
```
导入的数据是按照时间从近到远排列的，而股价又是携带单位的object类型，不能做数据分析。因此需要转换一下格式，然后按照时间逆序排列。由于数据量不大，我们索性用EXCEL完成这些简单的预处理步骤。    
处理完后展示一下：
```python
data.head()
```
>             Date   Close     Volume    Open    High     Low
>     0  05/10/2021  126.85   88071230  129.41  129.54  126.81
>     1  05/11/2021  125.91  126142800  123.50  126.27  122.77
>     2  05/12/2021  122.77  112172300  123.40  124.64  122.25
>     3  05/13/2021  124.97  105861300  124.58  126.15  124.26
>     4  05/14/2021  127.45   81917950  126.25  127.89  125.85

## 绘制近一年收盘价波动图
```python
import matplotlib.pyplot as plt
figure=plt.plot(data.Date,data.Close)
plt.show()
plt.figsave
```
![全年收盘价波动](https://github.com/TGF-B/Stock-Price-Prediction/blob/main/Figure_1.png)

但股市更常用**烛台图**（Candlestick），就是将开盘价，高点，低点，和收盘价全部展现在一张图中。
因此我们再会绘制一下烛台图。
```python
import plotly.graph_objects as go
figure=go.Figure(data=[go.Candlestick(x=go.Candlestick(x=data.index,
                                                        open=data["Open"],
                                                        high=data["High|],
                                                        low=data["low"],
                                                        close=data["Close"])
figure.update_layout(title="Time Series Analysis(Candlestick Chart)",
                      xaixs_rangeslider_visible=False)
figure.show()
```
![烛台图](https://github.com/TGF-B/Stock-Price-Prediction/blob/main/CandleStick.PNG)

查看一下相关性
```python
data=data.drop(“Date”，axis=1,inplace=True)
correlation=data.corr()
print(correlation["Close"],sort_values(ascending=True))
```
>     Close     1.000000
>     Low       0.994900
>     High      0.994831
>     Open      0.988654
>     Volume    0.276286
>     Name: Close, dtype: float64

## 训练LSTM模型
- 数据集分割
```python
x=data[["Open","High","Low","Volume"]]
y=data["Close"]
x=x.to_numpy() #以4维向量名成立n X 4的自变量矩阵
y=y.to_numpy() #以1维向量名成立n X 1的因变量矩阵
y=y.reshape(-1,1)#-1代表固定行数
from sklearn.model_selection import train_test_slpit
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
```
- 建立神经网络架构模型
```python
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()
```
>     Model: "sequential"
>     _________________________________________________________________
>      Layer (type)                Output Shape              Param #
>     =================================================================
>      lstm (LSTM)                 (None, 4, 128)            66560
>      lstm_1 (LSTM)               (None, 64)                49408
>      dense (Dense)               (None, 25)                1625
>      dense_1 (Dense)             (None, 1)                 26
>     =================================================================
>     Total params: 117,619
>     Trainable params: 117,619
>     Non-trainable params: 0
>     _________________________________________________________________

- 开始训练模型
```python
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=60)
```
>     Epoch 1/60
>     201/201 [==============================] - 1s 5ms/step - loss: 3.5582
>     Epoch 2/60
>     201/201 [==============================] - 1s 5ms/step - loss: 5.6847
>     Epoch 3/60
>     201/201 [==============================] - 1s 5ms/step - loss: 3.3240
>     ...
>     201/201 [==============================] - 1s 5ms/step - loss: 3.8948
>     Epoch 60/60
>     201/201 [==============================] - 1s 5ms/step - loss: 3.3578
>     <keras.callbacks.History object at 0x0000024CEFCFCE50>

我这里经过30,40,50,60,70,80次epoch的测试，只有60的损失是最低的，因此epoch times选了60。    
- 测试模型
训练模型所用的数据集是5月6日往前一年的，因此验证数据我们就用5月9日纳斯达克刚公布的。
```python
import numpy as np
#features=["Open”,"High","low","volume"]
features=np.array([[154.925,155.925,151.49,131577900]])
model.predict(features)
```
预测所得的收盘价为：

>    array([[153.51788]], dtype=float32)

而5月9日真实的收盘价为**152.06**美元！差了1.4美元。
 
 ## 结论
 从目前唯一的验证结果来看，我们的模型还是差强人意，本模型拿来做炒股的参考就还不错！
 明天我们将继续验证！







