#practice.py

import math
import numpy as np
from chainer import Variable
import chainer.functions as F
from chainer import optimizers
from Mychain import Mychain

#sinの教師データの用意
#y=sin(x)

x,y=[],[]
for i in np.linspace(-3,3,100):
    x.append([i])
    y.append([math.sin(i)])

x=Variable(np.array(x,dtype=np.float32))
y=Variable(np.array(y,dtype=np.float32))

#y=sin(x)の配列ができた

#Mychainでインスタンスを生成

model=Mychain()

#損失関数の計算

def forward(x,y,model):
    #伝播、tは推論値
    t=model.predict(x)
    #損失関数二乗誤差
    loss=F.mean_squared_error(t,y)
    return loss

#学習の用意、アダムを利用。SGDでもいいかも

optimizer=optimizers.Adam()
optimizer.setup(model)

#学習、各段階での損失関数の値だけ出力

for i in range(1000):
    loss=forward(x,y,model)
    if i%100==0:
        #損失関数の値
        print(loss.data)
    #パラメータを更新
    optimizer.update(forward,x,y,model)