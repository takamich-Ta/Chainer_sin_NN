#Mychain.py

from chainer import Chain
import chainer.links as L
import chainer.functions as F

# パラメーター、伝播、接続をクラスとして用意。Chainを継承している。

class Mychain(Chain):

    #パラメーターのユニット
    #(1,100)の入力層、(100,30)の隠れ層、(30,1)の出力層
    #隠れ層の次元は変更可能。精度が上がるかも。

    def __init__(self):
        super(Mychain,self).__init__(
            l1=L.Linear(1,100),
            l2=L.Linear(100,30),
            l3=L.Linear(30,1)
        )

    #伝播、接続の仕方をpredictとして設定
    def predict(self,x):
        #l1で入力xを受けてrelu関数
        h1=F.relu(self.l1(x))
        #l2で入力h1を受けてrelu関数
        h2=F.relu(self.l2(h1))
        #l3の出力を返す。
        return self.l3(h2)