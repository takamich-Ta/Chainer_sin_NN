#practice.py

import math
import numpy as np
from chainer import Variable
import chainer.functions as F
from chainer import optimizers
from Mychain import Mychain

#sin�̋��t�f�[�^�̗p��
#y=sin(x)

x,y=[],[]
for i in np.linspace(-3,3,100):
    x.append([i])
    y.append([math.sin(i)])

x=Variable(np.array(x,dtype=np.float32))
y=Variable(np.array(y,dtype=np.float32))

#y=sin(x)�̔z�񂪂ł���

#Mychain�ŃC���X�^���X�𐶐�

model=Mychain()

#�����֐��̌v�Z

def forward(x,y,model):
    #�`�d�At�͐��_�l
    t=model.predict(x)
    #�����֐����덷
    loss=F.mean_squared_error(t,y)
    return loss

#�w�K�̗p�ӁA�A�_���𗘗p�BSGD�ł���������

optimizer=optimizers.Adam()
optimizer.setup(model)

#�w�K�A�e�i�K�ł̑����֐��̒l�����o��

for i in range(1000):
    loss=forward(x,y,model)
    if i%100==0:
        #�����֐��̒l
        print(loss.data)
    #�p�����[�^���X�V
    optimizer.update(forward,x,y,model)