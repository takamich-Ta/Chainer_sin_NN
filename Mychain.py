#Mychain.py

from chainer import Chain
import chainer.links as L
import chainer.functions as F

# �p�����[�^�[�A�`�d�A�ڑ����N���X�Ƃ��ėp�ӁBChain���p�����Ă���B

class Mychain(Chain):

    #�p�����[�^�[�̃��j�b�g
    #(1,100)�̓��͑w�A(100,30)�̉B��w�A(30,1)�̏o�͑w
    #�B��w�̎����͕ύX�\�B���x���オ�邩���B

    def __init__(self):
        super(Mychain,self).__init__(
            l1=L.Linear(1,100),
            l2=L.Linear(100,30),
            l3=L.Linear(30,1)
        )

    #�`�d�A�ڑ��̎d����predict�Ƃ��Đݒ�
    def predict(self,x):
        #l1�œ���x���󂯂�relu�֐�
        h1=F.relu(self.l1(x))
        #l2�œ���h1���󂯂�relu�֐�
        h2=F.relu(self.l2(h1))
        #l3�̏o�͂�Ԃ��B
        return self.l3(h2)