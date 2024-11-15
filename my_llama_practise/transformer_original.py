import torch
from torch import nn


class TransformerDecoder(nn.Module):
    """
        解码器
    """

    def __init__(self,
                 num_layers, #解码器的层数
                 input_dim,
                 hide_dim,
                 n_heads
                 ):
        super().__init__()

        self._layers = nn.ModuleList(
            [TransformerLayer(input_dim,hide_dim,n_heads) for _ in range(num_layers)]
        )

    def forward(self,x):
        _x = x
        for _layer in self._layers:
            _x = _layer(_x)
        return _x



class TransformerLayer(nn.Module):
    """
    单层的Transformer结构
    """
    def __init__(self,input_dim,hide_dim,n_heads):
        super().__init__()

        self._att_norm = RMSNormal(input_dim)
        self._att_layer = Attention(input_dim,n_heads)
        self._ffn_norm = RMSNormal(input_dim)
        self._ffn_layer = FFN(input_dim,hide_dim)

    def forward(self,x):
        _x = x
        _x = self._att_norm(_x)
        _x = self._att_layer(_x)

        _x = x + _x

        _y = _x
        _y = self._ffn_norm(_y)
        _y = self._ffn_layer(_y)

        _y = _y + _x

        return _y

class Attention(nn.Module):

    def __init__(self,
                 input_dim,
                 n_heads
                #  _seq
                 ):
        super().__init__()

        self._n_heads = n_heads
        
        self._qw = nn.Linear(input_dim,input_dim)
        self._kw = nn.Linear(input_dim,input_dim)
        self._vw = nn.Linear(input_dim,input_dim)
        self._ow = nn.Linear(input_dim,input_dim)

        # _causul = torch.ones(_seq,_seq)
        # _causul = torch.triu(_causul,diagonal=1)
        # _causul[_causul==1]=-torch.inf

        # self.register_buffer("causul",_causul,persistent=False)


    def forward(self,x):
        _bn,_seq,_v_size = x.shape
        _nh = self._n_heads
        _h_size = _v_size // _nh

        _dk = _h_size**0.5

        _q,_k,_v = self._qw(x),self._kw(x),self._vw(x)

        _q = _q.reshape(_bn,_seq,_nh,_h_size)
        _k = _k.reshape(_bn,_seq,_nh,_h_size)
        _v = _v.reshape(_bn,_seq,_nh,_h_size)

        _q = _q.permute(0,2,1,3)
        _k = _k.permute(0,2,1,3)
        _v = _v.permute(0,2,1,3)

        _causul = torch.ones(_seq,_seq)
        _causul = torch.triu(_causul,diagonal=1)
        _causul[_causul==1]=-torch.inf
        _causul = _causul.to(x.device)

        _score = _q@_k.permute(0,1,3,2)/_dk
        _score =torch.softmax(_score + _causul,dim=-1)

        _o = _score@_v

        _o = _o.permute(0,2,1,3)
        _o = _o.reshape(_bn,_seq,-1)

        return self._ow(_o)
        

class FFN(nn.Module):

    def __init__(self,input_dim,hide_dim):
       super().__init__()

       self._w0 = nn.Linear(input_dim,hide_dim)
       self._w1 = nn.Linear(input_dim,hide_dim)
       self._w2 = nn.Linear(hide_dim,input_dim)

       self._gate = nn.SiLU()

    def forward(self,x):
        return self._w2(self._w0(x)*self._gate(self._w1(x)))
       

class RMSNormal(nn.Module):
    
    def __init__(self,input_dim):
        super().__init__()

        self._w = nn.Parameter(torch.randn(input_dim))

    def forward(self,x):
       return self._w*x/((x**2).sum()**0.5+1e-6)


