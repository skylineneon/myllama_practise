import torch
from torch import nn
def precompute_freqs_cis(dim, end, theta=50000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)


def reshape_for_broadcast(freqs_cis, x):
    freqs_cises = freqs_cis[:x.shape[1]]
    return freqs_cises[None, :, None] 
class TransformerDecoder(nn.Module):
    '''
    构建llama解码器
    '''
    def __init__(self,
                num_layers, # 解码器层数
                input_dim,
                n_heads,
                _seq,
                hide_dim
                 ):
        super().__init__()
        self._layers=nn.ModuleList(
            [TransformerLayer(input_dim,n_heads, _seq,hide_dim) for _ in range(num_layers)] # 循环层数构建多层Transformer

        )# 使用ModuleList构建网络
        
    def forward(self,x):
        _x=x
        for _layer in self._layers:
            _x=_layer(_x)
        return _x

class TransformerLayer(nn.Module):
    '''
    构建单层编码器
    包括RMSNorm、注意力层、FFN层
    '''
    def  __init__(self,
                  input_dim,
                  n_heads,
                  _seq,
                  hide_dim
                  ):
        super().__init__()
        '''
        根据llama结构初始化网络层
        '''
        self._att_norm=RMSNorm(input_dim)
        self._att_layer=Attention(input_dim,n_heads,_seq)
        self._ffn_norm=RMSNorm(input_dim)
        self._ffn_layer=FFN(input_dim,hide_dim)
    def forward(self,x):
        _x=x # 保存x便于做残差
        _x=self._att_norm(_x)
        _x=self._att_layer(_x)

        _x=_x+x

        _y=_x # 保存FFN的输入，便于做残差
        _y=self._ffn_norm(_y)
        _y=self._ffn_layer(_y)

        _y=_y+_x
        return _y
        
class RMSNorm(nn.Module):
    '''
    传入的形式是NSV  对V做Norm
    '''
    def __init__(self,input_dim):
        super().__init__()
        self._w=nn.Parameter(torch.randn(input_dim))
    def forward(self,x):
        return self._w*x/((x**2).mean(-1,keepdim=True)**0.5+1e-6)   



class Attention(nn.Module):
    def __init__(self,
                 input_dim,
                 n_heads,#分的头的数量
                _seq
                 ):
        super().__init__()
        self._n_heads=n_heads

        self._qw=nn.Linear(input_dim,input_dim) 
        self._kw=nn.Linear(input_dim,input_dim)
        self._vw=nn.Linear(input_dim,input_dim)

        _causul=torch.ones(_seq,_seq)
        _causul=torch.triu(_causul,diagonal=1)
        _causul[_causul==1]=-torch.inf
        '''
        想让_causul传到coda上,但不希望在反向传播时他被修改
        '''
        self.register_buffer("causul",_causul,persistent=False) # 如果不希望被保存，加persistent

        self._ow=nn.Linear(input_dim,input_dim)
    def forward(self,
                x,

                ):
        _bn,_seq,_v_size=x.shape # 记录x的形状，_bn：批次 _seq:序列的长度
        _nh=self._n_heads #记录头数
        _h_size = _v_size // _nh

        _dk=_h_size**0.5 #注意力公式里的dk

        _q,_k,_v=self._qw(x),self._kw(x),self._vw(x)# 结构为NSV

        _q = _q.reshape(_bn,_seq,_nh,_h_size) # 将NSV改成NSHV`，即将一个词向量切成好几份
        _k = _k.reshape(_bn,_seq,_nh,_h_size)
        _v = _v.reshape(_bn,_seq,_nh,_h_size)
        '''
        NSHV`中是S和V`做注意力，则需要交换位置
        转换后为NHSV`
        '''
        _q = _q.permute(0,2,1,3)
        _k = _k.permute(0,2,1,3)
        _v = _v.permute(0,2,1,3)

        #加上因果矩阵
        # _causul=torch.ones(_seq,_seq)
        # _causul=torch.triu(_causul,diagonal=1)
        # _causul[_causul==1]=-torch.inf
        # _causul=_causul.to(x.device)


        #做矩阵相乘，注意力，计算注意力得分
        _score=_q@_k.permute(0,1,3,2)/_dk
        _score=torch.softmax(_score+self.causul,dim=-1)
        '''
        _score为NHSS  _v为NHSV`  
        相乘得到NHSV`,即_o
        输出是NSV   所以要转换结构
        （V是分割前的  V`是分割后的）
        '''
        _o=_score@_v
        _o=_o.permute(0,2,1,3)
        _o=_o.reshape(_bn,_seq,-1)
        return self._ow(_o)

class FFN(nn.Module):
    def __init__(self,
                 input_dim, # 输入维度
                 hide_dim, # 隐藏层维度

                 ):
        super().__init__()
        self._w0=nn.Linear(input_dim,hide_dim)
        self._w1=nn.Linear(input_dim,hide_dim)
        self._w2=nn.Linear(hide_dim,input_dim)

        self._gate=nn.SiLU()
    def forward(self,x):
        '''
        
        '''
        return self._w2(self._w0(x)*self._gate(self._w1(x)))