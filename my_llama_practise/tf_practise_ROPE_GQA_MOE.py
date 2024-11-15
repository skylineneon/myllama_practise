import torch
from torch import nn
# 定义单位角度
def precompute_freqs_cis(dim, end, theta=50000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

# 提供旋转
def apply_rotary_emb(xq, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq)


def reshape_for_broadcast(freqs_cis, x):
    freqs_cises = freqs_cis[:x.shape[1]]
    return freqs_cises[None, :, None]

class TransformerDecoder(nn.Module):
    """
        解码器
    """

    def __init__(self,
                 num_layers, #解码器的层数
                 input_dim,
                 hide_dim,
                 n_heads,
                 max_len #定义最大长度
                 ):
        super().__init__()

        self._layers = nn.ModuleList(
            [TransformerLayer(input_dim,hide_dim,n_heads) for _ in range(num_layers)]
        )
        # 定义单位角度
        _freq_cis=precompute_freqs_cis(input_dim//n_heads,4096*2,50000)
        self.register_buffer("freq_cis",_freq_cis,persistent=False)

    def forward(self,x):
        _x = x
        for _layer in self._layers:
            _x = _layer(_x,self.freq_cis)
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
        

    def forward(self,x,freq_cis):
        _x = x
        _x = self._att_norm(_x)
        _x = self._att_layer(_x,freq_cis)

        _x = x + _x

        _y = _x
        _y = self._ffn_norm(_y)
        _y = self._ffn_layer(_y)

        _y = _y + _x

        return _y

class Attention(nn.Module):

    def __init__(self,
                 input_dim,
                 n_q_heads,
                 n_kv_heads
                #  _seq
                 ):
        super().__init__()
        '''
        q与kv成倍数关系
        '''
        self._n_q_heads = n_q_heads
        self._n_kv_heads = n_kv_heads 

        self._head_size=input_dim//self._n_q_heads
        self._group =n_q_heads//n_kv_heads

        self._qw = nn.Linear(input_dim,self._head_size*self._n_q_heads)
        self._kw = nn.Linear(input_dim,self._head_size*self._n_kv_heads)
        self._vw = nn.Linear(input_dim,self._head_size*self._n_kv_heads)
        self._ow = nn.Linear(input_dim,input_dim)

        # _causul = torch.ones(_seq,_seq)
        # _causul = torch.triu(_causul,diagonal=1)
        # _causul[_causul==1]=-torch.inf

        # self.register_buffer("causul",_causul,persistent=False)


    def forward(self,x,freq_cis):
        _bn,_seq,_v_size = x.shape
        
        _dk = self._head_size**0.5

        _q,_k,_v = self._qw(x),self._kw(x),self._vw(x)

        _q = _q.reshape(_bn,_seq,self._n_q_heads,self._head_size)
        _k = _k.reshape(_bn,_seq,self._n_kv_heads,self._head_size)
        _v = _v.reshape(_bn,_seq,self._n_kv_heads,self._head_size)

        # 给q和k加上旋转位置编码
        _q=apply_rotary_emb(_q,freq_cis[:_seq])
        _k=apply_rotary_emb(_k,freq_cis[:_seq])

        _q = _q.permute(0,2,1,3)
        _k = _k.permute(0,2,1,3)
        _v = _v.permute(0,2,1,3)

        _causul = torch.ones(_seq,_seq)
        _causul = torch.triu(_causul,diagonal=1)
        _causul[_causul==1]=-torch.inf
        _causul = _causul.to(x.device)
        '''
        此时_q结构为NH(q)SV`  _k的结构为NH(k)SV` 
        两者头数H不同无法相乘
        所以要把两者形状转成一致
        增加一个维度,把H(k)复制几份，然后再合并
        '''
        _k=_k[:,None].repeat(1,self._group,1,1,1).reshape(_bn,-1,_seq,self._head_size)
        _v=_v[:,None].repeat(1,self._group,1,1,1).reshape(_bn,-1,_seq,self._head_size)

        _score = _q@_k.permute(0,1,3,2)/_dk
        _score =torch.softmax(_score + _causul,dim=-1)

        _o = _score@_v

        '''
        将Q分组,上方是llama写法
        '''
        # _q = _q.reshape(_bn,self._group,self._n_kv_heads,_seq,self._head_size)
        # _k = _k[:,None]
        # _v = _v[:,None]
        
        # _score = _q@_k.permute(0,1,2,4,3)/_dk
        # _score =torch.softmax(_score + _causul,dim=-1)

        # _o = _score@_v
        # _o = _o.reshape(_bn,-1,_seq,self._head_size)

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
       
class Expert(nn.Module):
    def __init__(self,
                 num_experts,
                 top_k,# 选择几个专家
                 input_dim,
                 hide_dim
                 ):
        super().__init__()
        self._top_k=top_k
        self._experts =nn.ModuleList([FFN(input_dim,hide_dim) for _ in range(num_experts)]) 

        #建一个专家选择的门
        self._gate=nn.Linear(input_dim,num_experts)
    def forward(self,x):

        _bn,_seq,_vec=x.shape

        _gate_out=self._gate(x)
        _top_values, _top_indices = torch.topk(_gate_out, self._top_k, dim=-1)
        
        _output = torch.zeros(_bn,_seq,self._top_k,_vec).to(x.device)
        for _i in range(_bn):
            for _j in range(_seq):
                for _k in range(self._top_k):
                    _expert = self._experts[_top_indices[_i,_j,_k]]
                    _output[_i,_j,_k] = _expert(x[_i,_j])
        w = torch.softmax(_top_values,dim=-1)[:,:,:,None]
        return (w*_output).sum(dim=2)
    















class RMSNormal(nn.Module):
    
    def __init__(self,input_dim):
        super().__init__()

        self._w = nn.Parameter(torch.randn(input_dim))

    def forward(self,x):
       return self._w*x/((x**2).sum()**0.5+1e-6)



