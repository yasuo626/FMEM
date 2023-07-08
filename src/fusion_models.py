import torch
import torch.nn as nn

# base model for single embdeding input
class SingleEmbedsPerceptron(torch.nn.Module):
    def __init__(self,config,input_dim,output_dim,hidden_size1=512,hidden_size2=128 ):
        super(SingleEmbedsPerceptron, self).__init__()
        self.output_dim=output_dim
        self.dtype=config.dtype
        self.device=config.device

        self.linear1 = torch.nn.Linear(input_dim,hidden_size1)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size1,hidden_size2)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(hidden_size2, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return self.sigmoid(x)

class AttentionModel(torch.nn.Module):
    def __init__(self, config,num_embeds,max_embdes_length, output_dim, hidden_size1=512, hidden_size2=128):
        super(AttentionModel, self).__init__()
        self.output_dim = output_dim
        self.dtype = config.dtype
        self.device = config.device
        self.num_embeds=num_embeds
        self.max_embdes_length=max_embdes_length

        self.linear1 = torch.nn.Linear(max_embdes_length*num_embeds,hidden_size1)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(hidden_size2, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x.reshape(-1,self.num_embeds*self.max_embdes_length))
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return self.sigmoid(x)

# Transformer on multi embdeding inputs

class Transformer_AttentionModel(torch.nn.Module):
    def __init__(self,config,num_embeds, output_dim,nhead=1, num_encoder_layers=1, num_decoder_layers=1):
        super(Transformer_AttentionModel, self).__init__()
        self.transformer = nn.Transformer(d_model=num_embeds, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
        self.linear = nn.Linear(num_embeds, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:[batch,n,embeds] ->[batch,embeds,n] ->[embeds,batch,n]
        x = x.permute(2,0,1)
        out = self.transformer(x,x)
        out = out.permute(1, 0, 2)  # Reshape output tensor to (batch_size, seq_len, output_dim)
        out = self.linear(out[:,-1, :])  # Use only the last output token for classification
        return self.sigmoid(out)

# Fusion_Attention on multi embdeding inputs


class Fusion_AttentionModel(torch.nn.Module):
    def __init__(self,config,num_embeds,max_embdes_length,output_dim,core_range=None,stride=None,hidden_size1=512,hidden_size2=128 ):
        super(Fusion_AttentionModel, self).__init__()
        self.output_dim = output_dim
        self.dtype = config.dtype
        self.device = config.device

        self.num_embeds=num_embeds
        self.max_embdes_length=max_embdes_length

        if core_range is None:
            core_range=max(max_embdes_length//100,2)
        if stride is None:
            stride=max(core_range//2,1)
        assert core_range<=max_embdes_length

        self.pad_len=stride-(max_embdes_length-core_range)%stride if (max_embdes_length-core_range)%stride else 0
        self.seq_len=max_embdes_length+self.pad_len
        self.core_range=core_range
        self.stride=stride

        self.conv1d1=torch.nn.Conv1d(in_channels=num_embeds,out_channels=5,kernel_size=3,bias=True,padding='same')
        self.bn=torch.nn.BatchNorm1d(5)
        self.conv1d2=torch.nn.Conv1d(in_channels=5,out_channels=1,kernel_size=2,bias=True,padding='same')



        self.linear1 = torch.nn.Linear(max_embdes_length, hidden_size1)
        self.activation1 = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.1)
        self.linear2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.activation2 = torch.nn.ReLU()
        self.drop2 = torch.nn.Dropout(0.2)
        self.linear3 = torch.nn.Linear(hidden_size2, output_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        # out=torch.nn.functional.pad(inputs,(0,self.pad_len),mode="constant",value=0)
        # arg_s=torch.arange(0,self.seq_len-self.core_range,self.stride).unsqueeze(0).expand(self.num_embeds, -1)
        # arg_e=arg_s+self.core_range
        # print(out[:,arg_s:arg_e].shape)
        # print(out[:,arg_s+self.core_range].shape)
        out=self.conv1d1(inputs)
        out=self.bn(out)
        out=self.conv1d2(out).reshape(-1,self.max_embdes_length)
        out=self.activation1(self.linear1(out))
        out=self.activation2(self.linear2(out))
        print(out.shape)
        out=self.sigmoid(self.linear3(out))
        return out
        # return self.sigmoid(torch.zeros(self.output_dim))








# class PositionalEncoding(nn.Module):
#     def __init__(self,config,max_len=1024, d_model=64,dropout=0.1):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len,dtype=config.dtype).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
#
#
# class Transformer_AttentionModel(torch.nn.Module):
#     def __init__(self,config,num_embeds,max_embdes_length,output_dim,d_model=64,nhead=6):
#         super(Transformer_AttentionModel, self).__init__()
#         self.max_len = num_embeds*max_embdes_length
#         self.pos_encoder = PositionalEncoding(config,max_len=num_embeds*max_embdes_length,d_model= d_model, dropout=0.1)
#         self.transformer_encoder = torch.nn.TransformerEncoder(d_model=d_model, nhead=nhead)
#         self.encoder = torch.nn.Linear(num_embeds*max_embdes_length,d_model)
#         self.decoder = torch.nn.Linear(d_model, output_dim)
#         self.sigmoid = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         x=self.encoder(x.reshape(-1,self.max_len))
#         x=x.permute(1,0,2)
#         x=self.pos_encoder()
#
#         output = self.transformer_encoder(x)
#         return self.sigmoid(x)
#
#























