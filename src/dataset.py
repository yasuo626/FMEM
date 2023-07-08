import torch
import tqdm
from torch.utils.data import DataLoader,Dataset,DataChunk
from utils.utils import set_device,Config
import numpy as np



class SingleDataset(Dataset):
    def __init__(self,config:Config,ids,embeds,max_embeds_length,labels=None,train=True):
        self.device=config.device
        self.dtype=config.dtype
        self.max_embeds_length=max_embeds_length

        self.ids=ids
        self.embeds=embeds
        self.embeds_len=len(self.embeds[0])
        assert labels is not None if train else 1
        self.train=train
        self.labels=labels
    def __getitem__(self, index):
        data={}
        data['id']=self.get_id(index)
        data['embeds']=self.pad_or_cut(torch.tensor(self.embeds[index,:],dtype=self.dtype),self.max_embeds_length)
        if self.train:
            data['labels']=torch.tensor(self.labels[index,:],dtype=self.dtype)
        return data

    def get_id(self, item):
        return self.ids[item]

    def __len__(self):
        return self.ids.shape[0]

    def pad_or_cut(self, embeds, length):
        if embeds.shape[0] % length == 0:
            embeds.reshape([-1, length])
        if length > embeds.shape[-1]:
            return torch.nn.functional.pad(embeds, (0, length - embeds.shape[-1]), mode='constant', value=0)
        else:
            return embeds[:, :length] if len(embeds.shape) > 1 else embeds[:length]

class MultiDataset(Dataset):
    """
    :item:{'id':list[str],'embeds':tensor[n,num_ds,max_embeds_length],'labels':[n,num_labels]}
    """
    def __init__(self,config,ids,ds_list:list=None,max_embeds_length=1024,labels=None,train=True):
        """

        :param config: config
        :param ids:
        :param ds: the SingleDataset list
        :param labels: the train labels
        :param train:
        """
        self.device=config.device
        self.dtype=config.dtype
        assert np.all([d.device==self.device for d in ds_list])
        assert np.all([d.dtype==self.dtype for d in ds_list])


        self.num_embeds=len(ds_list)
        self.embeds_lens=[d.embeds_len for d in ds_list]
        self.max_embeds_length=max_embeds_length
        # assert np.all(np.array(self.embeds_lens)<=max_position_length)

        self.ids=ids
        self.embeds=ds_list
        assert labels is not None if train else 1
        self.train=train
        self.labels=labels

    def __getitem__(self, index):
        data={}
        data['id']=self.get_id(index)
        data['embeds']=self.auto_concate(
                [self.pad_or_cut(self.embeds[i][index]['embeds'],self.max_embeds_length) for i in range(self.num_embeds)]
              ,self.max_embeds_length)
        if self.train:
            data['labels']=torch.tensor(self.labels[index,:],dtype=self.dtype)
        return data

    def pad_or_cut(self,embeds, length):
        if embeds.shape[0]%length==0:
            embeds.reshape([-1,length])
        if length > embeds.shape[-1]:
            return torch.nn.functional.pad(embeds, (0, length - embeds.shape[-1]), mode='constant', value=0)
        else:
            return embeds[:, :length] if len(embeds.shape) > 1 else embeds[:length]

    def auto_concate(self,embeds, length):
        assert np.all(np.array([e.shape[-1] == length for e in embeds]))
        return torch.concat([embed.reshape([-1, length]) for embed in embeds])

    def get_id(self, item):
        return self.ids[item]

    def __len__(self):
        return self.ids.shape[0]

def get_ds():
    pass
def get_dl():
    pass
def dataset_split(ds,ratio=None,num=None):
    """
    :param ds:
    :param ratio: train dataset ratio
    :param num:
    :return:
    """
    assert ratio is not None or num is not None
    l=len(ds)
    if ratio:
        t=int(ratio*l)
        v=l-t
        return torch.utils.data.random_split(ds,[t,v]),t,v
        # return df,t,v
    else:
        return torch.utils.data.random_split(ds,[min(num,l),max(0,l-num)]),round(num/l,2),round(1-num/l,2)
        # return df,round(num/l,2),round(1-num/l,2)


if __name__=="__main__":
    config=Config('','','','','','',)
    ids1=np.load(r'C:\Users\23920\Desktop\research\demo\embeds\T5\train_ids.npy')
    embeds1=np.load(r'C:\Users\23920\Desktop\research\demo\embeds\T5\train_embeddings.npy')
    labels=np.load(r'C:\Users\23920\Desktop\research\demo\labels\make_labels_300_True_labels.npy')

    ids2=np.load(r'C:\Users\23920\Desktop\research\demo\embeds\EMS\train_ids.npy')
    embeds2=np.load(r'C:\Users\23920\Desktop\research\demo\embeds\EMS\train_embeds.npy')

    ids3=np.load(r'C:\Users\23920\Desktop\research\demo\embeds\ProtBert\train_ids.npy')
    embeds3=np.load(r'C:\Users\23920\Desktop\research\demo\embeds\ProtBert\train_embeddings.npy')

    ds1=SingleDataset(config,ids1,embeds1,labels=labels)
    ds2=SingleDataset(config,ids2,embeds2,labels=labels)
    ds3=SingleDataset(config,ids3,embeds3,labels=labels)
    ds=MultiDataset(config,ids1,[ds1,ds2,ds3],labels=labels)
    # print(ds1[:1]['embeds'])
    # print(ds2[:1]['embeds'])
    # print(ds[:1]['embeds'])

