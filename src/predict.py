import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import configparser
import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader,Dataset
from lightning.pytorch import seed_everything
from utils.utils import Config
from dataset import SingleDataset,MultiDataset,dataset_split
from fusion_models import SingleEmbedsPerceptron,AttentionModel,Transformer_AttentionModel,Fusion_AttentionModel
from training_module import get_optimizer,get_lossfc,get_metrics,get_scheduler


def predict(dl,nsample,nlabel,nbatch,model,device):
    """
        predict
    :param dl: dataloader
    :param nsample: num of prediction samples
    :param nlabel: predict labels num
    :param nbatch: dataloader batch size
    :param model: model nn.Module
    :return:pred_ids,preds_labels
    """
    pred_ids=[]
    preds=torch.empty([nsample,nlabel])
#     for i,inputs in enumerate(test_dl):
    for i,inputs in tqdm.tqdm(enumerate(dl),total=nsample//nbatch):
        pred_ids.extend(inputs["id"])
        preds[i*nbatch:(i+1)*nbatch,:]=model(inputs['embeds'].to(device))
    return np.array(pred_ids),preds.detach().numpy()

def create_prediction(pred_ids,preds,label_names,threshold=0.0,chunk_size=None,path=''):
    """

    :param pred_ids:
    :param preds:
    :param label_names:
    :param threshold: confidence threshold
    :param chunk_size:
    :param path: save path
    :return:
    """
    if chunk_size is None:
        chunk_size=len(preds)//100
    assert chunk_size<=len(preds)
    subs=pd.DataFrame()
    chunks=[range(i,min(i+chunk_size,len(preds))) for i in range(0,len(preds),chunk_size)]
    for chunk in chunks:
        print(chunk)
#     for chunk in tqdm.tqdm(chunks,total=len(preds)//chunk_size):
        sub=pd.DataFrame(data=preds[chunk],columns=label_names,index=pred_ids[chunk])
        sub=sub.T.unstack().reset_index(name='pred')
        sub=sub.loc[sub['pred']>threshold,:]
        subs=pd.concat([subs,sub],axis=0)
    if path!='':
        try:
            subs.to_csv(path,sep='\t',header=False,index=False)
        except:
            subs.to_csv('./submission.tsv',sep='\t',header=False,index=False)
            print('path error! auto save to ./submission.tsv')
    return subs
def load_ckpt_state_dict(state_dict,old_name="",new_name=""):
    old_name=old_name+'.'
    get_state_dict={}
    for k in state_dict.keys():
        if old_name in k:
            get_state_dict[k.replace(old_name,new_name)]=state_dict[k]
        else:
            get_state_dict[k]=state_dict[k]
    return get_state_dict

import argparse
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-task_name',type=str,required=True,help='')
    parser.add_argument('-use_model',type=str,choices= ['sep', 'am', 'ta', 'fa'],default='sep',help='')
    parser.add_argument('-model_conf',type=str,required=True,help='')
    parser.add_argument('-ckpt_path',type=str,required=True,help='')
    parser.add_argument('-ids_path',type=str,required=True,help='')
    parser.add_argument('-label_names_path',type=str,required=True,help='')

    parser.add_argument('-embeds_path',type=str,default="",help='')
    parser.add_argument('-embeds_path_list',type=str,nargs="+",help='')
    parser.add_argument('-num_labels',type=int,required=True,help='')

    parser.add_argument('-output_dir',type=str,default="./",help='')
    parser.add_argument('-save_path',type=str,default="",help='')
    parser.add_argument('-seed',type=int,default=1024,help='')

    parser.add_argument('-max_embeds_length',type=int,default=1024,help='')
    parser.add_argument('-accelerator',type=str,choices=['gpu', 'cpu', 'tpu'],default='cpu',help='')

    parser.add_argument('-confidence_threshold',type=float,default=0.01,help='')


    args=parser.parse_args()
    # if args.embeds_path!="":
    #     args.embeds_path_list=None
    assert args.embeds_path is not None or args.embeds_path_list is not None

    if args.save_path=="":
        args.save_path=args.output_dir+f"/prediction_{args.task_name}_{args.use_model}_{args.num_labels}.tsv"
    conf = configparser.ConfigParser()
    conf.read(args.model_conf)
    kwargs=conf['model_conf']

    seed_everything(args.seed)
    config = Config(task_name=args.task_name,label_path="",ids_path=args.ids_path,
                    embeds_path=args.embeds_path,embeds_path_list=args.embeds_path_list,
                    output_dir=args.output_dir,log_dir=args.output_dir+"/logs",model_name=args.use_model,num_labels=args.num_labels,
                    max_position_length=args.max_embeds_length,device=args.accelerator,)

    ids = np.load(config.ids_path,allow_pickle=True)
    label_names = np.load(args.label_names_path,allow_pickle=True)

    embeds, model, dl=  None, None, None,

    # data
    is_fusion=True if args.use_model in [ 'am', 'ta', 'fa'] else False
    if is_fusion:
        multi_ds=[]
        for path in config.embeds_path_list:
            embed=np.load(path)
            ds=SingleDataset(config,ids=ids,embeds=embed,max_embeds_length=args.max_embeds_length,train=False)
            multi_ds.append(ds)
        ds=MultiDataset(config,ids,multi_ds,max_embeds_length=config.max_position_length,train=False)

    else:
        embeds=np.load(config.embeds_path)
        ds = SingleDataset(config,ids=ids,embeds=embeds,max_embeds_length=args.max_embeds_length,train=False)
    dl = DataLoader(ds, batch_size=config.batch, shuffle=False, drop_last=False)
    nsample=len(ds)

    # model
    if args.use_model=="sep":
        model=SingleEmbedsPerceptron(config,input_dim=config.max_position_length,output_dim=config.num_labels,
                                     hidden_size1=int(kwargs['hidden_size1']),
                                     hidden_size2=int(kwargs['hidden_size2'])
                                     )
    elif args.use_model=="am":
        model=AttentionModel(config,num_embeds=len(args.embeds_path_list),max_embdes_length=config.max_position_length, output_dim=config.num_labels,
                                         hidden_size1=int(kwargs['hidden_size1']),
                                         hidden_size2=int(kwargs['hidden_size2'])
                             )
    elif args.use_model=="ta":
        model=Transformer_AttentionModel(config,num_embeds=len(args.embeds_path_list),output_dim=config.num_labels)

    elif args.use_model=="fa":
        model=Fusion_AttentionModel(config,num_embeds=len(config.embeds_path_list),
                                    max_embdes_length=config.max_position_length,
                                    hidden_embedding=1024,
                                    output_dim=config.num_labels,
                                    hidden_size1=int(kwargs['hidden_size1']),
                                    hidden_size2=int(kwargs['hidden_size2'])
                                    )

    model.load_state_dict(load_ckpt_state_dict(torch.load(args.ckpt_path)['state_dict'],"infrance","") )
    model=model.to(config.device)
    model.eval()
    pred_ids,preds_labels=predict(dl,nsample=nsample,nlabel=config.num_labels,nbatch=config.batch,model=model,device=config.device)
    create_prediction(pred_ids=pred_ids,preds=preds_labels,label_names=label_names,threshold=args.confidence_threshold,path=args.save_path)






