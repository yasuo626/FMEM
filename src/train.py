

import os
import sys
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import lightning.pytorch as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger

import wandb
import logging

import matplotlib.pyplot as plt
import numpy as np
import math
import configparser
import torch
from torch.utils.data import Dataset,DataLoader,DataChunk

from utils.utils import Config
from dataset import SingleDataset,MultiDataset,dataset_split
from fusion_models import SingleEmbedsPerceptron,AttentionModel,Transformer_AttentionModel,Fusion_AttentionModel
from training_module import get_optimizer,get_lossfc,get_metrics,get_scheduler




class Lightning_Model(pl.LightningModule):
    def __init__(self,config,model,lossfc,optimizer,metrics,scheduler=None,tdl=None,vdl=None,train=True,is_log=True):
        """
            pytorch lightning model to train model
        :param config:config
        :param model: torch.nn.Module
        :param lossfc: func(pred,labels)
        :param optimizer: optimizer(model.parms,lr)
        :param metrics: {"metric name":func ...} func is a metric class contain update(pred,label),compute() two function
        :param train_dl:
        :param val_dl:
        :param scheduler: scheduler(optimizer) scheduler contain step method which could set the learning rate for optimizer
        :param train: boolean:True means training mode
        """
        super(Lightning_Model,self).__init__()
        self.conf=config

        self.infrance=model
        self.lossfc=lossfc
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.metrics=metrics
        self.state=train

        self.tdl,self.vdl=tdl , vdl


        self.is_log=is_log
        self.train_logs={ name:[] for name in metrics}
        self.val_logs={ name:[] for name in metrics}
        self.train_logs['train_loss']=[]
        self.val_logs['val_loss']=[]
    def forward(self,x):
        return self.infrance(x)
    def training_step(self, batch, batch_idx):
        inputs=batch
        ids=inputs["id"]
        embeds=inputs['embeds'].to(config.device)
        labels=inputs['labels'].to(config.device)
        pred=self.infrance(embeds)
        self.optimizer.zero_grad()
        for name in self.metrics:
            self.metrics[name].update(pred,labels.to(torch.int))
            self.train_logs[name].append(self.metrics[name].compute().item())
        loss=self.lossfc(pred,labels)
        self.train_logs["train_loss"].append(loss.item())
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def on_train_epoch_end(self):
        if self.is_log:
            for name in self.metrics:
                self.log("train_"+name, float(np.mean(self.train_logs[name])))
            self.log('train_loss', float(np.mean(self.train_logs['train_loss'])))

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            inputs=batch
            ids=inputs["id"]
            embeds=inputs['embeds'].to(config.device)
            labels=inputs['labels'].to(config.device)
            pred=self.infrance(embeds)
            self.optimizer.zero_grad()

            for name in self.metrics:
                self.metrics[name].update(pred,labels.to(torch.int))
                self.val_logs[name].append(self.metrics[name].compute().item())
            loss=self.lossfc(pred,labels)
            self.val_logs["val_loss"].append(loss.item())
        return {}


    def on_validation_epoch_end(self):
        if self.is_log:
            for name in self.metrics:
                self.log("val_"+name, float(np.mean(self.val_logs[name])))
            self.log('val_loss', float(np.mean(self.val_logs['val_loss'])))

    def configure_optimizers(self):
        return self.optimizer

    def train_dataloader(self):
        return self.tdl
def training():
    pass

import argparse
import json


def parse_dict_arg(arg_value):
    try:
        arg_dict = json.loads(arg_value)
        for key, value in arg_dict.items():
            if isinstance(value, str):
                try:
                    int_value = int(value)
                    arg_dict[key] = int_value
                except ValueError:
                    try:
                        float_value = float(value)
                        arg_dict[key] = float_value
                    except ValueError:
                        pass
        return arg_dict
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid JSON format")

def save_model_conf(config,path="",**kwargs):
    conf = configparser.ConfigParser()
    kwargs.update(config.params)
    conf['model_conf']=kwargs
    if path!="":
        with open(path, 'w') as configfile:
            conf.write(configfile)

if __name__ == "__main__":
    """
    -project_name
    -ids_path
    -embeds_path
    -labels_path
    -num_labels
    -output_dir
    -save_path

    -use_model ['sep','am','ta','fa']

    -optimizer ['adam','sgd','adamw']

    -scheduler ['linear','constant','cosine','warmup','none']
        linear:start_factor=0.9, end_factor=1.0
        constant:
        cosine:T_0=10,eta_min=0.05
        warmup: lr=0.1 minilr=0 warm_epoch=30,total_epoch=100,restart=True
    -lossfc ['bce','ce']

    -metrics ['f1score','accuracy','r2']
        num_classes=300, criteria="weighted","macro"

    -batch_size int
    -lr float
    -total_epoch int
    -val_every_n_epoch int
    -accelerator ['gpu','cpu','tpu']
    -logger ['wandb','tboard','none']

    -max_embeds_length
    -kwargs {"T_0":10,'eta_min':0.05 ...}
    """


    parser=argparse.ArgumentParser()
    parser.add_argument('-project_name',type=str,required=True,help='')
    parser.add_argument('-ids_path',type=str,required=True,help='')
    parser.add_argument('-embeds_path',type=str,default="",help='')
    parser.add_argument('-embeds_path_list',type=str,nargs="+",help='')
    parser.add_argument('-labels_path',type=str,required=True,help='')
    parser.add_argument('-num_labels',type=int,required=True,help='')
    parser.add_argument('-output_dir',type=str,default="./",help='')
    parser.add_argument('-save_path',type=str,default="",help='')
    parser.add_argument('-seed',type=int,default=1024,help='')


    parser.add_argument('-use_model',type=str,choices= ['sep', 'am', 'ta', 'fa'],default='sep',help='')
    parser.add_argument('-optimizer',type=str,choices= ['adam', 'sgd', 'adamw'],default='adam',help='')
    parser.add_argument('-lossfc',type=str,choices= ['bce','ce'],default='bce',help='')
    parser.add_argument('-scheduler',type=str,choices= ['linear', 'constant', 'cosine', 'warmup','none'],default='none',help='')
    parser.add_argument('-metrics',type=str,choices=['f1score','aucprc','accuracy', 'r2score'],nargs="+",help='')

    parser.add_argument('-lr',type=float,default=3e-4,help='')
    parser.add_argument('-batch_size',type=int,default=64,help='')
    parser.add_argument('-total_epoch',type=int,default=15,help='')
    parser.add_argument('-val_every_n_epoch',type=int,default=3,help='')

    parser.add_argument('-accelerator',type=str,choices=['gpu', 'cpu', 'tpu'],default='cpu',help='')
    parser.add_argument('-logger',type=str,choices=['wandb', 'tboard', 'none'],default='none',help='')

    parser.add_argument('-kwargs', type=parse_dict_arg)

    parser.add_argument('-max_embeds_length',type=int,default=1024,help='')
    parser.add_argument('-multilength_list',type=int,nargs="+",help='')

    args=parser.parse_args()
    kwargs={
        'train_spilit_ratio':0.8,
        'model':{'hidden_size1':512,'hidden_size2':128,},
        'optimizer':{},
        'scheduler':{'start_factor':1.0, 'end_factor':1.0},
        'lossfc':{},
        'metrics':{
            'f1score':{'num_classes':args.num_labels, 'criteria':"weighted"},
            'accuracy':{'threshold':0.1, 'criteria':"hamming"},
            'aucprc':{'num_labels':args.num_labels,'average':"macro"},
            'r2score':{},
        }
    }
    args.kwargs=parse_dict_arg(args.kwargs if args.kwargs is not None else '{"x":0}')
    for k in args.kwargs.keys():
        kwargs[k]=args.kwargs[k]

    assert args.embeds_path!="" or args.embeds_path_list is not None
    # args.project_name
    # args.ids_path
    # args.embeds_path
    # args.labels_path
    # args.num_labels
    # args.output_dir
    # args.save_path
    # args.use_model
    # args.optimizer
    # args.scheduler
    # args.metrics
    # args.lr
    # args.batch_size
    # args.total_epoch
    # args.val_every_n_epoch
    # args.accelerator
    # args.logger
    # args.kwargs

    if args.multilength_list is None:
        args.multilength_list=[1]*(len(args.embeds_path_list) if args.embeds_path_list is not None else 1)




    seed_everything(args.seed)
    config = Config(task_name=args.project_name,label_path=args.labels_path,ids_path=args.ids_path,
                    embeds_path=args.embeds_path,embeds_path_list=args.embeds_path_list,
                    output_dir=args.output_dir,log_dir=args.output_dir+"/logs",model_name=args.use_model,num_labels=args.num_labels,
                    max_position_length=args.max_embeds_length,device=args.accelerator,
                    batch=args.batch_size,lr=args.lr,epoch=args.total_epoch,)

    ids = np.load(config.ids_path)
    labels = np.load(config.labels_path)
    embeds, model, tdl, vdl, optimizer, scheduler, lossfc, metrics = None, None, None, None, None, None, None, None,

    # get single or multi ds
    is_fusion=True if args.use_model in [ 'am', 'ta', 'fa'] else False
    if is_fusion:
        multi_ds=[]
        for i,path in enumerate(config.embeds_path_list):
            embed=np.load(path)
            ds=SingleDataset(config,ids=ids,embeds=embed,max_embeds_length=args.multilength_list[i]*config.max_position_length,labels=labels,train=True)
            multi_ds.append(ds)
        ds=MultiDataset(config,ids,multi_ds,max_embeds_length=config.max_position_length,labels=labels,train=True)

    else:
        embeds=np.load(config.embeds_path)
        ds = SingleDataset(config,ids=ids,embeds=embeds,max_embeds_length=config.max_position_length,labels=labels,train=True)

    [tds, vds], n1, n2 = dataset_split(ds,kwargs['train_spilit_ratio'])
    tdl, vdl = DataLoader(tds, batch_size=config.batch, shuffle=True, drop_last=True), DataLoader(vds,batch_size=config.batch,shuffle=False,drop_last=False,)

    # get model
    if args.use_model=="sep":
        model=SingleEmbedsPerceptron(config,input_dim=config.max_position_length,output_dim=config.num_labels,
                                     hidden_size1=kwargs['model']['hidden_size1'],
                                     hidden_size2=kwargs['model']['hidden_size2'],
                                     )
        save_model_conf(config,f"../conf/{args.project_name}_{args.use_model}.ini",input_dim=config.max_position_length,output_dim=config.num_labels,
                                     hidden_size1=kwargs['model']['hidden_size1'],
                                     hidden_size2=kwargs['model']['hidden_size2'],)

    elif args.use_model=="am":
        model=AttentionModel(config,num_embeds=len(args.embeds_path_list),max_embdes_length=config.max_position_length, output_dim=config.num_labels,
                                         hidden_size1=kwargs['model']['hidden_size1'],
                                         hidden_size2=kwargs['model']['hidden_size2'])
        save_model_conf(config,f"../conf/{args.project_name}_{args.use_model}.ini",input_dim=config.max_position_length,output_dim=config.num_labels,
                                     hidden_size1=kwargs['model']['hidden_size1'],
                                     hidden_size2=kwargs['model']['hidden_size2'],)

    elif args.use_model=="ta":
        model=Transformer_AttentionModel(config,num_embeds=len(args.embeds_path_list),output_dim=config.num_labels)
        save_model_conf(config,f"../conf/{args.project_name}_{args.use_model}.ini",num_embeds=len(args.embeds_path_list),output_dim=config.num_labels)
    elif args.use_model=="fa":
        model=Fusion_AttentionModel(config,num_embeds=len(config.embeds_path_list),
                                    max_embdes_length=config.max_position_length,
                                    output_dim=config.num_labels,
                                    hidden_size1=kwargs['model']['hidden_size1'],
                                    hidden_size2=kwargs['model']['hidden_size2'],
                                    )
        save_model_conf(config,f"../conf/{args.project_name}_{args.use_model}.ini",num_embeds=len(config.embeds_path_list),
                                    max_embdes_length=config.max_position_length,
                                    output_dim=config.num_labels,
                                    hidden_size1=kwargs['model']['hidden_size1'],
                                    hidden_size2=kwargs['model']['hidden_size2'],)
    else:
        raise ValueError("invalid model")
    model=model.to(config.device)


    # train optim
    optimizer = get_optimizer(model, args.optimizer, lr=config.lr,**kwargs['optimizer'])
    if scheduler is not None:
        scheduler = get_scheduler(args.scheduler, optimizer,**kwargs['scheduler'])
    lossfc = get_lossfc(args.lossfc,**kwargs['lossfc'])
    metrics = {metric: get_metrics(metric, **(kwargs['metrics'][metric])).to(config.device) for metric in args.metrics}


    # controller
    is_log=True
    logger=None
    if args.logger=='none':
        is_log=False
    elif args.logger=='wandb':
        logger = WandbLogger(project=config.task_name,save_dir=config.log_dir)
    elif args.logger=='tboard':
        pass
    else:
        print("unknown logger")


    # model save
    # save=False
    # if args.save_path!="":
    save=True
    if not os.path.exists(config.output_dir+f'/{config.task_name}'):
        os.mkdir(config.output_dir+f'/{config.task_name}')
    plmodel = Lightning_Model(config=config,model=model, lossfc=lossfc, optimizer=optimizer, scheduler=scheduler, metrics=metrics,tdl=None,vdl=None, train=True,is_log=is_log)
    trainer = pl.Trainer(accelerator=args.accelerator, check_val_every_n_epoch=args.val_every_n_epoch,
                         enable_checkpointing=save, max_epochs=args.total_epoch,
                         logger=logger,default_root_dir=config.output_dir+f'/{config.task_name}')
    trainer.fit(plmodel, tdl, vdl)









