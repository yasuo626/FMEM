
import torch
def set_device(device:str):
    device=device.lower()
    assert device in ['cpu','cuda','gpu']
    if device=='gpu':
        device='cuda'
    try:
        udevice=torch.device(device)
    except:
        print("error device")
        udevice=torch.device('cpu')
    return udevice

class Config(object):

    def load_conf(self, params_dict):
        self.task_name = params_dict.get('task_name')
        self.model_name = params_dict.get('model_name')
        self.num_labels = params_dict.get('num_labels')
        self.max_position_length =  int(params_dict.get('max_position_length'))

        self.lr = float(params_dict.get('lr'))
        self.batch = int(params_dict.get('batch'))
        self.epoch = int(params_dict.get('epoch'))

        self.params = {
            'task_name': self.task_name,
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_position_length': self.max_position_length,
            'lr': float(self.lr),
            'batch': int(self.batch),
            'epoch': int(self.epoch)
        }


    def copy(self,config):
        self.labels_path=config.label_path
        self.ids_path=config.ids_path
        self.embeds_path=config.embeds_path
        self.output_dir=config.output_dir
        self.model_name=config.model_name
        self.log_dir=config.log_dir

        self.num_labels=config.num_labels
        self.max_position_length=config.max_position_length

        self.device=set_device(config.device)
        self.dtype=config.dtype

        self.lr=config.lr
        self.batch=config.batch
        self.epoch=config.epoch

    def __init__(self,task_name,label_path,ids_path,output_dir,log_dir,
                 model_name,embeds_path=None,embeds_path_list=None,num_labels=300,max_position_length=1024,device='cpu',dtype=torch.float32,
                 batch=128,lr=2e-3,epoch=10,
                 ):
        self.task_name=task_name
        self.labels_path=label_path
        self.ids_path=ids_path
        assert embeds_path is not None or embeds_path_list is not None
        self.embeds_path=embeds_path
        self.embeds_path_list=embeds_path_list
        self.output_dir=output_dir
        self.model_name=model_name
        self.log_dir=log_dir

        self.num_labels=num_labels
        self.max_position_length=max_position_length

        self.device=set_device(device)
        self.dtype=dtype

        self.lr=lr
        self.batch=batch
        self.epoch=epoch

        self.params = {
            'task_name': self.task_name,
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_position_length':  int(self.max_position_length),
            'lr': float(self.lr),
            'batch': int(self.batch),
            'epoch': int(self.epoch)
        }
