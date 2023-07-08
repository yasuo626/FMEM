import torch
# optimizer
import torch.optim
def get_optimizer(model:torch.nn.Module,optimizer:str="adam",**kwargs):
    """
        params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) – learning rate (default: 1e-3)
        betas (Tuple[float, float], optional) – coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional) – term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional) – weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional) – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond (default: False)
        foreach (bool, optional) – whether foreach implementation of optimizer is used. If unspecified by the user (so foreach is None), we will try to use foreach over the for-loop implementation on CUDA, since it is usually significantly more performant. (default: None)
        maximize (bool, optional) – maximize the params based on the objective, instead of minimizing (default: False)
        capturable (bool, optional) – whether this instance is safe to capture in a CUDA graph. Passing True can impair ungraphed performance, so if you don’t intend to graph capture this instance, leave it False (default: False)
        differentiable (bool, optional) – whether autograd should occur through the optimizer step in training. Otherwise, the step() function runs in a torch.no_grad() context. Setting to True can impair performance, so leave it False if you don’t intend to run autograd through this instance (default: False)
        fused (bool, optional) – whether the fused implementation (CUDA only) is used. Currently, torch.float64, torch.float32, torch.float16, and torch.bfloat16 are supported. (default: None)

    :param model:
    :param optimizer:
    :param kwargs:
    :return:
    """
    selections={
        'adam':torch.optim.Adam,
        'sgd':torch.optim.SGD,
        'adamw':torch.optim.AdamW,
    }
    return selections[optimizer](model.parameters(),**kwargs)


# lr_scheduler
from torch.optim.lr_scheduler import LinearLR,ConstantLR,CosineAnnealingWarmRestarts
class WarmupLR(object):
    def __init__(self,optimizer,lr,minilr,warm_epoch,total_epoch,restart=False):
        """
        :param optimizer: opt
        :param lr: max learn rate
        :param minilr: min learn rate
        :param warm_epoch:
        :param total_epoch:
        :param restart: reset while epoch > total
        """
        self.lr=lr
        self.warm_epoch=warm_epoch
        self.total_epoch=total_epoch
        self.restart=restart
        self.minilr=minilr
        self.count=0
        self.cur=0
        self.opt=optimizer
        for group in self.opt.param_groups:
            group['lr']=self.get_lr()
        self.count+=1
        self.cur+=1


    def warm_up_lr(self):
        if self.cur>self.total_epoch:
            if self.restart:
                self.cur=0
            else:
                return self.minilr
        return max(self.minilr,self.calc(self.lr,self.cur, self.warm_epoch, self.total_epoch))

    def calc(self,lr, epoch,warm_epoch,total_epoch):
        if epoch < warm_epoch:
            return lr * epoch / warm_epoch
        else:
            return (math.cos((epoch - warm_epoch) / (total_epoch - warm_epoch) * math.pi) + 1) / 2 * lr

    def step(self):
        for group in self.opt.param_groups:
            group['lr']=self.get_lr()
        self.count+=1
        self.cur+=1

    def get_lr(self):
        return self.warm_up_lr()
def get_scheduler(sched,optimizer,**kwargs):
    """
    linearLR:
        optimizer (Optimizer) – Wrapped optimizer.
        start_factor (float) – The number we multiply learning rate in the first epoch. The multiplication factor changes towards end_factor in the following epochs. Default: 1./3.
        end_factor (float) – The number we multiply learning rate at the end of linear changing process. Default: 1.0.
        total_iters (int) – The number of iterations that multiplicative factor reaches to 1. Default: 5.
        last_epoch (int) – The index of the last epoch. Default: -1.
    Constant:
        optimizer (Optimizer) – Wrapped optimizer.
        factor (float) – The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int) – The number of steps that the scheduler decays the learning rate. Default: 5.
        last_epoch (int) – The index of the last epoch. Default: -1.
    cosine:
        optimizer (Optimizer) – Wrapped optimizer.
        T_0 (int) – Number of iterations for the first restart.
        T_mult (int, optional) – A factor increases
        eta_min (float, optional) – Minimum learning rate. Default: 0.
        last_epoch (int, optional) – The index of last epoch. Default: -1.
        verbose (bool) – If True, prints a message to stdout for each update. Default: False.
    warmup:
        optimizer: opt
        lr: max learn rate
        minilr: min learn rate
        warm_epoch:
        total_epoch:
        restart: reset while epoch > total

    exp:
        scheduler = LinearLR(opt, start_factor=0.1, total_iters=10)
        scheduler = CosineAnnealingWarmRestarts(opt,T_0=10,eta_min=0.05)
        scheduler = WarmupLR(opt,0.1,0,30,100,restart=True)

    :param optimizer:
    :param kwargs:
    :return:
    """
    selections={
        'linear':LinearLR,
        'constant':ConstantLR,
        'restart':CosineAnnealingWarmRestarts,
        'warmup':WarmupLR
    }
    return selections[sched](optimizer,**kwargs)


# loss
def get_lossfc(loss,**kwargs):
    selections={
        'ce':torch.nn.CrossEntropyLoss,
        'bce':torch.nn.BCELoss,
        'bcelog':torch.nn.BCEWithLogitsLoss,
        'mse':torch.nn.MSELoss
    }
    return selections[loss](**kwargs)


# eval

import torcheval
from torcheval.metrics import R2Score,MultilabelAccuracy,MultilabelAUPRC
from torchmetrics.classification import F1Score


def get_metrics(metric,**kwargs):
    """
    sample:
        f1_score = F1Score(num_labels=config.num_labels, criteria="hamming")
        accuracy = MultilabelAccuracy(num_labels=config.num_labels, criteria="hamming")
    :param metric:
        classify:
            AUPRC: Average Precision under the Precision-Recall Curve for multilabel classification.
                num_labels
                average:'macro':return mean ,None:return each label socres,'weighted':weighted by true samples
            MULTILABELACCURACY:
                threshold: Threshold for converting input into predicted labels for each sample.
                criteria:
                    'exact_match' [default]: The set of labels predicted for a sample must exactly match the corresponding set of labels in target. Also known as subset accuracy.
                    'hamming':Fraction of correct labels over total number of labels.
                    'overlap': The set of labels predicted for a sample must overlap with the corresponding set of labels in target.
                    'contain': The set of labels predicted for a sample must contain the corresponding set of labels in target.
                    'belong': The set of labels predicted for a sample must (fully) belong to the corresponding set of labels in target.
        regression:
            F1Score:
                num_classes
                average:'macro':return mean ,none:return each label socres,'weighted':weighted by true samples
            R2SCORE:
    :param kwargs:
    :return:
    """
    selections={
        'f1score':F1Score,
        'accuracy':MultilabelAccuracy,
        'auprc':MultilabelAUPRC,
        'r2score':R2Score,
    }
    return selections[metric](**kwargs)