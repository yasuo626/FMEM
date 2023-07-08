
import argparse
import gc
import os.path
import sys

import numpy as np
import transformers
from Bio import SeqIO
import re
import tqdm



print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def get_fasta(fasta_path,format='fasta'):
    return SeqIO.parse(fasta_path,format)
def get_fasta_size(fasta):
    i=0
    for i,_ in enumerate(fasta):
        pass
    return i+1

from utils.utils import set_device

def create_embeds(fasta_path,tokenizer,model,embeds_shape:list,max_position_length,output_dir,task_name,model_name,seq_format='fasta',device='cpu',re_func=None,seq_encode=None,save=True,show_process=False):
    """
        create embeds for proteins,output to output_dir/task_name_ids.npy,task_name_embeds.npy
    :param fasta_path: str path
    :param tokenizer: encode the seq
    :param model: model to get embeds
    :param max_position_length: based on model
    :param output_dir:  output dir
    :param task_name: str
    :param model_name: str
    :param seq_format: fasta or other SeqIo supports
    :param re_func: re_func(protein.id) to get the right id from different structure of id
    :param seq_encode: seq_encode(protein.seq,tokenizer,model) return the seq's embeds
    :param show_process: boolen True to show process bar
    :return: ids,embeds
    """
    device=set_device(device)
    model=model.to(device)
    model.eval()
    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir)
        except:
            raise OSError(f"cant create dir:{output_dir}")

    if re_func is None:
        re_func=lambda x:x

    if seq_encode is None:
        def seq_encode(seq,tokenizer,model):
            input_ids=" ".join(re.sub(r"[UZOB]", "X", str(seq)))
            inputs = tokenizer(input_ids, max_length=max_position_length, padding='max_length', truncation=True,
                               return_tensors='pt')
            return model(**(inputs.to(device)), )["pooler_output"].reshape(-1).cpu().detach().numpy()


    fasta_seqs=get_fasta(fasta_path,seq_format)
    total=get_fasta_size(fasta_seqs)
    out_shape=list(embeds_shape)
    out_shape.insert(0,total)

    fasta_seqs=get_fasta(fasta_path,seq_format)
    ids,embeds=np.empty([total],dtype='U20'),np.empty(out_shape,dtype=float)
    if show_process:
        for i, protein in tqdm.tqdm(enumerate(fasta_seqs),total=total):
            ids[i]=re_func(protein.id)
            embeds[i]=seq_encode(protein.seq,tokenizer,model)
    else:
        for i, protein in enumerate(fasta_seqs):
            ids[i]=re_func(protein.id)
            embeds[i]=seq_encode(protein.seq,tokenizer,model)

    ids=ids.squeeze()
    embeds=embeds.squeeze()
    gc.collect()
    if save:
        try:
            if not os.path.exists(output_dir + f'/{model_name}'):
                os.mkdir(output_dir + f'/{model_name}')
            np.save(output_dir+f'/{model_name}'+f'/{task_name}_ids.npy',ids)
            np.save(output_dir+f'/{model_name}'+f'/{task_name}_embeds.npy',embeds)
        except:
            print(f"save error ,auto save to : output_dir/temp_XX.npy")
            np.save(output_dir+f'/temp_ids.npy',ids)
            np.save(output_dir+f'/temp_embeds.npy',embeds)
    return ids, embeds

def create_embeds_plus(fasta_path,tokenizer,model,embeds_shape:list,max_position_length,batch,output_dir,task_name,model_name,seq_format='fasta',device='cpu',re_func=None,seq_encode=None,save=True,show_process=False):
    """
        create embeds for proteins,output to output_dir/task_name_ids.npy,task_name_embeds.npy
    :param fasta_path: str path
    :param tokenizer: encode the seq
    :param model: model to get embeds
    :param max_position_length: based on model
    :param output_dir:  output dir
    :param task_name: str
    :param model_name: str
    :param seq_format: fasta or other SeqIo supports
    :param re_func: re_func(protein.id) to get the right id from different structure of id
    :param seq_encode: seq_encode(protein.seq,tokenizer,model) return the seq's embeds
    :param show_process: boolen True to show process bar
    :return: ids,embeds
    """
    device=set_device(device)
    model=model.to(device)
    model.eval()
    if not os.path.isdir(output_dir):
        try:
            os.mkdir(output_dir)
        except:
            raise OSError(f"cant create dir:{output_dir}")

    if re_func is None:
        re_func=lambda x:x


    fasta_seqs=get_fasta(fasta_path,seq_format)
    total=get_fasta_size(fasta_seqs)
    out_shape=list(embeds_shape)
    out_shape.insert(0,total)

    fasta_seqs=get_fasta(fasta_path,seq_format)
    ids,embeds=np.empty([total],dtype='U20'),np.empty(out_shape,dtype=float).squeeze()

    chunks = [range(i, min(i + batch, total)) for i in range(0, total, batch)]
    seqs=[]
    for i,seq in enumerate(fasta_seqs):
        ids[i] = re_func(seq.id)
        seqs.append(" ".join(re.sub(r"[UZOB]", "X", str(seq))))
    seqs = np.array(seqs)
    if show_process:
        for i, chunk in tqdm.tqdm(enumerate(chunks),total=len(chunks)):
            encode = tokenizer.batch_encode_plus(seqs[chunk], max_length=max_position_length, padding='max_length',
                                                 return_tensors='pt') #'input_ids' token_type_ids 'attention_mask'
            embeds[chunk]= model(**(encode.to(device)), )["pooler_output"].cpu().detach().numpy()
    else:
        for i, chunk in enumerate(chunks):
            encode = tokenizer.batch_encode_plus(seqs[chunk], max_length=max_position_length, padding='max_length',
                                                 return_tensors='pt') #'input_ids' token_type_ids 'attention_mask'
            embeds[chunk]= model(**(encode.to(device)), )["pooler_output"].cpu().detach().numpy()

    ids=ids.squeeze()
    embeds=embeds.squeeze()
    gc.collect()
    if save:
        try:
            if not os.path.exists(output_dir + f'/{model_name}'):
                os.mkdir(output_dir + f'/{model_name}')
            np.save(output_dir+f'/{model_name}'+f'/{task_name}_ids.npy',ids)
            np.save(output_dir+f'/{model_name}'+f'/{task_name}_embeds.npy',embeds)
        except:
            print(f"save error ,auto save to : output_dir/temp_XX.npy")
            np.save(output_dir+f'/temp_ids.npy',ids)
            np.save(output_dir+f'/temp_embeds.npy',embeds)
    return ids, embeds


# MODELS={
#     't5':{
#         'tokenizer':transformers.AutoTokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50"),
#         'model':transformers.T5Model.from_pretrained("Rostlab/prot_t5_xl_uniref50"),
#           },
#     'protbert':{
#         'tokenizer':transformers.BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False),
#         'model':transformers.BertModel.from_pretrained("Rostlab/prot_bert"),
#           },
#     'esm':{
#         'tokenizer':transformers.EsmTokenizer.from_pretrained(''),
#         'model':transformers.EsmModel.from_pretrained(''),
#           },
#     'esm':{
#         'tokenizer':transformers.from_pretrained(''),
#         'model':transformers.EsmModel.from_pretrained(''),
#           },
# }
MODELS={
    'prot_t5_xl_uniref50':{
        # 'tokenizer':transformers.AutoTokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50'),
        # 'model':transformers.AutoModel.from_pretrained('Rostlab/prot_t5_xl_uniref50')
    },
    'prot_t5_xl_half_uniref50-enc': {
        # 'tokenizer': transformers.AutoTokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc'),
        # 'model': transformers.AutoModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc')
    },
    'prot_t5_base_mt_uniref50':{
        # 'tokenizer':transformers.AutoTokenizer.from_pretrained('Rostlab/prot_t5_base_mt_uniref50'),
        # 'model':transformers.AutoModel.from_pretrained('Rostlab/prot_t5_base_mt_uniref50')
    },
    'prot_t5_xxl_bfd': {
        # 'tokenizer': transformers.AutoTokenizer.from_pretrained('Rostlab/prot_t5_xxl_bfd'),
        # 'model': transformers.AutoModel.from_pretrained('Rostlab/prot_t5_xxl_bfd')
    },
    'prot_t5_xxl_uniref50':{
        # 'tokenizer':transformers.AutoTokenizer.from_pretrained('Rostlab/prot_t5_xxl_uniref50'),
        # 'model':transformers.AutoModelForSeq2SeqLM.from_pretrained('Rostlab/prot_t5_xxl_uniref50')
    },
    'prot_t5_xl_bfd': {
        # 'tokenizer': transformers.AutoTokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd'),
        # 'model': transformers.AutoModelWithLMHead.from_pretrained('Rostlab/prot_t5_xl_bfd')
    },



    'prot_bert_bfd':{
        # 'tokenizer':transformers.AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd'),
        # 'model':transformers.AutoModelForMaskedLM.from_pretrained('Rostlab/prot_bert_bfd')
    },
    'prot_bert': {
        # 'tokenizer': transformers.AutoTokenizer.from_pretrained('Rostlab/prot_bert'),
        # 'model': transformers.AutoModelForMaskedLM.from_pretrained('Rostlab/prot_bert')
    },
    'prot_albert': {
        # 'tokenizer': transformers.AutoTokenizer.from_pretrained('Rostlab/prot_albert'),
        # 'model': transformers.AutoModel.from_pretrained('Rostlab/prot_albert')
    },
    'prot_bert_bfd_ss3': {
        # 'tokenizer': transformers.AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd_ss3'),
        # 'model': transformers.AutoModelForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd_ss3')
    },
    'prot_bert_bfd_membrane':{
        # 'tokenizer':transformers.AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd_membrane'),
        # 'model':transformers.AutoModelForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd_membrane')
    },
    'prot_bert_bfd_localization': {
        # 'tokenizer': transformers.AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd_localization'),
        # 'model': transformers.AutoModelForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd_localization')
    },


    'prot_electra_generator_bfd': {
        # 'tokenizer': transformers.AutoTokenizer.from_pretrained('Rostlab/prot_electra_generator_bfd'),
        # 'model': transformers.AutoModelForMaskedLM.from_pretrained('Rostlab/prot_electra_generator_bfd')
    },
    'prot_electra_discrimator_bfd':{
        # 'tokenizer':transformers.AutoTokenizer.from_pretrained('Rostlab/prot_electra_discrimator_bfd'),
        # 'model':transformers.AutoModelForPreTraining.from_pretrained('Rostlab/prot_electra_discrimator_bfd')
    },


    'prot_xlnet': {
        # 'tokenizer': transformers.AutoTokenizer.from_pretrained('Rostlab/prot_xlnet'),
        # 'model': transformers.AutoModel.from_pretrained('Rostlab/prot_xlnet')
    },
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_tokenizer_model(model_name,pretrained_path=""):
    fromhub=True
    if pretrained_path !="":
        fromhub=False

    models={"embeds_shape":[1,1024]}
    if model_name == 'prot_t5_xl_uniref50':
        models['tokenizer']=ransformers.AutoTokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50') if fromhub else transformers.AutoTokenizer.from_pretrained(pretrained_path,local_files_only=True)
        models['model']=transformers.AutoModelForSeq2SeqLM.from_pretrained('Rostlab/prot_t5_xl_uniref50') if fromhub else transformers.AutoModelForSeq2SeqLM.from_pretrained(pretrained_path,local_files_only=True)
        models['embeds_shape']=[1,1024,1024]
    elif model_name == 'prot_t5_xl_half_uniref50-enc':
        models['tokenizer']=transformers.AutoTokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc')  if fromhub else transformers.AutoTokenizer.from_pretrained(pretrained_path, do_lower_case=False)
        models['model']=transformers.AutoModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc') if fromhub else transformers.AutoModel.from_pretrained(pretrained_path,local_files_only=True)

    elif model_name == 'prot_t5_xxl_bfd':
        models['tokenizer']=transformers.AutoTokenizer.from_pretrained('Rostlab/prot_t5_xxl_bfd') if fromhub else transformers.AutoTokenizer.from_pretrained(pretrained_path,local_files_only=True)
        models['model']=transformers.AutoModel.from_pretrained('Rostlab/prot_t5_xxl_bfd') if fromhub else transformers.AutoModel.from_pretrained(pretrained_path,local_files_only=True)

    elif model_name == 'prot_t5_xxl_uniref50':
        models['tokenizer']=ransformers.AutoTokenizer.from_pretrained('Rostlab/prot_t5_xxl_uniref50') if fromhub else transformers.AutoTokenizer.from_pretrained(pretrained_path,local_files_only=True)
        models['model']=transformers.AutoModelForSeq2SeqLM.from_pretrained('Rostlab/prot_t5_xxl_uniref50')

    elif model_name == 'prot_t5_xl_bfd':
        models['tokenizer']=transformers.AutoTokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd') if fromhub else transformers.AutoTokenizer.from_pretrained(pretrained_path,local_files_only=True)
        models['model']=transformers.AutoModelWithLMHead.from_pretrained('Rostlab/prot_t5_xl_bfd') if fromhub else transformers.AutoModelWithLMHead.from_pretrained(pretrained_path,local_files_only=True)


    elif model_name == 'prot_bert_bfd':
        models['tokenizer']=transformers.AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd') if fromhub else transformers.AutoTokenizer.from_pretrained(pretrained_path,local_files_only=True)
        models['model'] =transformers.AutoModelForMaskedLM.from_pretrained('Rostlab/prot_bert_bfd') if fromhub else transformers.AutoModelForMaskedLM.from_pretrained(pretrained_path,local_files_only=True)
        models['embeds_shape']=[1,1024,30]
    elif model_name == 'prot_bert':
        models['tokenizer']=transformers.BertTokenizer.from_pretrained('Rostlab/prot_bert') if fromhub else transformers.BertTokenizer.from_pretrained(pretrained_path,local_files_only=True)
        models['model']=transformers.BertModel.from_pretrained('Rostlab/prot_bert') if fromhub else transformers.BertModel.from_pretrained(pretrained_path,local_files_only=True)
    elif model_name == 'prot_albert':
        models['tokenizer']=transformers.AlbertTokenizer.from_pretrained('Rostlab/prot_albert') if fromhub else transformers.AlbertTokenizer.from_pretrained(pretrained_path,local_files_only=True)
        models['model']=transformers.AutoModel.from_pretrained('Rostlab/prot_albert') if fromhub else transformers.AutoModel.from_pretrained(pretrained_path,local_files_only=True)
        models['embeds_shape']=[1,4096]

    elif model_name == 'prot_bert_bfd_ss3':
        models['tokenizer']=transformers.AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd_ss3') if fromhub else transformers.AutoTokenizer.from_pretrained(pretrained_path,local_files_only=True)
        models['model']=transformers.AutoModelForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd_ss3') if fromhub else transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_path,local_files_only=True)
    elif model_name == 'prot_bert_bfd_membrane':
        models['tokenizer']=transformers.AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd_membrane') if fromhub else transformers.AutoTokenizer.from_pretrained(pretrained_path,local_files_only=True)
        models['model']=transformers.AutoModelForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd_membrane') if fromhub else transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_path,local_files_only=True)
    elif model_name == 'prot_bert_bfd_localization':
        models['tokenizer']=transformers.AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd_localization') if fromhub else transformers.AutoTokenizer.from_pretrained(pretrained_path,local_files_only=True)
        models['model']=transformers.AutoModelForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd_localization') if fromhub else transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_path,local_files_only=True)


    elif model_name == 'prot_electra_generator_bfd':
        models['tokenizer']=transformers.AutoTokenizer.from_pretrained('Rostlab/prot_electra_generator_bfd') if fromhub else transformers.AutoTokenizer.from_pretrained(pretrained_path,local_files_only=True)
        models['model'] =transformers.AutoModelForMaskedLM.from_pretrained('Rostlab/prot_electra_generator_bfd') if fromhub else transformers.AutoModelForMaskedLM.from_pretrained(pretrained_path,local_files_only=True)
    elif model_name == 'prot_electra_discrimator_bfd':
        models['tokenizer']=transformers.AutoTokenizer.from_pretrained('Rostlab/prot_electra_discrimator_bfd') if fromhub else transformers.AutoTokenizer.from_pretrained(pretrained_path,local_files_only=True)
        models['model']=transformers.AutoModelForPreTraining.from_pretrained('Rostlab/prot_electra_discrimator_bfd') if fromhub else transformers.AutoModelForPreTraining.from_pretrained(pretrained_path,local_files_only=True)
    elif model_name == 'prot_xlnet':
        models['tokenizer']=transformers.XLNetTokenizer.from_pretrained('Rostlab/prot_xlnet') if fromhub else transformers.XLNetTokenizer.from_pretrained(pretrained_path,local_files_only=True)
        models['model']=transformers.AutoModel.from_pretrained('Rostlab/prot_xlnet') if fromhub else transformers.AutoModel.from_pretrained(pretrained_path,local_files_only=True)
    else :
        raise ValueError("wrong model ")
    # if isinstance(models['model'],tuple):
    #     models['model']=models['model'][0]
    return models
ALL_MODELS=[
    # t5
    'prot_t5_xl_uniref50', 'prot_t5_xl_half_uniref50-enc', 'prot_t5_base_mt_uniref50',
    'prot_t5_xxl_bfd', 'prot_t5_xxl_uniref50', 'prot_t5_xl_bfd',
    # bert
    'prot_bert_bfd', 'prot_bert', 'prot_albert',
    'prot_bert_bfd_ss3', 'prot_bert_bfd_membrane', 'prot_bert_bfd_localization',
    # xlnet
    'prot_electra_generator_bfd', 'prot_electra_discrimator_bfd', 'prot_xlnet']
if __name__== '__main__':
    parser = argparse.ArgumentParser(description="make embeds")
    parser.add_argument('-fasta_path',type=str,required=True,help='')
    parser.add_argument('-model',type=str,choices=ALL_MODELS,required=True,help='')
    parser.add_argument('-embeds_type',type=str,choices=["logits","last_hidden_state","pooler_output"],default="pooler_output",help='')
    parser.add_argument('-device',type=str,choices=['cpu','gpu'],default='cpu',help='')
    parser.add_argument('-pretrained_path',type=str,default='',help='')
    parser.add_argument('-task_name',type=str,help='')
    parser.add_argument('-max_position_length',type=int,default=1024,help='')
    parser.add_argument('-batch',type=int,default=0,help='')
    parser.add_argument('-output_dir',type=str,default='../embeds',help='')
    parser.add_argument('-seq_format',type=str,default='fasta',help='')
    parser.add_argument('-save',type=bool,default=True,help='')
    parser.add_argument('-show_process',type=bool,default=True,help='')

    args = parser.parse_args()
    models=get_tokenizer_model(args.model,args.pretrained_path)
    tokenizer,model=models['tokenizer'], models['model']

    print(count_parameters(model))
    seq_encode=None
    if args.embeds_type in ["logits","last_hidden_state"]:
        def seq_encode(seq, tokenizer, model):
            input_ids=" ".join(re.sub(r"[UZOB]", "X", str(seq)))
            inputs = tokenizer(input_ids, max_length=max_position_length, padding='max_length', truncation=True,
                               return_tensors='pt')
            return model(**(inputs.to(device)), )[args.model].cpu().detach().numpy()


    if args.batch==0:
        create_embeds(args.fasta_path,tokenizer,model,
            device=args.device,
            max_position_length=args.max_position_length,
            output_dir=args.output_dir,
            task_name=args.task_name,
            model_name=args.model,
            embeds_shape=models['embeds_shape'],
            seq_format =args.seq_format,
            re_func = None, seq_encode = seq_encode,
            save = args.save, show_process = args.show_process,
                      )
    else:
        create_embeds_plus(args.fasta_path,tokenizer,model,
            device=args.device,
            max_position_length=args.max_position_length,
            batch=args.batch,
            output_dir=args.output_dir,
            task_name=args.task_name,
            model_name=args.model,
            embeds_shape=models['embeds_shape'],
            seq_format =args.seq_format,
            re_func = None, seq_encode = seq_encode,
            save = args.save, show_process = args.show_process,
                      )

