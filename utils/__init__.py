from utils import *

"""
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
demo_dir='/C/Users/23920/Desktop/research/demo'
 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=
 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 
 
 python make_embeds.py \
    -task_name prot_t5_xl_half_uniref50-enc \
    -fasta_path $demo_dir/dataset/cafa-5-protein-function-prediction/Train/train_sequences.fasta \
    -model prot_t5_xl_half_uniref50-enc \
    -device cpu \
    -pretrained_path $demo_dir/pretrained/models--Rostlab--prot_t5_xl_half_uniref50-enc \
    -output_dir ../embeds


prot_bert_bfd
 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python 
 
 python make_embeds.py \
    -task_name train \
    -fasta_path $demo_dir/dataset/cafa-5-protein-function-prediction/Train/train_sequences.fasta \
    -model prot_bert_bfd \
    -device cpu \
    -pretrained_path $demo_dir/pretrained/models--Rostlab--prot_bert_bfd \
    -output_dir ../embeds/prot_bert_bfd
    
    
     python make_embeds.py \
    -task_name train \
    -fasta_path $demo_dir/dataset/cafa-5-protein-function-prediction/Train/train_sequences.fasta \
    -model prot_bert \
    -device cpu \
    -pretrained_path $demo_dir/pretrained/ProtBert \
    -output_dir ../embeds/prot_bert

     python make_embeds.py \
    -task_name train \
    -fasta_path $demo_dir/dataset/cafa-5-protein-function-prediction/Train/train_sequences.fasta \
    -model prot_albert \
    -device cpu \
    -pretrained_path $demo_dir/pretrained/models--Rostlab--prot_albert \
    -output_dir ../embeds



"""
