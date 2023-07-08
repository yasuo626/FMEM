# Fusional Multi Embed Model(FMEM) :Optimization of Protein Language Models on Gene Ontology Prediction Tasks by Fusing Embeddings

### ABSTRACT
我们通过组合嵌入对下游任务进行优化，与普通模型相比较达到了高效的性能提升
([paper link](http://res.aidroid.top/share/paper/vgg.pdf))
优点：
1.准确率高
2.模型参数少
3.推理快
4.扩展性高，适用性广泛


### How to Use
1. download data from uniprot(train.fasta,train.tsv)
    <br>the fasta store the proterin id and seqs
    <br>the tsv store protein id and term labels 
2. use preprocess modules to create embeds and labels
   * download model
   <br> before you make embeds,you have to download pretrained model first.
   or you will atuodownload the model in the hub.you can put model to <a>"/pretrained"</a>
   * make embeds
      ```
      python make_embeds.py \
        -task_name protbert \
        -fasta_path C:\\Users\\23920\\Desktop\\research\\demo\\dataset\\cafa-5-protein-function-prediction\\Train\\train_sequences.fasta \
        -model protbert \
        -pretrained_path C:\\Users\\23920\\AppData\\Local\\huggingface\\protbert \
        -output_dir ../output
      ```
   * make labels
       ```
       python make_labels.py \
           -task_name make_labels\
           -term_path $demo_dir/dataset/cafa-5-protein-function-prediction/Train/train_terms.tsv \
           -ids_path $demo_dir/embeds/ProtBert/train_ids.npy \
           -num_labels 300\
           -label term \
           -method freq \
           -weight_csv $demo_dir/dataset/cafa-5-protein-function-prediction/IA.txt \
           -output_dir $demo_dir/labels \
           -save True
       ```
3. select model<br>

    - single embeds input model
      - SingleEmbedsPerceptron:fullconnect for single embed
    - multi embeds input model
      - AttentionModel:fullconnect for multi embed
      - Transformer_AttentionModel:transformer with multi embed
      - Fusion_AttentionModel: fusional multi embed model FMEM

4. training

set :
demo_dir='/root/prot'
demo_dir='/C/Users/23920/Desktop/research/demo'

single-embeds:
```
python train.py \
    -project_name cafa_task \
    -ids_path   $demo_dir/embeds/ProtBert/train_ids.npy \
    -embeds_path $demo_dir/embeds/ProtBert/train_embeddings.npy \
    -labels_path $demo_dir/labels/make_labels_500_weighted_labels.npy \
    -num_labels 500 \
    -metrics f1score \
    -accelerator 'gpu' \
    -logger wandb \
    -output_dir $demo_dir/output \
    -save_path $demo_dir/output/model \
    -lr 2e-3 \
    -batch_size 128 \
    -total_epoch 15
```

multi-embeds:
```
    python train.py \
    -project_name cafa_task \
    -use_model am \
    -ids_path   $demo_dir/embeds/ProtBert/train_ids.npy \
    -num_labels 500 \
    -metrics f1score accuracy \
    -accelerator 'gpu' \
    -logger none \
    -output_dir $demo_dir/output \
    -save_path $demo_dir/output/model \
    -labels_path $demo_dir/labels/make_labels_500_weighted_labels.npy \
    -embeds_path_list \
        $demo_dir/embeds/ProtBert/train_embeddings.npy $demo_dir/embeds/T5/train_embeddings.npy $demo_dir/embeds/EMS/train_embeds.npy \
    -lr 2e-3 \
    -batch_size 128 \
    -total_epoch 20
```


```
    python train.py \
    -project_name protein \
    -use_model ta \
    -ids_path   $demo_dir/embeds/ProtBert/train_ids.npy \
    -num_labels 300 \
    -metrics f1score \
    -accelerator 'gpu' \
    -logger wandb \
    -output_dir $demo_dir/output \
    -save_path $demo_dir/output/model \
    -labels_path $demo_dir/labels/make_labels_300_True_labels.npy \
    -embeds_path_list \
        $demo_dir/embeds/ProtBert/train_embeddings.npy $demo_dir/embeds/T5/train_embeddings.npy $demo_dir/embeds/EMS/train_embeds.npy \
    -lr 1e-3 \
    -batch_size 64 \
    -total_epoch 20
```
```
    python train.py \
    -project_name protein \
    -use_model fa \
    -ids_path   $demo_dir/embeds/ProtBert/train_ids.npy \
    -num_labels 300 \
    -metrics f1score \
    -accelerator 'gpu' \
    -logger wandb \
    -output_dir $demo_dir/output \
    -save_path $demo_dir/output/model \
    -labels_path $demo_dir/labels/make_labels_300_True_labels.npy \
    -embeds_path_list \
        $demo_dir/embeds/ProtBert/train_embeddings.npy $demo_dir/embeds/T5/train_embeddings.npy $demo_dir/embeds/EMS/train_embeds.npy\
    -lr 1e-3 \
    -batch_size 64 \
    -total_epoch 20

```


7. make prediction
```
 python predict.py \
    -task_name predict \
    -use_model am \
    -model_conf $demo_dir/conf/protein_am.ini \
    -ckpt_path $demo_dir/output/cafa_task/lightning_logs/version_0/checkpoints/epoch=17-step=16002.ckpt \
    -ids_path $demo_dir/embeds/ProtBert/test_ids.npy \
    -label_names_path $demo_dir/labels/make_labels_500_weighted_label_names.npy \
    -embeds_path $demo_dir/embeds/ProtBert/test_embeddings.npy \
    -embeds_path_list \
            $demo_dir/embeds/ProtBert/train_embeddings.npy $demo_dir/embeds/T5/train_embeddings.npy $demo_dir/embeds/EMS/train_embeds.npy \
    -num_labels 500 \
    -output_dir $demo_dir/output \
    -max_embeds_length 1024 \
    -accelerator gpu \
    -confidence_threshold 0.1
```

 python predict.py \
    -task_name predict \
    -use_model sep \
    -model_conf $demo_dir/conf/protein_sep.ini \
    -ckpt_path $demo_dir/output/logs/cafa_task/cszgd84u/checkpoints/epoch=14-step=13335.ckpt \
    -ids_path $demo_dir/embeds/ProtBert/test_ids.npy \
    -label_names_path $demo_dir/labels/make_labels_500_weighted_label_names.npy \
    -embeds_path $demo_dir/embeds/ProtBert/test_embeddings.npy \
    -embeds_path_list \
            $demo_dir/embeds/ProtBert/train_embeddings.npy $demo_dir/embeds/T5/train_embeddings.npy $demo_dir/embeds/EMS/train_embeds.npy \
    -num_labels 500 \
    -output_dir $demo_dir/output \
    -max_embeds_length 1024 \
    -accelerator gpu \
    -confidence_threshold 0.0

