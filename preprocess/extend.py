"""
get extend train data
1.
    uniprot500k.tsv(Entry	Length	Gene Ontology IDs)
    uniprot500k_full.tsv()
2.
    uniprot500k.fasta(id,seq)
"""
import numpy as np
import pandas as pd

import re

import tqdm
from Bio import SeqIO
from quickpanda import file_operate,preprocess,ploting,utils


def get_GO_list(x,pattern,s,e):
    if pattern is None:
        # pattern='\[GO:[0-9]*\];'
        pattern='GO:[0-9]{6,8}'
    return [x[s:e] for x in re.findall(pattern,x)]
class Extender:
    def __init__(self,output=''):
        self.out_dir=output
    def config(self):
        self.out_dir=r'C:/Users/23920/Desktop/research/demo/dataset/extend'

        self.extend_terms=self.out_dir+'/extend_terms.tsv'
        self.extend_fasta=r'C:\Users\23920\Desktop\research\demo\dataset\extend\extend_sequences.fasta'
        self.train_terms=r'C:\Users\23920\Desktop\research\demo\dataset\cafa-5-protein-function-prediction\Train\train_terms.tsv'
        self.train_fasta=r'C:\Users\23920\Desktop\research\demo\dataset\cafa-5-protein-function-prediction\Train\train_sequences.fasta'
        self.total_terms=r'C:\Users\23920\Desktop\research\demo\dataset\extend\total_terms.tsv'
        self.total_fasta=r'C:\Users\23920\Desktop\research\demo\dataset\extend\total_sequences.fasta'

        self.ia_path=r'C:\Users\23920\Desktop\research\demo\dataset\cafa-5-protein-function-prediction\IA.txt'
        self.uniprot_min =r'C:\Users\23920\Desktop\research\demo\dataset\extend\uniprot500k_min.tsv'
        self.uniprot_full =r'C:\Users\23920\Desktop\research\demo\dataset\extend\uniprot500k_full.tsv'
        self.uniprot_nseq =r'C:\Users\23920\Desktop\research\demo\dataset\extend\uniprot500k_full_nseq.tsv'
        self.uniprot_clean =r'C:\Users\23920\Desktop\research\demo\dataset\extend\uniprot500k_clean_min.tsv'
        self.uniprot_fasta=r'C:\Users\23920\Desktop\research\demo\dataset\extend\uniprot500k.fasta'
        self.drop_prot=r'C:\Users\23920\Desktop\research\demo\dataset\extend\drop_prot.npy'
    def clean_extend_terms(self,df,use_cols=None,pattern=None,s=0,e=None):
        """
        :param uniprot_tsv:
        :param use_cols:[entry id ,gos,aspect]
        :return: drop(drop prot ids),clean_df(E)
        """
        if use_cols is None:
            use_cols = ['EntryID', 'term', 'aspect']
        fop=file_operate.FileOperator()
        fop.read_df('1',df)
        pre=preprocess.PreProcessor(fop)
        drop=fop.dfs['1'][use_cols[0]].values[pre.get_na_idx()['1']]
        pre.dropna()
        df=fop.dfs['1'].loc[:,use_cols]
        allow_terms=np.unique(pd.read_csv(self.ia_path,sep='\t',header=None).iloc[:,0].values)

        # get terms list
        df[use_cols[1]] = df.loc[:, use_cols[1]].map(lambda x:get_GO_list(x,pattern,s,e))
        df=df.explode('term')
        out_terms=self.get_out_terms(np.unique(df['term']),allow_terms)
        out_df=df[df['term'].isin(out_terms)]
        out_prot=np.array(list(out_df.groupby(['EntryID']).groups.keys()))
        drop=np.concatenate([drop,out_prot],axis=0)

        clean_df=df[~df['EntryID'].isin(out_prot)]
        # clean_df.to_csv(self.out_dir+'/extend_terms.tsv',sep='\t',header=True,index=False)
        return drop,clean_df

    def get_out_terms(self,allterms,allow_terms):
        return allterms[np.argwhere(np.isin(allterms, allow_terms) == False)].reshape(-1)

    def make_extend_terms(self,tsv_path,namefunc):
        fop = file_operate.FileOperator()
        fop.read_file('1', tsv_path, sep='\t')
        fop.show_files_desc()
        fop.read_df('2',fop.dfs['1'].loc[:,['Entry','GeneOntology(biologicalprocess)']])
        fop.read_df('3',fop.dfs['1'].loc[:,['Entry','GeneOntology(cellularcomponent)']])
        fop.read_df('4',fop.dfs['1'].loc[:,['Entry','GeneOntology(molecularfunction)']])
        fop.dfs['2']['aspect']='BPO'
        fop.dfs['3']['aspect']='CCO'
        fop.dfs['4']['aspect']='MFO'
        fop.dfs['2'].columns=['EntryID','term','aspect']
        fop.dfs['3'].columns=['EntryID','term','aspect']
        fop.dfs['4'].columns=['EntryID','term','aspect']
        dfs=[]
        drops=[]
        for fid in ['2','3','4']:
            drop,clean_df=self.clean_extend_terms(fop.dfs[fid])
            dfs.append(clean_df)
            drops.append(drop)
        df=pd.concat(dfs,axis=0)
        df.to_csv(self.extend_terms,sep='\t',header=True,index=False)

        protids = np.array(list(df.groupby(['EntryID']).groups.keys()))
        self.clean_fasta(self.uniprot_fasta,protids,des_path=self.extend_fasta,namefunc=namefunc)

    def clean_fasta(self,path,tids,namefunc,des_path='',save=True):
        """
        :param path: fasta file path, should include name param
        :param tids: allow ids
        :param namefunc: the func to get valid name  such:lambda s: s.split('|')[0]
        :param des_path:where to save fasta file
        :param save:save option
        :return:the fasta seqs which ids are in the tids
        """
        seqs = SeqIO.parse(path, 'fasta')
        names = []
        for s in tqdm.tqdm(seqs):
            names.append(namefunc(s))

        ids = np.argwhere(np.isin(np.array(names), tids)).reshape(-1)
        seqs = SeqIO.parse(path, 'fasta')
        clean = []
        cur = -1
        for i, idx in tqdm.tqdm(enumerate(ids)):
            iter_times = ids[i] - cur
            for n in range(iter_times - 1):
                next(seqs)
            cur = cur + iter_times
            clean.append(next(seqs))
        if save and des_path!='':
            SeqIO.write(clean,des_path,'fasta')
        return clean
    def concat(self,items):
        fop=file_operate.FileOperator()
        fastas=[]
        funcs=[]
        for i,item in enumerate(items):
            fop.read_file(f'{i}',item[0])
            fastas.append(item[1])
            funcs.append(item[2])
        df=fop.concat(axis=0).drop_duplicates()

        tids=np.array(list(df.groupby(['EntryID']).groups.keys()))
        final=[]
        for i,fasta in enumerate(fastas):
            clean = self.clean_fasta(fasta, tids,namefunc=funcs[i], save=False)
            final.extend(clean)
            print(f'get seqs:{len(clean)},total:{len(final)}')
        df.to_csv(self.total_terms,sep='\t',header=True,index=False)
        SeqIO.write(final,self.total_fasta,'fasta')
    def autoprocess(self):
        self.make_extend_terms(self.uniprot_nseq,namefunc=lambda s:s.id.split('|')[1])
        self.concat([(self.train_terms,self.train_fasta,lambda s: s.name),(self.extend_terms,self.extend_fasta,lambda s:s.id.split('|')[1])])


e=Extender()
e.config()
e.autoprocess()














