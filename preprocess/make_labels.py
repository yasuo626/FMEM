
import argparse
import numpy as np
import pandas as pd
import transformers
from Bio import SeqIO

def generate_labels(path_or_df,label,k,by=None,method='nunique',weight_csv=None,show=False):
    """
    get top-k labels for label col by selecting from each by_groups which is classified by [by].
    :param method: [nunique,freq] nunique:weigh by each group unique class nums,freq:weigh by each group sample frequency
    :param path_or_df: (DF/str) dataframe or path
    :param label: (str)the name of label col
    :param k:(int) the num of select labels
    :param by: (list) the cols list which use to groupby
    :param show: (bool) show the statistics table?
    :return: (numpy.array)[k,] the label values which selected
    :return:
    """
    if isinstance(path_or_df,str):
        df=pd.read_csv(path_or_df,sep='\t')
    else:
        df=path_or_df
    if by is None and method!='weighted':
        return df.groupby([label]).count().iloc[:,0].reset_index(name='freq').sort_values(by='freq',ascending=False).reset_index(drop=True).loc[:k-1,label].values
    elif by is None and method=='weighted':
        weight=pd.read_csv(weight_csv,sep='\t',header=None)
        weight.columns=['term','weight']
        df=df.groupby([label]).count().iloc[:, 0].reset_index(name='times')
        df=df.merge(weight,on='term')
        df['importance'] = df['times'] * df['weight']
        sum_importance = df['importance'].sum()
        df['norm_importance'] = df['importance'] / sum_importance
        df = df.sort_values('norm_importance', ascending=False).reset_index(drop=True)
        return df.loc[:k-1,label].values
    else:
        pass

    g = df.groupby(by).count()
    assert len(g) <= k and k < len(df.groupby(label))
    assert method in ['nunique','freq']
    uniquen = []
    for x in df.groupby(by).groups.items():
        uniquen.append(len(pd.unique(df.loc[x[1], label])))
    g['nunique'] = uniquen
    if method=='freq':
        s = g[label].sum()
        g['freq'] = g[label].map(lambda x: x / s)
        g['nassign'] = g['freq'].map(lambda x: math.floor(x * k))
        g = g.sort_values(by=label, ascending=False)
    elif nunique:
        s = g['nunique'].sum()
        g['freq'] = g['nunique'].map(lambda x: x / s)
        g['nassign'] = g['freq'].map(lambda x: math.floor(x * k))
        g = g.sort_values(by='nunique', ascending=False)
    else:
        pass
    lest = k - g['nassign'].sum()
    y = [lest]
    def assign_lest(x, y):
        if y[0] <= 0:
            return x
        allow = x['nunique'] - x['nassign']
        if y[0] <= allow:
            x['nassign'] += y[0]
            y[0] = 0
            return x
        x['nassign'] += allow
        y[0] -= allow
        return x
    g = g.apply(assign_lest, axis=1, args=(y,))
    if show:
        print(g)
    labels = np.empty([k],dtype=df.dtypes[label])
    last = 0
    for l in df.groupby(by).groups.items():
        num = int(g.loc[l[0], 'nassign'])
        # print(df[df.index.isin(l[1])].groupby(label).count().iloc[:, 0].reset_index(name='freq').sort_values(by='freq',ascending=False))
        labels[last:last + num] =df[df.index.isin(l[1])].groupby(label).count().iloc[:, 0]\
         .reset_index(name='freq').sort_values(by='freq',ascending=False)[label].values[:num]
        last += num
    return np.array(list(set(labels)))


def create_labels(term_or_path,ids_or_path,num_labels,output_dir,task_name,label='term',by=None,method='nunique',weight_file=None,save=True,show=False):
    """
        create labels for the terms
    :param term_or_path: str or dataframe
    :param ids_or_path: str or dataframe
    :param num_labels: list or int [100,200 ...]
    :param output_dir: str
    :param task_name: str
    :param label: label col name
    :param by: group by aspect to select labels
    :param method: nunique ,freq,weighted ,see generate_labels() for detail
    :param weight_file: the file give all the label's weight with out header,col1 is label ,col2 is weight
    :param save: Bool
    :param show: Bool show the labels assigned freq
    :return: None
    """
    ids=ids_or_path
    term_df=term_or_path
    if isinstance(term_or_path,str):
        term_df=pd.read_csv(term_or_path,sep='\t')
    if isinstance(ids_or_path,str):
        ids=np.load(ids_or_path)

    if not  'EntryID' in term_df.columns and 'term' in term_df.columns:
        raise ValueError("the data must contain column EntryID as protein name col and term as GO label col")

    if method=='weighted':
        assert weight_file is not None

    term_labels=generate_labels(term_df,label,num_labels,by=by,method=method,weight_csv=weight_file,show=show)

    terms_used = term_df[(term_df.term.isin(term_labels)) & (term_df.EntryID.isin(ids))]
    id_labels = terms_used.groupby('EntryID')['term'].apply(list).to_dict()
    term2idx = {label: i for i, label in enumerate(term_labels)}

    labels_matrix = np.zeros((len(ids), len(term_labels)))
    for index,i in enumerate(ids):
        # add label with [] which id was droped.
        if i not in id_labels.keys():
            id_labels[i]=[]
        idxs=[term2idx[term] for term in term_labels if term in id_labels[i]]
        labels_matrix[index, idxs] = 1
    if save:
        np.save(output_dir+f'/{task_name}_{num_labels}_{method}_label_names.npy',term_labels)
        np.save(output_dir+f'/{task_name}_{num_labels}_{method}_labels.npy',labels_matrix)
    return labels_matrix



if __name__== '__main__':
    parser = argparse.ArgumentParser(description="make embeds")
    parser.add_argument('-task_name',type=str,required=True,help='')
    parser.add_argument('-term_path',type=str,required=True,help='')
    parser.add_argument('-ids_path',type=str,required=True,help='')
    parser.add_argument('-num_labels',type=int,nargs='+',default=[300,500],help='allow multi nums')
    parser.add_argument('-output_dir',type=str,default='../labels',help='')
    parser.add_argument('-label',type=str,default='term',help='')
    parser.add_argument('-by',type=str,nargs='+',default=None,help='')
    parser.add_argument('-method',type=str,default='nunique',help='')
    parser.add_argument('-weight_csv',type=str,default=None,help='')
    parser.add_argument('-save',type=bool,default=True,help='')
    parser.add_argument('-show',type=bool,default=True,help='')

    args = parser.parse_args()


    for n in args.num_labels:
        create_labels(term_or_path=args.term_path,ids_or_path=args.ids_path,
                      num_labels=n,output_dir=args.output_dir,
                      task_name=args.task_name,
                      label=args.label,by=args.by,method=args.method,weight_file=args.weight_csv,
                      save=args.save,show=args.show
                      )

