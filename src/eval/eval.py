
import argparse
import logging
import os
import pandas as pd
import numpy as np

from graph import Graph
from Parser import obo_parser, gt_parser, pred_parser, ia_parser
from evaluation import get_leafs_idx, get_roots_idx, evaluate_prediction

import random
import os.path


class Check:
    def __init__(self):
        pass
        self.i=0
    def check(self,info=''):
        print(f'check_{self.i}:{info}')
        self.i+=1

def setup_seed(seed):
    """
    set random seed
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except:
        print('no torch')

def make_ground_truth(train_path,des_dir,rs=42,portion=0.2,keep_aspect=False):
    setup_seed(rs)
    assert os.path.exists(des_dir)
    df=pd.read_csv(train_path,sep='\t')
    eval_path=des_dir+f'\eval{portion}_{rs}.tsv'
    train_path=des_dir+rf'\train{1-portion}_{rs}.tsv'

    grouped=df.groupby(['EntryID'])
    unique_ids = grouped.groups.keys()
    n=int(len(unique_ids)*portion)
    eval_ids = np.random.choice(list(unique_ids), size=n, replace=False)
    train_ids=np.setdiff1d(np.array(list(unique_ids)),eval_ids)

    train_df = pd.concat([grouped.get_group(id) for id in train_ids])
    train_df.to_csv(train_path,sep='\t',header=False,index=False)
    eval_df = pd.concat([grouped.get_group(id) for id in eval_ids])
    if not keep_aspect:
        eval_df=eval_df.drop(['aspect'],axis=1)
    eval_df.to_csv(eval_path,sep='\t',header=False,index=False)

# make_ground_truth(
#     r'C:\Users\23920\Desktop\research\demo\dataset\cafa-5-protein-function-prediction\Train\train_terms.tsv',
#     rf'C:\Users\23920\Desktop\research\demo\dataset\cafa-5-protein-function-prediction',
#     portion=0.05,
#                   )

def eval(pred_dir,gt_file,obo_file,evals=None,out_dir='.',ia=None,prop='fill',norm='cafa',th_step=0.001,max_terms=1000,save=False):
    evals_dict={
        'wf':{'type':'wf','columns':["wpr", "wrc", "wf"],
               'items':[('wf', ['wrc', 'wpr'])]},
        'all':{'type':'all','columns':["cov", "pr", "rc", "f", "wpr", "wrc", "wf", "mi", "ru", "s"],
               'items':[('f', ['rc', 'pr']), ('wf', ['wrc', 'wpr']), ('s', ['ru', 'mi'])]},
    }
    if evals is None:
        evals=evals_dict['wf']
        assert ia is not None
    else:
        evals=evals_dict[evals]
    no_orphans=False
    threads=5

    out_folder = os.path.normpath(out_dir) + "/"
    if not os.path.isdir(out_folder):
        os.mkdir(out_folder)

    # Set the logger
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    rootLogger = logging.getLogger()
    # rootLogger.setLevel(logging.DEBUG)
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler("{0}/info.log".format(out_folder))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # Parse and set information accretion (optional)
    ia_dict = None
    if ia is not None:
        ia_dict = ia_parser(ia)
    # Parse the OBO file and creates a different graph for each namespace
    ontologies = []

    for ns, terms_dict in obo_parser(obo_file).items():
        ontologies.append(Graph(ns, terms_dict, ia_dict, not no_orphans))
    #     logging.info("Ontology: {}, roots {}, leaves {}".format(ns, len(get_roots_idx(ontologies[-1].dag)), len(get_leafs_idx(ontologies[-1].dag))))

    # Set prediction files
    pred_folder = os.path.normpath(pred_dir) + "/"  # add the tailing "/"
    pred_files = []


    for root, dirs, files in os.walk(pred_folder):
        for file in files:
            pred_files.append(os.path.join(root, file))
    # logging.debug("Prediction paths {}".format(pred_files))

    # Parse ground truth file
    gt = gt_parser(gt_file, ontologies)

    # Tau array, used to compute metrics at different score thresholds
    tau_arr = np.arange(0.01, 1, th_step)

    # Parse prediction files and perform evaluation
    dfs = []



    for file_name in pred_files:
        prediction = pred_parser(file_name, ontologies, gt, prop, max_terms)
        df_pred = evaluate_prediction(prediction, gt, ontologies, tau_arr, norm, threads)
        df_pred['filename'] = file_name.replace(pred_folder, '').replace('/', '_')
        dfs.append(df_pred)
        # logging.info("Prediction: {}, evaluated".format(file_name))
    df = pd.concat(dfs)

    # Save the dataframe
    df = df[df['cov'] > 0].reset_index(drop=True)
    df.set_index(['filename', 'ns', 'tau'], inplace=True)
    df.to_csv('{}/evaluation_all.tsv'.format(out_folder),
              # columns=["cov", "pr", "rc", "f", "wpr", "wrc", "wf", "mi", "ru", "s"],
              columns=evals['columns'],
              float_format="%.5f", sep="\t")
    # Calculate harmonic mean across namespaces for each evaluation metric
    # for metric, cols in [('f', ['rc', 'pr']), ('wf', ['wrc', 'wpr']), ('s', ['ru', 'mi'])]:

    df_wf=None
    for metric, cols in evals['items']:
        index_best = df.groupby(level=['filename', 'ns'])[metric].idxmax() if metric in ['f', 'wf'] else df.groupby(['filename', 'ns'])[metric].idxmin()
        df_best = df.loc[index_best]
        df_best['max_cov'] = df.reset_index('tau').loc[[ele[:-1] for ele in index_best]].groupby(level=['filename', 'ns'])['cov'].max()
        df_wf=df_best
        if save:
            df_best.to_csv('{}/evaluation_best_{}.tsv'.format(out_folder, metric),
                           # columns=["cov", "pr", "rc", "f", "wpr", "wrc", "wf", "mi", "ru", "s", "max_cov"],
                           columns=evals['columns']+['max_cov'],
                           float_format="%.5f", sep="\t")
    if evals['type']=='wf':
        return df_wf
    else:
        return None

if __name__ == '__main__':
    # freeze_support()
    df=eval(
        pred_dir=r'/prediction',
         # gt_file=r'C:\Users\23920\Desktop\research\demo\output\ground_truth_test.tsv',
         gt_file=r'C:\Users\23920\Desktop\research\demo\dataset\cafa-5-protein-function-prediction\eval0.05_42.tsv',
         obo_file=r'/dataset/cafa-5-protein-function-prediction/Train/go-basic.obo',
         ia=r'C:\Users\23920\Desktop\research\demo\dataset\cafa-5-protein-function-prediction\IA.txt',
         evals='wf'
         )
    print(df['wf'].values)



