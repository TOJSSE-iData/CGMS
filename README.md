# A complete graph-based approach with multi-task learning for predicting synergistic drug combinations

## Cite

To be added...

## Envs

The code requires python 3.7+ and packages listed in the `requiements.txt`. To create environment to run the code, you can simply run:

~~~bash
pip3 install -r requirements.txt
~~~

## Fast run

To run cross-validation, using the following cmd:

~~~bash
# run on single fold with gpu, 1 for example:
python3 cv_cgms.py config.yaml --fold 1 --gpu 0 
# run on 5 fold at the same time, please pay attention to the cuda memory :) all folds will be run on the same gpu 
python3 cv_cgms.py config.yaml --gpu 0
~~~

If the cross-validation is run fold-by-fold, you can summary the metrics by running:

~~~bash
python3 cv_cgms.py config.yaml --suffix <your-suffix> --metric
~~~

To train AutoEncoder for drug features or cell line features, using the following cmd:

~~~bash
# to train drug ae:
python3 train_ae.py 0 --epoch 50 --dim 256 --lr 0.001
# to train cell ae:
python3 train_ae.py 1 --epoch 100 --dim 256 --lr 0.001
~~~

## Detailed configs

There are several settings could be configured, to run CGMS with other settings, you can create your own config file.

### Features

~~~yaml
drug_feat: output/drug_feat_ae.npy  # specify drug feature matrix in numpy ndarray format, rows are drugs and cols are features
cell_feat: output/cell_feat_ae.npy  # specify cell line feature matrix in numpy ndarray format, rows are cell line and cols are features
~~~

### Dataset

~~~yaml
synergy: data/syn4cv.csv  # specify synergy dataset in csv format
sensitivity: data/sen_syn_feat_dcnt5.csv  # specify sensitivity dataset in csv format
~~~

### dataloader

~~~yaml
dataloader:
  eval_factor: 4  # batch_size *= eval_factor when eval model
  num_workers: 4  # other params for pytorch dataloader
  prefetch_factor: 2
  pin_memory: true
~~~

### Others

The other setttings like `folds`, `hyperparams`, `gpu`, `suffix`, etc., are easy to be understand, please refer to the provided template `config.yaml` for details.

You can also remove config items from the config file and specify the config item when running with cmd line. Such config items include `fold,gpu,suffix,seed`.


## File profiles

- cgms: module of this work
    - \_\_init__.py: python module init file
    - datasets.py: PyTorch Dataset used for training
    - models.py: PyTorch implemenation of CGMS and AutoEncoder
    - utils.py: tool functions
- data: data dir, more information could be found in [Supplementary Material S1](link/to/be/added)
    - {cell/drug}2id.tsv: cell line or drug names to the index
    - raw_{cell/drug}_feat.npy: raw cell line or drug features, not encoded by AutoEncoder
    - syn4cv.csv: synergy dataset for cross-validation, in the leave-drug combination-out scenario
    - sen_syn_feat_dcnt5.csv: sensitivity dataset for cross-validation, in the leave-drug combination-out scenario
    - {syn/sen}4cv_lv_{c/d}[_{0~4}out].csv: synergy or sensitivity datasets for cross-validation, in the leave-cell line-out or the leave-drug-out scenario
- output: output dir
    - {cell/drug}_feat_ae.npy: cell line or drug features, encoded by AutoEncoder
    - cv_{suffix}: output result dir
- config*.yaml: config files for training, see later section for details
- cv_cgms.py: main file for cross-validation in the leave-drug combination-out scenario
- cv_cgms_lv_{c/d}_out.py: main file for cross-validation in the leave-cell line-out or the leave-drug-out scenario
- train_ae.py: train AutoEncoder
- requirements.txt: freezed list of required python modules

## Licence

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

![](by-nc-sa.svg)
