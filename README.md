# A complete graph-based approach with multi-task learning for predicting synergistic drug combinations

## Envs

- python: 3.7+
- torch: 1.8.2
- dgl: 0.7.2
- scikit-learn: 0.23.2
- numpy: 1.18.5
- pandas: 1.4.0

## Run

To train AutoEncoder, using the following cmd:

~~~bash
# to train drug ae:
python3 train_ae.py 0 --epoch 50 --dim 256 --lr 0.001
# to train cell ae:
python3 train_ae.py 1 --epoch 100 --dim 256 --lr 0.001
~~~

To run cross-validation, using the following cmd:

~~~bash
# run on single fold with gpu, 1 for example:
python3 cv_cgm.py config.yaml --fold 1 --gpu 0 
# run on 5 fold:
python3 cv_cgm.py config.yaml --gpu 0
~~~

## Cite

To be added...