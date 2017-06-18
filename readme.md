# Machine Learning Experiment 
## Repo pour la soutenance de machine learning

Avant de lancer des scripts il faut lancer la commande
```
python init_env.py
```
Ce qui va créer la base Sqlite qui stocke les résultats des différents executions

Quand vous lancez un script vous pouvez modifier les hyperparametres en les passant en argument
```
python mnist_momentum.py -l 0.1 -b 50
```

enfin vous pouvez visualiser les différent scripts en lancant la commande
```
tensorboard --logdir=Logs/
```