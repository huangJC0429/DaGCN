python dagcn.py --dataset cora --epochs 2000 --hidden 8 --concat 1  --lr 0.01 --dropout 0.6 --lam 1.3 --I True
python dagcn.py --dataset citeseer --epochs 2000 --hidden 8 --concat 1  --lr 0.01 --dropout 0.6 --lam 1.1 --I True
python dagcn.py --dataset pubmed --epochs 2000 --hidden 4 --concat 1 --lr 0.02 --dropout 0.5 --tem 0.6 --lam 0.1


# all datasets of amazon share the same parameters.

python dagcn-amazon.py --dataset Computers --epochs 500 --hidden 64 --concat 1 --lr 0.01 --tem 0.4 --dropout 0.5 --lam 1.3
python dagcn-amazon.py --dataset CS --epochs 500 --hidden 128 --concat 1 --lr 0.01 --tem 0.5 --dropout 0.6 --lam 1.3
python dagcn-amazon.py --dataset Photo --epochs 500 --hidden 64 --concat 1 --lr 0.01 --tem 0.5 --dropout 0.6 --lam 0.1
python dagcn-amazon.py --dataset Physics --epochs 500 --hidden 128 --concat 1 --lr 0.01 --tem 0.5 --dropout 0.6 --lam 1.3