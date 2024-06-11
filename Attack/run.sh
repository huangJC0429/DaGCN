
--cora--
python main.py --dataset cora --epochs 2000 --hidden 8 --concat 1  --lr 0.01 --dropout 0.6 --lam 1.2 --I True --ptb_rate 0.0

--citeseer--
python main.py --dataset citeseer --epochs 2000 --hidden 8 --concat 1  --lr 0.01 --dropout 0.6 --lam 0.0 --ptb_rate 0.0

--pubmed--
python main.py --dataset pubmed --epochs 2000 --hidden 4 --concat 1 --lr 0.02 --dropout 0.5 --tem 0.6 --lam 0.0 --ptb_rate 0.0