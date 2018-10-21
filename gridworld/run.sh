python main.py --n_trial=100 --max_step=500 --method='DH' --ent_known=1 --beta=5 >DH_known.txt
python main.py --n_trial=100 --max_step=500 --method='DH' --ent_known=0 --beta=5 >DH.txt

python main.py --n_trial=100 --max_step=500 --method='DO' --ent_known=1 --beta=5 >DO_known.txt
python main.py --n_trial=100 --max_step=500 --method='DO' --ent_known=0 --beta=5 >DO.txt

python main.py --n_trial=100 --max_step=500 --method='MBIE' --ent_known=1 --beta=0.2 >MBIE.txt
python main.py --n_trial=100 --max_step=500 --method='MBIE_NS' --ent_known=0 --beta=0.2 >MBIE.txt



