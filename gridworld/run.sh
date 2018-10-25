#python main.py --n_trial=2 --max_step=500 --method='DH' --ent_known=1 --beta=50 >DH_known.txt
python main.py --n_trial=2 --max_step=500 --method='DH' --ent_known=0 --beta=50 >DH.txt

python main.py --n_trial=2 --max_step=500 --method='DO' --ent_known=1 --beta=50 >DO_known.txt
python main.py --n_trial=2 --max_step=500 --method='DO' --ent_known=0 --beta=50 >DO.txt

python main.py --n_trial=2 --max_step=500 --method='MBIE' --ent_known=1 --beta=0.02 >MBIE.txt
python main.py --n_trial=2 --max_step=500 --method='MBIE_NS' --ent_known=0 --beta=0.02 >MBIE_NS.txt



