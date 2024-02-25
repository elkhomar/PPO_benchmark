#python train.py -m logging.group="Halfcheetah entropy comparison" +algorithm.ent_coef=1e-1
python train.py -m logging.group="Halfcheetah algo comparison" algorithm._target_="sb3_contrib.TQC"
