# Runs an experiment with different seeds
for i in $(seq 1 10);
do
    python test_env.py seed=i
done
