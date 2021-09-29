echo " starting main simulation"
mpiexec -n 4 python main.py $1
mpiexec -n 4 python merge.py ../experiments/$1/data/
