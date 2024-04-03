from mpi4py import MPI


from DataGen.genDataNonTemp import genNonTemp
import sys


comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
myrank = comm.Get_rank()

gen = str(sys.argv[1])
L = int(sys.argv[2])
T = int(sys.argv[3])
R = None
if gen == 'net':
    R = int(sys.argv[4])
start_dir = ''

info = R, L, T, start_dir, gen

genNonTemp(info)