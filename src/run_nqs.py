import quantax as qtx
from jax import numpy as jnp
import jax
#jax.distributed.initialize()
from quantax.operator import create_u, create_d, annihilate_u, annihilate_d
from matplotlib import pyplot as plt
from quantax.symmetry import Identity, Trans2D, C4v, SpinInverse, Translation, Trans3D, Trans2D, Flip, Rotation
from quantax.model.fermion_mf import _get_pfaffian_indices
import equinox as eqx
from quantax.operator import *
from observables import *
import argparse
import time

qtx.set_default_dtype(jnp.float64)

parser = argparse.ArgumentParser()
parser.add_argument(  "-l",  "--length"         , type=int   , default = 4      , help="length of physical system")
parser.add_argument(  "-w",  "--width"          , type=int   , default = 4      , help="width of physical system")
parser.add_argument(  "-Np", "--Np"             , type=int   , default = 32     , help="Number of particles")
parser.add_argument(  "-sl", "--sublattice"     , type=int, nargs=3, default = [2,2,2], help="list that specifies the sublattice")
parser.add_argument(  "-UMF",  "--UMF"          , type=float , default = 3.     , help="On-site repulsion strength for MF optimization") 
parser.add_argument(  "-J",  "--J"              , type=float , default = 0.5    , help="Spin exchange")
parser.add_argument(  "-Jperp","--Jperp"        , type=float , default = 0.5    , help="Spin exchange")
parser.add_argument(  "-tperp",  "--tperp"      , type=float , default = 0.     , help="Hopping between layers")
parser.add_argument(  "-p",  "--pairing"        , type=float , default = 0.     , help="Strength of pairing term")
parser.add_argument(  "-c",  "--c"              , type=float , default = 0.1    , help="Prefactor of (N-Ntarget)**2")
parser.add_argument(  "-stepsMF", "--stepsMF"   , type=int   , default = 1000   , help="Number of optimization steps for MF optimization")
parser.add_argument(  "-steps", "--steps"       , type=int   , default = 1000   , help="Number of optimization steps")
parser.add_argument(  "-layers", "--layers"     , type=int   , default = 4      , help="Number of network layers")
parser.add_argument(  "-features", "--features" , type=int   , default = 24	 , help="Number of features")
parser.add_argument(  "-nhid", "--nhid"         , type=int   , default = 4	, help="Number of hidden fermions")
parser.add_argument(  "-nsamples", "--nsamples" , type=int   , default = 1000   , help="Number of samples")
parser.add_argument(  "-rtol",  "--rtol"        , type=float , default = 1e-12  , help="Tolerance for SR step")
parser.add_argument(  "-lr",  "--lr"            , type=float , default = 0.01   , help="Learning rate")
parser.add_argument(  "-loadMF", "--loadMF"     , type=int   , default = 1	, help="if 1: loads MF, 0: doesnt load MF")
parser.add_argument(  "-load", "--load"         , type=int   , default = 0	, help="if 1: loads from previous runs, 0: runs the optimization")
parser.add_argument(  "-det", "--det"           , type=int   , default = 0	, help="if 1: determinant, 0: Pfaffian")

args   = parser.parse_args()
L1         = args.length
L2         = args.width
Ntarget    = args.Np
sublattice = (args.sublattice[0],args.sublattice[1],args.sublattice[2])

#MF optimization parameter
uMF        = args.UMF
pairing    = args.pairing
c          = args.c
nstepsMF   = args.stepsMF


#NQS optimization parameter
J            = args.J
Jperp        = args.Jperp
tperp        = args.tperp
layers       = args.layers
features     = args.features
nhid         = args.nhid
nsamples     = args.nsamples
nsteps       = args.steps
lr           = args.lr
rtol         = args.rtol
if sublattice[1]*sublattice[2]>20:
  max_parallel = (50,25)
elif sublattice[1]*sublattice[2]<=20 and sublattice[1]*sublattice[2]>4:
  max_parallel = (50,50)
else:
  max_parallel = (100,100)

run_optimization = False

if args.loadMF==1:
  if args.det==0:
    filename = f"Lx{L1}_Ly{L2}_Nt{Ntarget}_J{J}_Jperp{Jperp}_tperp{tperp}_sublattice{sublattice[0]}_{sublattice[1]}_{sublattice[2]}_MF{uMF}_{pairing}_{c}_{nstepsMF}_layers{layers}_features{features}_nhid{nhid}_nsamples{nsamples}_nsteps{nsteps}_lr{lr}_rtol{rtol}"
  else:
    filename = f"Lx{L1}_Ly{L2}_Nt{Ntarget}_J{J}_Jperp{Jperp}_tperp{tperp}_sublattice{sublattice[0]}_{sublattice[1]}_{sublattice[2]}_detMF{uMF}_{nstepsMF}_layers{layers}_features{features}_nhid{nhid}_nsamples{nsamples}_nsteps{nsteps}_lr{lr}_rtol{rtol}"
else:
  filename = f"Lx{L1}_Ly{L2}_Nt{Ntarget}_J{J}_Jperp{Jperp}_tperp{tperp}_sublattice{sublattice[0]}_{sublattice[1]}_{sublattice[2]}_layers{layers}_features{features}_nhid{nhid}_nsamples{nsamples}_nsteps{nsteps}_lr{lr}_rtol{rtol}"


lattice = qtx.sites.Grid((2,L1, L2), Nparticle=((Ntarget+1)//2,Ntarget//2), boundary=(1,1,1), is_fermion=True, double_occ=False)

# Define Hamiltonian
H = BilayertJ(lattice,J=J, Jperp=Jperp, tperp=tperp)

# define network, variational state, sampler and optimizer
pg_symm = Rotation(angle=jnp.pi/2, axes=(-2,-1), sector=0) + Flip(1,sector=0) + Flip(2,sector=0)
net     = qtx.model.ResSumGconv(layers,features,pg_symm=pg_symm,spin_parity=1,dtype=jnp.float32,final_activation=lambda x: x, project=False)
model   = qtx.model.BackflowPfaffian(pairing_net=net,Nhidden=nhid,sublattice=sublattice,trans_symm = Trans3D())

_, _, _, mf_params = _get_pfaffian_indices(sublattice, 2*lattice.N)
total_params = sum(x.size for x in jax.tree_util.tree_leaves(net) if isinstance(x, jnp.ndarray)) + mf_params
print(f"Total parameters: {total_params}")


if args.loadMF==1 and args.load==0:
  if args.det==0:
    F = jnp.load(f'MF_results/orbs_Lx{L1}_Ly{L2}_Nt{Ntarget}_U{uMF}_pairing{pairing}_c{c}_nsteps{nstepsMF}_sublattice{sublattice[0]}_{sublattice[1]}_{sublattice[2]}.npy')
  else:
    U = jnp.load(f'MF_results/det_orbs_Lx{L1}_Ly{L2}_Nt{Ntarget}_U{uMF}_nsteps{nstepsMF}.npy')
    U1, U2 = U[:,::2], U[:, 1::2]
    F = U1 @ U2.T - U2 @ U1.T
    F = F.ravel()
  model = eqx.tree_at(lambda tree: tree.layers[-1].F, model, F)
  #this is only to evaluate the bare MF energy
  MFmodel = qtx.model.Pfaffian(dtype=jnp.float64, sublattice=sublattice)
  MFmodel = eqx.tree_at(lambda MFmodel: MFmodel.F, MFmodel, F)
  MFstate = qtx.state.Variational(MFmodel, max_parallel=12345)
  MFsampler = qtx.sampler.HopExchangeMix(MFstate, nsamples, ratio=1.0, thermal_steps=100)
  samples = MFsampler.sweep()
  EMF = H.expectation(MFstate, samples)
  print("loaded MF state with energy",EMF)
  jnp.save(f"MF_results/E_Lx{L1}_Ly{L2}_Nt{Ntarget}_U{uMF}_pairing{pairing}_c{c}_nsteps{nstepsMF}_sublattice{sublattice[0]}_{sublattice[1]}_{sublattice[2]}.npy", EMF)


if args.load==0:
  state = qtx.state.Variational(model, max_parallel=max_parallel)
  energy_data = []
  variance_data = []
else:
  filename2 = filename
  print("Load previous calculations:", "states/"+filename2)
  try:
    state = qtx.state.Variational(model, max_parallel=max_parallel, param_file="states/"+filename2)
  except FileNotFoundError:
    filename2 = filename.split(f"nsteps{nsteps}")[0]+f"nsteps500"+filename.split(f"nsteps{nsteps}")[1]
    print("Didnt work! Load instead:", "states/"+filename2)
    state = qtx.state.Variational(model, max_parallel=max_parallel, param_file="states/"+filename2)
  energy_data = list(np.load("results/energy_"+filename2+".npy"))
  variance_data = list(np.load("results/energy_variance_"+filename2+".npy"))
  nsteps -= len(energy_data)

sampler = qtx.sampler.HopExchangeMix(state, nsamples, ratio=1.0, thermal_steps=100)
samples = sampler.sweep()

tdvp = qtx.optimizer.TDVP(state,H,solver=qtx.optimizer.auto_pinv_eig(rtol=rtol))


if run_optimization and nsteps>0:
  lr_decay = (lr-lr/2)/nsteps
  E, VarE = H.expectation(state, samples, return_var=True)
  print("initial", E, VarE)
  energy_data.append(E)
  variance_data.append(VarE)
  for i in range(nsteps):
    start_time = time.time()
    samples = sampler.sweep()
    step = tdvp.get_step(samples)
    state.update(step*lr)
    E = tdvp.energy
    energy_data.append(E)
    VarE = tdvp.VarE
    variance_data.append(VarE)
    end_time = time.time()
    lr -= lr_decay
    print(i,"/", nsteps, E, tdvp.VarE, "(",end_time - start_time,"s)")
    if jnp.isnan(E): break
    if i%10==0 or i==nsteps-1:
        print(i,"/", nsteps, E, tdvp.VarE)
        np.save("results/energy_"+filename+".npy", energy_data)
        np.save("results/energy_variance_"+filename+".npy", variance_data)
        state.save("states/"+filename)

