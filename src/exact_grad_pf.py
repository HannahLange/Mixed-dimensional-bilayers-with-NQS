import quantax as qtx
from jax import numpy as jnp
import jax
from quantax.operator import create_u, create_d, annihilate_u, annihilate_d
from matplotlib import pyplot as plt
from quantax.symmetry import Identity, Trans2D, C4v, SpinInverse, Translation, Trans3D
import equinox as eqx
from quantax.operator import *
from exact_grad_pf_helper import *
from observables import *
import argparse
import os
import sys


parser = argparse.ArgumentParser()
parser.add_argument(  "-l",  "--length"         , type=int   , default = 4      , help="length of physical system")
parser.add_argument(  "-w",  "--width"          , type=int   , default = 4      , help="width of physical system")
parser.add_argument(  "-Np", "--Np"             , type=int   , default = 32     , help="Number of particles")
parser.add_argument(  "-sl", "--sublattice"     , type=int, nargs=3, default = [2,2,2], help="list that specifies the sublattice")
parser.add_argument(  "-U",  "--U"              , type=float , default = 3.     , help="On-site repulsion strength") 
parser.add_argument(  "-p",  "--pairing"        , type=float , default = 0.     , help="Strength of pairing term")
parser.add_argument(  "-c",  "--c"              , type=float , default = 0.1    , help="Prefactor of (N-Ntarget)**2")
parser.add_argument(  "-steps", "--steps"       , type=int   , default = 1000   , help="Number of optimization steps")
parser.add_argument(  "-load", "--load"	        , type=int   , default = 0      , help="if 1: loads from previous runs, 0: runs the optimization")


args   = parser.parse_args()
L1         = args.length
L2         = args.width
Ntarget    = args.Np
sublattice = (args.sublattice[0],args.sublattice[1],args.sublattice[2])

u          = args.U
pairing    = args.pairing
c          = args.c
n_steps    = args.steps

filename = f"Lx{L1}_Ly{L2}_Nt{Ntarget}_U{u}_pairing{pairing}_c{c}_nsteps{n_steps}_sublattice{sublattice[0]}_{sublattice[1]}_{sublattice[2]}"

file_to_check = f'MF_results/orbs_'+filename+'.npy'
if os.path.exists(file_to_check):
    print(f"Skip MF training: '{file_to_check}' already exists.")
    sys.exit(0)

lattice = qtx.sites.Grid((2, L1, L2), Nparticle=((Ntarget+1)//2,Ntarget//2), boundary=(0,1,1), is_fermion=True, double_occ=False)
model = qtx.model.Pfaffian(dtype=jnp.float64, sublattice=sublattice)


#some definitions
F = model.F
index = model.index
N = lattice.N
_, u_weight, idx = gen_U(N,u).jax_op_list[0]
H = qtx.operator.Hubbard(U=u)
T = gen_kin_energy(N,H)



@jax.jit
def E_HF(F, T, u_weight, index, c):
    # get the full F matrix (expand w.r.t. the sublattice)
    F_full = F[index]
    F_full = 1j*(F_full - F_full.T)

    # F = U diag(D) U*T
    D, U = jnp.linalg.eigh(F_full)
    U = U[:,:N][:,:,None]
    U = jnp.concatenate((U.real,U.imag),-1).reshape(2*N,2*N)
    norm = jnp.linalg.norm(U,axis=0)

    #rescale U and D
    U = U/norm[None]
    Dn = D[:N]*norm[:N]**2

    # define the angle
    theta = jnp.arctan(Dn)
    v = jnp.sin(theta)
    D_factor = jnp.diag(jnp.repeat(v ** 2, 2))
    rho = U.conj() @ D_factor @ U.T

    # kinetic energy
    ET = -jnp.sum(rho * T)

    # interaction energy
    v_ = jnp.sin(theta) * jnp.cos(theta)
    v_ = jnp.stack([v_, jnp.zeros_like(v_)], axis=1).flatten()[:-1]
    v_ = jnp.diag(v_, k=1)
    D_factor = v_ - v_.T
    cc_correlation = U @ D_factor @ U.T

    #chemical potential
    rho_ii = jnp.diag(rho)[:N]
    rho_jj = jnp.diag(rho)[N:]

    # pairing term
    pairing_mat = gen_pairing_mat(pairing, N, lattice)

    #EV = fben(rho, cc_correlation, u_weight, idx) - mu * jnp.sum(rho_ii + rho_jj) + jnp.sum(pairing_mat*cc_correlation)
    #constraint = c * (jnp.where(jnp.isclose(jnp.sum(rho_ii + rho_jj)/Ntarget,0.,atol=0.05),0,jnp.sum(rho_ii + rho_jj)) - Ntarget)**2
    EV = fben(rho, cc_correlation, u_weight, idx) + jnp.sum(pairing_mat*cc_correlation) + c*(jnp.sum(rho_ii + rho_jj) - Ntarget)**2 # - u/2*jnp.sum(rho_ii + rho_jj)
    return ET + EV


Egrad = jax.jit(jax.value_and_grad(E_HF, argnums=0))

if args.load==0:
  Ndata = qtx.utils.DataTracer()
  energy = qtx.utils.DataTracer() 

  for i in range(n_steps):
      E, (gradF) = Egrad(F, T, u_weight, index, c)
      energy.append(E)

      # calculate particle number
      F_full = F[index]
      F_full = 1j*(F_full - F_full.T)
      D, U = jnp.linalg.eigh(F_full)
      U = U[:,:N][:,:,None]
      U = jnp.concatenate((U.real,U.imag),-1).reshape(2*N,2*N)
      norm = jnp.linalg.norm(U,axis=0)
      U = U/norm[None]
      Dn = D[:N]*norm[:N]**2
      theta = jnp.arctan(Dn)

      v = jnp.sin(theta)
      D_factor = jnp.diag(jnp.repeat(v ** 2, 2))
      rho = U.conj() @ D_factor @ U.T
      rho_ii = jnp.diag(rho)[:N]
      rho_jj = jnp.diag(rho)[N:]
      Nparticle = jnp.sum(rho_ii + rho_jj) #2 * jnp.sum(jnp.sin(theta) ** 2)
      Ndata.append(Nparticle/(2*L1*L2))

      if jnp.isnan(gradF).any(): break
      # update F
      F -= gradF * 0.01
      #c += 10/n_steps
      if i % 50 == 0:
          print(i, E, Nparticle)

  jnp.save(f'MF_results/theta_'+filename+'.npy', theta)
  jnp.save(f'MF_results/orbs_'+filename+'.npy', F)
  jnp.save(f'MF_results/energy_'+filename+'.npy', energy)
  jnp.save(f'MF_results/N_'+filename+'.npy', Ndata)
elif args.load==1:
  F = jnp.load(f'MF_results/orbs_'+filename+'.npy', F)
  theta = jnp.load(f'MF_results/theta_'+filename+'.npy', theta)

drop_index, drop = find_sharp_drop(theta**2)
print(f"Sharp drop detected at index {drop_index} with value change {drop}")
print(theta**2)

F_full = F[index]
F_full = 1j*(F_full - F_full.T)
den_exact = calculate_exact_density(F_full,lattice.N)
print(den_exact[0]+den_exact[2], den_exact[1]+den_exact[3])
print(den_exact[0]-den_exact[2], den_exact[1]-den_exact[3])

model = eqx.tree_at(lambda model: model.F, model, F.ravel())
state = qtx.state.Variational(model, max_parallel=12345)

sampler = qtx.sampler.HopExchangeMix(state, 1000, ratio=1.0, thermal_steps=10)

samples = sampler.sweep()
print(jnp.max(state(samples.spins)))
#assert np.all(np.isclose(sampler.wf,state(samples.spins)))

