from jax import numpy as jnp
import quantax as qtx
from quantax.operator import create_u, create_d, annihilate_u, annihilate_d, number_u, number_d



# some helper functions
def fben(cdagc, cc, weight, idx):
    """
        Calculates the exact energy.
    """
    energy = cdagc[idx[:,0],idx[:,1]]*cdagc[idx[:,2],idx[:,3]]
    energy -= cdagc[idx[:,0],idx[:,3]]*cdagc[idx[:,2],idx[:,1]].conj()
    energy += cc[idx[:,0],idx[:,2]]*cc[idx[:,1],idx[:,3]].conj()

    return jnp.sum(energy*weight)


def gen_kin_energy(N,H):
    """
        Generates the kinetic energy matrix.
    """
    T = jnp.zeros((2*N, 2*N), dtype=jnp.float64)
    T_idx = H.jax_op_list[0][2]
    T = T.at[T_idx[:, 0], T_idx[:, 1]].set(1)
    return T

def gen_pairing_mat(pairing, N, lattice):
    """
        Generates the pairing matrix.
    """
    pairing_mat = jnp.zeros([2*N,2*N])
    for neighbor in lattice.get_neighbor():
      x1, x2 = lattice.xyz_from_index[neighbor[0]], lattice.xyz_from_index[neighbor[1]]
      diff_z = jnp.abs(x1[1]- x2[1])
      pairing_mat = jnp.where(diff_z == 1,
                              pairing_mat.at[neighbor[0]+N, neighbor[1]].set(pairing).at[neighbor[1]+N, neighbor[0]].set(pairing),
                              pairing_mat)
    return pairing_mat


def gen_U(N,u):
    def onsite(i):
      return create_u(i)*annihilate_u(i)*create_d(i)*annihilate_d(i)
    U = 0
    for i in range(N):
      U = U + u*onsite(i)
    return U


def find_sharp_drop(arr, threshold_ratio=0.1):
    """
    Finds the index where a sharp drop in the array occurs.

    Parameters:
    - arr: 1D array-like (list or np.array)
    - threshold_ratio: a float indicating the proportion of the max value change 
      considered to be a 'sharp drop'. Default is 0.5 (i.e., 50% or more drop).

    Returns:
    - drop_index: index **after** the sharpest drop (i.e., the drop is from arr[i] to arr[i+1])
    - drop_value: the actual difference value at that drop
    """
    arr = jnp.array(arr)
    diffs = jnp.diff(arr)
    
    # Find where the drop is sharpest
    min_diff = jnp.min(diffs)
    drop_index = jnp.argmin(diffs)
    
    # Optionally check if it's a "sharp enough" drop
    if abs(min_diff) > threshold_ratio * jnp.max(arr):
        return drop_index + 1, min_diff
    else:
        return None, None



def calculate_exact_density(F,N,split=4):
    """
        Calculates the exact density.
    """
    D, U = jnp.linalg.eigh(F)
    U = U[:,:N][:,:,None]
    U = jnp.concatenate((U.real,U.imag),-1).reshape(2*N,2*N)
    norm = jnp.linalg.norm(U,axis=0)
    U = U/norm[None]
    Dn = D[:N]*norm[:N]**2
    theta = jnp.arctan(Dn)
    v = jnp.sin(theta)
    D_factor = jnp.diag(jnp.repeat(v ** 2, 2))
    rho = U.conj() @ D_factor @ U.T
    occ = jnp.diag(rho)
    if split==4: 
        return jnp.split(occ,4) #returns up_top, up_bottom, down_top, down_bottom
    elif split==2: 
        return jnp.split(occ,2)
    else: 
        raise NotImplementedError
