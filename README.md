Code and data to the paper ["Simulating superconductivity in mixed-dimensional $t_\parallel$-$J_\parallel$-$J_\perp$ bilayers with neural quantum states"](https://arxiv.org/pdf/2602.10091)

Using Gutzwiller-projected Hidden Fermion Pfaffian States, we study the mixed-dimensional bilayer model

$$
\begin{aligned}
\mathcal{H} ={}&
-t_\parallel \sum_{\langle \mathbf{i},\mathbf{j} \rangle,\mu,\sigma}
\hat{\mathcal{P}}_G
\left(
\hat{c}^{\dagger}_{\mathbf{i},\mu,\sigma}\hat{c}_{\mathbf{j},\mu,\sigma}
+ \text{h.c.}
\right)
\hat{\mathcal{P}}_G
\\
&+ J_\parallel \sum_{\langle \mathbf{i},\mathbf{j} \rangle,\mu}
\left(
\hat{\mathbf{S}}_{\mathbf{i},\mu}\cdot \hat{\mathbf{S}}_{\mathbf{j},\mu}
-\frac{1}{4}\hat{n}_{\mathbf{i},\mu}\hat{n}_{\mathbf{j},\mu}
\right)
\\
&+ J_\perp \sum_{\mathbf{i}}
\left(
\hat{\mathbf{S}}_{\mathbf{i},0}\cdot \hat{\mathbf{S}}_{\mathbf{i},1}
-\frac{1}{4}\hat{n}_{\mathbf{i},0}\hat{n}_{\mathbf{i},1}
\right).
\end{aligned}
$$

Here $\hat{c}^{(\dagger)}_{\mathbf{i},\mu,\sigma}$ annihilates (creates) a fermion
at site $\mathbf{i}$ in layer $\mu=0,1$ with spin $\sigma=\pm \tfrac{1}{2}$,
and $\hat{\mathbf{S}}_{\mathbf{i},\mu}$ and $\hat{n}_{\mathbf{i},\mu}$ are the
spin and density operators.
