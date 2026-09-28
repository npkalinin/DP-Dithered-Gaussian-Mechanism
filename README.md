# Dithered Gaussian Mechanism

Code accompanying the paper [Dithered Gaussian Mechanism for Randomness-Efficient Differential Privacy](https://arxiv.org/abs/2607.06320) by Nikita P. Kalinin and Rasmus Pagh.

Mathematically, the dithered Gaussian mechanism can be understood as rounding the output of the Gaussian mechanism to a randomly shifted grid. After fixing the grid shift, the rounded output has an explicit discrete distribution whose CDF can be written directly using the Gaussian CDF. We can therefore sample the rounded output directly, without first sampling continuous Gaussian noise.

This repository contains an implementation of this direct sampling procedure.

The mechanism has three main properties:

- It is a post processing of the Gaussian mechanism and therefore inherits its privacy guarantees and Gaussian based privacy accounting.
- Its output lies on a discrete grid, avoiding floating point vulnerabilities associated with directly releasing finite precision samples from a continuous Gaussian distribution.
- It is randomness efficient, requiring substantially fewer random bits for the privacy critical sampling step.
- Its coordinate-wise noise distribution is unbiased and has small total variation distance from Gaussian noise.

## Dithered Gaussian mechanism

Let $f(X)\in\mathbb{R}^d$ be the vector to privatize, let $\sigma>0$ be the Gaussian noise scale, and let $\xi>0$ be the grid width.

Sample public $a,b\sim\mathrm{Uniform}([0,1))$ and define the grid

```math
\gamma_i = (ai+b) \bmod 1.
```

Let $y_i\sim\mathcal N(0,\sigma^2)$ and define

```math
Z_i = \Bigg\lfloor  \frac{f(X)_i+y_i}{\xi}-\gamma_i +\frac{1}{2}\Bigg\rfloor.
```

The mechanism outputs

```math
\mathcal M(f(X))_i = \xi(Z_i+\gamma_i).
```

Thus, it rounds the Gaussian mechanism output to the shifted grid $\xi(\mathbb Z+\gamma_i)$.  Conditioned on $\gamma_i$, the CDF of $Z_i$ is

```math
\Pr[Z_i\leq k]
=
\Phi\left(
\frac{
\xi\left(k+\gamma_i+\frac{1}{2}\right)-f(X)_i
}{
\sigma
}
\right),
```

where $\Phi$ is the standard Gaussian CDF.
This is the discrete distribution sampled by the implementation.

## Sampling procedure

To sample $Z_i$, consider a uniform random variable $U_i\in[0,1)$. The desired value is the unique integer $k_i$ satisfying $F_i(k_i-1)\leq U_i<F_i(k_i)$.

Instead of generating $U_i$ to a fixed precision, we generate its random bits in blocks. The bits generated so far define an interval $[L_i,R_i)$ containing $U_i$. The midpoint is used to obtain a candidate $k_i$, and the candidate is accepted once

```math
[L_i,R_i)\subseteq[F_i(k_i-1),F_i(k_i)).
```

Otherwise, more random bits are generated for that coordinate.

The fast path is vectorized in PyTorch. Candidates are computed with `torch.special.ndtri`, while the CDF boundaries are evaluated with `torch.special.ndtr`. Coordinates are accepted only when the containment test can be certified with a conservative numerical margin.

Coordinates that remain unresolved after the fast refinement steps are passed to the `mpmath` library. There, the original `float64` parameters are converted to their exact rational values and the CDF boundaries are recomputed using arbitrary precision. The precision is increased as more random bits are generated until the coordinate can be resolved.

In this way, finite precision affects only which numerical path is used; the sampler does not truncate the support of the distribution.

## Citation

If you use this code in your work, please cite:

```bibtex
@misc{kalinin2026dithered,
  title   = {Dithered Gaussian Mechanism for Randomness-Efficient Differential Privacy},
  author  = {Kalinin, Nikita P. and Pagh, Rasmus},
  note    = {arXiv preprint arXiv:2607.06320},
  year    = {2026}
}
```
