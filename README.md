
# fractional_SPDE

Numerical simulation codes for the paper

**J. Barré and H. Horii,  
"Large Deviations, Gradient Flows and Fluctuating Hydrodynamics for Fractional Diffusion."**

This repository contains numerical implementations of the fluctuating
hydrodynamic description of fractional diffusion and direct simulations of
independent symmetric Lévy particles.

## Files

### `stochastic_fractional_diffusion_repository_fixed.c`

Simulation of the fractional SPDEs using a Fourier spectral method.

The code simultaneously simulates

- the SPDE with additive noise, whose covariance is evaluated using the
  deterministic density (`det` in the output filenames), and
- the SPDE with multiplicative noise, whose covariance is evaluated using the
  stochastic density itself (`self` in the output filenames).

### `levy_dynamics_repository.c`

Direct particle simulation of independent symmetric Lévy flights on the
periodic domain.

The Lévy stability index `alpha` can be specified as a command-line argument.
For example,

```bash
./particle_simulation 1.0
```

runs the particle simulation with `alpha = 1`.

## Requirements

The SPDE simulation requires

- FFTW3
- GSL
- OpenMP

The particle simulation requires OpenMP.

On macOS with Homebrew:

```bash
brew install fftw gsl libomp
```

## Compilation on macOS

### SPDE simulation

Using Apple Clang:

```bash
clang -O3 \
  -Xpreprocessor -fopenmp \
  stochastic_fractional_diffusion_repository_fixed.c \
  -I$(brew --prefix libomp)/include \
  -I$(brew --prefix fftw)/include \
  -I$(brew --prefix gsl)/include \
  -L$(brew --prefix libomp)/lib \
  -L$(brew --prefix fftw)/lib \
  -L$(brew --prefix gsl)/lib \
  -lomp -lfftw3_threads -lfftw3 -lgsl -lgslcblas -lm \
  -o spde_simulation
```

Run with

```bash
./spde_simulation
```

### Particle simulation

```bash
clang -O3 \
  -Xpreprocessor -fopenmp \
  levy_dynamics_repository.c \
  -I$(brew --prefix libomp)/include \
  -L$(brew --prefix libomp)/lib \
  -lomp -lm \
  -o particle_simulation
```

For example,

```bash
./particle_simulation 1.0
```

## Parameters

The main simulation parameters are specified near the beginning of each
source file.

These include the stability index, time step, number of time steps, number of
independent realizations, effective particle number, and spatial grid size.

## Output files

The programs generate several groups of output files.

### SPDE simulation

The suffix `det` denotes the additive-noise SPDE, while `self` denotes the
multiplicative-noise SPDE.

#### Initial density

`initial_profile_analytic.txt`

contains the normalized analytic initial density.

#### Test-function observables

Files such as

```text
phi_cosx_moments_Z_det_*.txt
phi_sinx_moments_Z_det_*.txt
phi_cosx_moments_Z_self_*.txt
phi_sinx_moments_Z_self_*.txt
```

contain statistics of the projections

```text
Z(t) = <phi,u(t)>
```

for several test functions. They include the mean, variance, third and fourth
centered moments, and the standard error.

#### Density statistics

Files of the form

```text
density_moments_step*_det_*.txt
density_moments_step*_self_*.txt
```

contain pointwise moments of the density over the Monte Carlo realizations.

Files of the form

```text
density_representative_step*_det_*.txt
density_representative_step*_self_*.txt
```

contain representative realizations of the density field.

Files of the form

```text
density_mean_step*_det_*.txt
density_mean_step*_self_*.txt
```

contain density profiles averaged over the realizations.

Files of the form

```text
mean_profile_det_*.txt
mean_profile_self_*.txt
```

contain the final averaged profiles.

#### Raw samples

When raw sample storage is enabled,

```text
samples_Z_det.bin
samples_Z_self.bin
```

contain raw samples of the test-function observables, while

```text
samples_u_det.bin
samples_u_self.bin
```

contain raw density-field samples.

The corresponding

```text
samples_Z_meta.txt
samples_u_meta.txt
```

files describe the dimensions and storage layout of the binary data.

These binary files can become very large for the full simulation parameters.

### Particle simulation

The particle code generates analogous statistics from direct simulations of
independent Lévy particles.

#### Initial density

Files containing the analytic initial profile and representative empirical
histograms describe the initial particle distribution.

#### Test-function observables

Files containing `phi` and `moments` in their names give statistics of
projections of the empirical density against the test functions used in the
paper.

Raw samples of these observables are also written when the corresponding
storage option is enabled.

#### Density statistics

Files of the form

```text
density_moments_step*_particles_*.txt
```

contain pointwise moments of the particle histogram over independent
realizations.

Files of the form

```text
density_representative_step*_particles_*.txt
```

contain representative particle-density histograms.

Files of the form

```text
density_mean_step*_particles_*.txt
```

contain particle-density profiles averaged over the realizations.

#### Raw density samples

When enabled,

```text
samples_u_particles.bin
```

contains raw histogram-density samples.

The corresponding metadata file records the dimensions and observation times.
