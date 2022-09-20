# Firedrake explicit RK scaling test

Based on Firedrake's DG advection demo.

Basic usage

```bash
python python advection_scaling.py       # 40-by-40 quad DQ1 mesh
python python advection_scaling.py -r 3  # refine mesh by 3x, 120-by-120 mesh
python python advection_scaling.py -o    # save VTK output
```

Example strong scaling test

```bash
mpiexec python python advection_scaling.py -r 12 --nsteps 100 -log_view
```
