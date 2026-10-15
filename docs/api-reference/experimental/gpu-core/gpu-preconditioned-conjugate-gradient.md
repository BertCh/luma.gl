import {GPUOperationContract} from '@site/src/components/docs/gpu-operation-contract';

# Preconditioned conjugate gradient

<GPUOperationContract operation="gpu-preconditioned-conjugate-gradient" />

## Overview

Solves (I + 4L)x = b by conjugate gradient to diffusion-smooth a 16 × 16 elevation block.

::lens{scenario=gpu-core.conjugate-gradient-program rung=result caption="Largest smoothing change: +197.8 m at row 8, column 6; final residual norm: 175 m."}

`createGPUConjugateGradientProgram` is the unpreconditioned path. Here `b` is the 16 × 16 elevation block at tile rows and columns 24–39, `L` is the 5-point graph Laplacian with zero-flux (Neumann) boundaries, and `x` starts at 0. The loop is bounded at 12 iterations and stops once the squared residual is at most 35,000 m² (a residual norm of 187 m). The datum receipt selects `x[134]` (row 8, column 6): its five CSR coefficients, residual, and solution after each update, with the search direction and GPU scalar `alpha` used for that update. Receipt datum 0 denotes this one selected unknown. The residual series is read from the solver's own GPU scalars, and the record keeps both the intent graph and the compiled graph.

Conjugate gradient (CG) solves sparse symmetric positive-definite systems `Ax = b`. Preconditioned conjugate gradient (PCG) applies an inexpensive approximation to `A^-1` so that the transformed problem converges in fewer iterations.

#### How it computes

{/* lens:createGPUConjugateGradientProgram */}

::lens{scenario=gpu-core.conjugate-gradient-program rung=kernel caption="Inspect the SpMV dispatch and retained WGSL; dot-product kernels reduce into the GPU value arena."}

::lens{scenario=gpu-core.conjugate-gradient-program rung=graph caption="Fold the 12 unrolled, gated iterations back into the CG loop."}

::lens{scenario=gpu-core.conjugate-gradient-program rung=datum datum=0 caption="Follow x[134]: five matrix entries, residual, and each update using the captured alpha and search direction."}

::lens{scenario=gpu-core.conjugate-gradient-program rung=iterations caption="Residual norm falls from 38,903 m to 175 m in 11 iterations; later iterations are gated off."}

::lens{scenario=gpu-core.conjugate-gradient-program rung=formula caption="Read alpha and beta as ratios of GPU-resident dot products."}

## Why preconditioning matters

CG convergence depends on the spectrum of the matrix. Poorly conditioned systems may require many iterations even though every individual iteration is inexpensive. A preconditioner `M` approximates `A` while remaining much cheaper to invert.

```text
A x = b
  │
  ├── residual r = b - A x
  │
  ├── solve approximately M z = r
  │
  └── use z to choose the next search direction
```

A useful preconditioner therefore trades a little additional work per iteration for substantially fewer iterations.

## Jacobi preconditioning

The Jacobi preconditioner keeps only the diagonal of the matrix:

```text
M = diag(A)
M^-1 = diag(1 / A[i,i])
```

For independently chunked CSR input, `GPUJacobiPreconditioner` sums diagonal entries across chunk boundaries and stores the reciprocal. Duplicate diagonal entries follow the same summation convention as SpMV. Missing or zero diagonals produce zero. `GPUElementwise` applies the preconditioner with a multiply:

```text
z[i] = inverseDiagonal[i] * r[i]
```

The reciprocal diagonal is constructed once and reused across solver iterations.

Jacobi is deliberately the first preconditioner because it is simple, parallel and exposes the solver architecture without hiding it behind a complicated factorization.

## PCG iteration

With `z = M^-1 r`, PCG becomes:

```text
r₀ = b - A x₀
z₀ = M^-1 r₀
p₀ = z₀
ρ₀ = r₀ · z₀

repeat
    q = A p
    α = ρ / (p · q)
    x = x + α p
    r = r - α q

    stop if ||r|| is small enough

    z = M^-1 r
    ρnew = r · z
    β = ρnew / ρ
    p = z + β p
    ρ = ρnew
```

This is intentionally expressed in terms of reusable GPU graph operations:

```text
adaptive SpMV ───────────────┐
                             │
hierarchical reduction ─ dot ├─ GPUScalar
                             │      │
Jacobi ─ vector multiply     │   scalar arithmetic
                             │      │
                             └── vector MADD
                                      │
                               GPU convergence gate
```

PCG is therefore a composition test for the GPU graph architecture rather than a private solver kernel.

## Graph API and convergence

`GPUJacobiPCG` accepts graph data or vector views for CSR row offsets, column indices, values,
right-hand side and solution. The operands can have independent chunk boundaries. Solver scratch
follows the right-hand side topology and remains graph-owned. Input storage is borrowed; only the
solution and declared scratch are written.

```ts
import {GPUJacobiPCG} from '@luma.gl/gpgpu/gpu-core/gpu-pcg';

const solver = new GPUJacobiPCG({
  rowOffsets, columnIndices, values, rhs, solution,
  columns: rhs.length,
  iterations: 64,
  toleranceSquared: 1e-12
});
graph.add(solver);
const {initialResidualSquared, residualSquared, breakdown} = solver.getResult();
const compiled = graph.compile();
compiled.encode(encoder, {parameters: undefined});
```

`iterations` is a maximum budget. `toleranceSquared` is an absolute squared residual threshold,
defaulting to `1e-12`. Every encoding starts from the current solution and computes `rhs - A * solution`.
An already-converged initial guess performs no solver updates. GPU predicates stop later updates
when the residual reaches the tolerance; no per-iteration CPU readback is required.

The result scalars can feed later graph operations or explicit diagnostic copies. `breakdown` is
nonzero if a nonpositive preconditioned residual product or search curvature prevents a valid step.
Such a solve preserves its last valid solution instead of dividing by zero. This is an SPD solver;
these checks do not establish that an arbitrary input matrix is positive definite. Small positive
curvature and ill-conditioned inputs remain subject to ordinary float32 accuracy limits.

`createGPUConjugateGradientProgram()` provides the unpreconditioned semantic program path. It
also checks the initial residual before entering its loop, stops on nonpositive curvature, and
returns `residualSquared` and `breakdown` program scalars in addition to its resource bindings.

Both paths use `GPUDotProductScalar`: bounded hierarchical reduction over borrowed spans, optional
subgroup reduction, and direct scalar-arena output. Each gated level uses its own exact workgroup
count, including multidimensional dispatch and independently chunked vector updates.

## Hero example: interactive Poisson solve

PCG enables an end-to-end scientific-compute example with a strong visual and architectural story:

```text
paint sources / obstacles / boundaries
                 │
                 ▼
           Poisson system
                 │
                 ▼
             sparse CSR
                 │
                 ▼
       Jacobi-preconditioned CG
                 │
                 ▼
          GPU solution field
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
 beauty visualization   engineering view
                           iterations
                           residual
                           SpMV strategy
                           GPU timings
```

The important property is interactivity: changing the domain changes real GPU computation rather than selecting a canned result.

## Future preconditioners

The solver should accept preconditioners as composable operations rather than encode Jacobi as a permanent special case. More sophisticated options can be considered after the PCG composition and hero workload are proven. Jacobi provides the baseline contract.
