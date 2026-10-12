# Roadmap

Milestone 1 is done: tape, expression templates, three models, deterministic MC, a risk engine with checkpointing, and an AAD-vs-bump benchmark suite.

## M2: Hand-written adjoint of the tree
- Write `binomial_crr_adjoint(...)`: run the forward induction storing node values, then propagate adjoints backwards by hand (the `max` routes to the winning branch).
- **Target:** ≥5x faster than the tape version at 100 steps, identical Greeks to 1e-12.
- **Lesson:** production libraries hand-code adjoints for their hottest kernels and use the tape for everything else.

## M3: SIMD Monte Carlo
- Price 4 (NEON) or 8 (AVX2) paths per instruction in `basket_price`: structure-of-arrays normals, vectorized `exp`.
- Compare auto-vectorization (`-O3 -march=native`, check with `-Rpass=loop-vectorize`) against explicit intrinsics or `std::experimental::simd`.

## M4: Parallel risk engine
- Shard positions across threads; each thread owns a tape and accumulates input adjoints; reduce in shard order.
- **Answer:** scaling efficiency from 1 to 8 threads on your Mac's performance cores.

## M5: Gamma
- Forward-over-reverse: run the AAD sweep with dual numbers to get one Hessian column per sweep.
- Compare against bump-of-AAD (central difference of AAD deltas).

## M6: Better Monte Carlo
- Full correlation matrix via Cholesky (and its adjoint).
- Sobol sequences with Brownian-bridge construction; measure the variance reduction.

## M7: Event-driven risk
- Subscribe to a tick stream (feedline's SPSC ring); recompute only positions whose underlying ticked.
- Report tick-to-risk latency percentiles.
