# greekstream design

## Reverse-mode AD in four sentences

1. Evaluate the function forward. For every intermediate result, record its local partials with respect to its direct arguments.
2. Seed the output's adjoint with 1.
3. Walk the record backwards. Each node adds `partial × its adjoint` into each argument's adjoint.
4. When the walk reaches the inputs, each input's adjoint is ∂output/∂input. One sweep serves *all* inputs.

The sweep visits every node once, so the cost is proportional to the forward evaluation: typically 2–10x one valuation, regardless of how many inputs there are.

## The tape

```cpp
struct Node { double adjoint; uint32_t n; double* partials; double** args; };
```

- **Three block arenas** (nodes, partials, argument pointers) of fixed-size blocks. Blocks are retained across `rewind`/`clear`, so the tape stops allocating after the first evaluation.
- **`Tape::active()`** is a `thread_local` pointer. Each thread records onto its own tape with no synchronization. `ActiveTape` is the RAII switch.
- **Mark / rewind:** `set_mark()` checkpoints the tape position. `propagate_to_mark()` sweeps only what was recorded after it, and `rewind_to_mark()` discards it. Nodes below the mark keep accumulating adjoints. `propagate_mark_to_start()` finally sweeps the region below the mark once.

## Expression templates

Operators on `Expression` types return `Binary<Op, L, R>` / `Unary<Op, A>` objects. They store children (leaves by reference, sub-expressions by value) and their own value. They touch the tape only when assigned to a `Number`:

```cpp
template <Expression E> static Node* record(const E& e) {
    Node* node = tape.record(E::num_numbers);   // count known at compile time
    e.template push<0>(*node, 1.0);             // chain rule, offsets known at compile time
    return node;
}
```

`push<Offset>` recurses: a binary op multiplies the incoming weight by its local partial and forwards `push<Offset>` to the left child and `push<Offset + L::num_numbers>` to the right. A leaf `Number` writes `(weight, &its adjoint)` into slot `Offset`. Passive constants (`Const`) have `num_numbers = 0`, and their `push` compiles to nothing.

**Hazard:** expressions hold references. `auto e = a * b;` dangles once `a`/`b` die. Always assign to `T`. The rule is documented in the header and followed throughout the models.

**Self-assignment:** `y = y * y + y` must read y's *old* node and value. `operator=` computes the value and records the node before overwriting either (test-enforced).

## Models are written once

Every model is a template on `T`. Unqualified calls resolve correctly for both instantiations: `using std::log;` makes `log(double)` visible, and ADL finds `aad::log` for expression types. `ncdf`/`fmax` have double overloads in `greekstream` and expression overloads in `greekstream::aad`.

## Monte Carlo AAD with checkpointing

```
per thread:  record spots, vols, rate              ┐ below mark
             record setup: (r - v²/2)T, v√T, e^(-rT)┘
             set_mark()
             for each block b (stream(seed, b)):
                reset adjoints
                for each path: record payoff → propagate_to_mark → rewind_to_mark
                save adjoints of spots + setup nodes into results[b]
main:        sum results in block order (deterministic)
             re-record setup, seed its adjoints with the sums, propagate_to_start
             → vegas and rho; spot adjoints are already deltas
```

The live tape never exceeds one path, so memory is O(N underlyings), not O(paths).

## Risk engine checkpointing

PV = Σ qᵢ·Vᵢ, so ∂PV/∂x = Σ qᵢ·∂Vᵢ/∂x. Record the market inputs once, then for each position record Vᵢ, seed its adjoint with qᵢ, sweep to the mark and rewind. Peak tape = one position (≈10k nodes for a 100-step American tree), versus 2.1M nodes / 184 MB for the whole book. The sweep stays in L2.

## Bump-and-revalue, done fairly

- **Common random numbers:** every bumped revaluation reuses the same seed, so Monte Carlo noise cancels in the difference.
- **Forward differences** (2N+2 pricings) are what the speedup tables compare against. Central differences (4N+2) would double the bump column.
- **Bump sizes:** relative 1e-4 to 1e-6 for spots, absolute for vol and rate. Tests use the tolerances these imply.

## Determinism contract

- `Rng::stream(seed, block)` gives an independent xoshiro256** stream per block.
- Block size is fixed by configuration, not by thread count.
- Reduction is in block order, so floating-point addition order never depends on scheduling.
- `BasketMc.BitIdenticalAcrossThreadCounts` asserts exact equality: a run on 16 threads can be debugged on a laptop.
