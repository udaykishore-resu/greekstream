# C++ feature policy

Risk engines are throughput-bound, not tail-latency-bound, so this repo allows more abstraction than limitbook and feedline. The abstraction has to dissolve at compile time, though.

## Used, and why

| Feature | Where | Why |
|---|---|---|
| Expression templates | `aad/number.hpp` | Fewer tape nodes = less memory traffic in the sweep; the main AAD performance lever |
| Concepts (`Expression`) | AAD operators | Operators only match AAD types, so `double * double` is never hijacked; readable errors |
| `if constexpr` | `Binary::push` | Branches for zero-argument children (constants) vanish at compile time |
| Templates on the number type | every model | One implementation priced in `double` or differentiated with `aad::Number`: no drift between "pricing code" and "risk code" |
| `thread_local` | `Tape::active()` | Per-thread tapes, no locks |
| `std::numbers::sqrt2`, `inv_sqrtpi` | normal pdf/cdf | Exact constants, no hand-typed digits |
| Defaulted copy = shared node | `aad::Number` | A copy is the *same* variable for differentiation; documented, deliberate |
| `__extension__ unsigned __int128` | `Rng` | Unbiased bounded integers (Lemire) without `-Wpedantic` noise |
| Deleted copy/move | `Tape`, `ActiveTape` | Nodes point into the tape; moving it would dangle every `Number` |

## Deliberately avoided

| Feature | Why |
|---|---|
| `auto` for expression results | Expressions hold references; `auto` captures a dangling view. Assign to `T` |
| Implicit `Number → double` conversion | Would silently drop derivatives; use `.value()` explicitly |
| `std::uniform_*_distribution`, `std::normal_distribution` | Implementation-defined streams break determinism across libstdc++/libc++ |
| Box-Muller | Breaks the one-to-one uniform → normal map that quasi-random sequences need; inverse CDF instead |
| Virtual "Instrument" hierarchy on the hot path | Not needed yet; positions are plain data and pricing dispatch is a branch |
| `std::mdspan` (C++23) | Natural for path matrices later; not yet in AppleClang's libc++ for all versions in use |

## Numeric conventions

- Vega and rho are per **1.00** change (not per 1% or per 1 bp). Scale at the presentation layer.
- Rates are continuously compounded; time is in years.
