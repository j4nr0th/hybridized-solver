# hybsol — feedback from integrating it into `fdg`

Written while wiring `hybsol` into the `fdg` FEM library as the solver for
continuity-constrained mixed problems (an `interp` submodule). Every claim below is
measured; reproductions are given so you can check them in your own repo. Nothing here
is a criticism of the design — the block solver is correct and exact where it applies,
and the notes are about where its current contract meets a real workload and what would
unlock more.

## The use case, for context

A mesh of hypercube elements, each with a dense local operator
`A_e = [[M_q, Dᵀ], [D, 0]]` (`M_q` the mass matrix of a trial k-form, `D` the incidence
operator applied to the test field's mass matrix), plus continuity constraints coupling
elements across shared boundary objects. Solving the augmented system
`[[A, N], [Nᵀ, 0]]` is the natural formulation: element blocks are nonsingular, the
multipliers fall out of the factorization, and no Schur complement is ever formed. That
is the shape `hybsol` fits, and it is why I reached for it.

---

## 1. Correctness / contract — the unpivoted precondition needs to be *findable*

**Concern: correctness, API contract.**

`decompose` requires every **leading principal minor of every block** to be
nonsingular, because there is no pivoting. A block that is nonsingular and well
conditioned can still fail:

```python
# nonsingular, cond(A) = 1.13e1, rank 4/4
D = np.array(
    [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
BlockSystem.from_blocks([4], [0], [0], D.ravel()).decompose()
# ValueError: decompose: zero pivot in LU decomposition
```

The leading 3×3 minor is singular. `hybsol.h` documents the precondition as
`CUTL_ASSERT`-checked caller guarantees, but from a caller's side this is not a
"caller mistake" they can detect — it is a property of the system they just built.

**What would help, in rough order of value to me:**

1. **A cheap `is_factorable()` predicate** — walk the block elimination symbolically
   (no numerics) and report whether any diagonal block has a singular leading minor,
   plus *which block index*. Ideally this reuses the walk you already do in
   `hybsol_fill_plan`, so it costs nothing extra.
2. **Name the offending block in the error.** `decompose: zero pivot` does not say
   *where*. For a 3000-block system that is a genuinely painful debug.
3. **Consider documenting this as a first-class limitation** in the Python docstring of
   `decompose` and in `docs/python_api.rst`, with the 4×4 example above. It is the single
   fact that determines whether a formulation is usable with `hybsol` at all.

With a predicate I could, for instance, pick a different block granularity per problem
instead of discovering the problem 20 minutes in.

## 2. Correctness — near-singular blocks fail *silently*

**Concern: correctness, numerical robustness.**

There is a real asymmetry, and it is the more dangerous half of the same issue:

| system | behaviour |
|---|---|
| exactly singular block | raises `ValueError: decompose: zero pivot in LU decomposition` |
| near-singular block (cond ~1e16) | **factorizes without error and returns garbage** |

Concrete case: a 2-form/3-form element with a 0-form trace test — the local stiffness
`A_e = DᵀM₁D` has rank 15 of 16 (constants in the nullspace), cond 7.1e16. Handed to
`hybsol`, the solve returned `|x| ≈ 2.6e16` where the pivoted reference gives 90.7.
`decompose` reported success, and `solve` reported success.

**What would help:**

- **`solve(..., check=True)` or a `solve_checked()`**: verify `‖Ax − b‖` against a
  tolerance and raise if it is out of band. A caller that has been burned once will
  always want this on. It is cheap relative to the factorization.
- Alternatively, have `decompose` detect a near-zero pivot *relative to the block norm*
  rather than only an exact zero, and report it. The risk is false positives on
  legitimately ill-conditioned systems, so a separate opt-in mode seems safer than
  changing the default.
- At minimum, document the asymmetry plainly: "exactly singular raises; numerically
  singular may silently produce a meaningless solution".

## 3. Performance — the per-pass scan dominates large factorizations

**Concern: performance, algorithmic complexity.**

The ready-set search is a full rescan every pass:

```c
for (uint64_t i = 0; i < sys->n; ++i) { /* check status, collect ready */ }
```

The pass count equals the depth of the elimination chain, so the scan contributes
`O(depth × n_blocks)` on top of the real work. This is the dominant cost at the sizes I
care about. Measured on a mixed FEM problem, element blocks (44×44) + one multiplier
block per boundary object:

| elements | blocks | recorded ops | decompose |
|---|---|---|---|
| 96 | 320 | 11,075 | 0.03 s |
| 384 | 1,376 | 114,775 | 1.42 s |
| 864 | 3,168 | 421,643 | 6.65 s |

**What would help:** replace the rescan with a **worklist / frontier** — maintain a
queue of rows whose blocking row has just finished, and only visit those. That turns
`O(depth × n)` into `O(eliminations)`. The recorded operation count is already the right
order of magnitude, so this should be close to a constant-factor win on the same
algorithm, not a speculative one.

## 4. Performance — the operation record is expensive and rarely needed

**Concern: performance, API design.**

`hybsol_system_operations` exposes every recorded step, and the Python
`operations()` returns `tuple[tuple[int, ...], ...]` — a Python tuple of tuples. For the
864-element case that is 421,643 tuples; any caller who touches it pays for a full
materialization it almost certainly does not need. The C side has
`hybsol_system_apply_operations` for replay, which is the right shape.

**What would help:**

- A `record_operations=False` option on `decompose` (default on for compatibility) for
  callers who factor once and solve once. The fill pool is already sized symbolically,
  so this may be close to free.
- In Python, expose `n_operations` directly (currently only `len(operations())`).
- Keep `operations()` for inspection, but document that it is O(ops) in Python objects and
  point at `apply_operations` for bulk use.

## 5. Correctness / API safety — `compute_reordering` can produce a *wrong* ordering

**Concern: correctness, API safety — I consider this the sharpest edge in the API.**

`compute_reordering` returns a coloring: it groups blocks that share **no** non-zero
block. That is a legitimate thing to want, but it is **not** an elimination ordering for
your own `decompose`, and feeding it back in is a correctness hazard, not a slowdown:

```python
system.compute_reordering("greedy")
system.reorder_blocks(ordering)
system.decompose()
# ValueError: decompose: zero pivot in LU decomposition
```

This fails at every size I tried. The reason is instructive: in an augmented system the
multiplier region is *structurally zero*, so the multiplier blocks look like a perfect
independent set to a non-adjacency coloring, and the coloring is free to schedule them
**before** the element blocks they depend on. A block whose diagonal is still zero at
pivot time cannot be inverted, and there is no pivoting to recover.

So the block order here encodes a genuine partial order, and the API currently offers
something that looks like a general ordering tool but silently violates the
precondition of the factorization in the same module.

**What would help — any one of these:**

- Rename/re-document so the intent is unambiguous, e.g. state that it produces a
  *coloring* for callers who want to schedule independent blocks themselves, and that it
  is **not** valid input to `reorder_blocks` for this factorization.
- Better: give `decompose` a way to take precedence constraints, or add
  `compute_reordering(..., respect=<level vector>)`, so a caller can say "elements before
  multipliers" and get a valid permutation.
- Or: have `reorder_blocks` verify that the resulting order is still valid for the
  decomposition (no block placed before one it depends on) and raise with a clear message
  rather than letting the failure surface later as a zero pivot.

Even with perfect documentation I would keep `reorder=False` by default, because the
failure is a wrong answer, not an exception, in the cases where the pivot happens to be
merely small.

## 6. Performance / memory — dense coupling blocks are mostly structural zeros

**Concern: performance, memory.**

Blocks are dense by construction. In the FEM use case the coupling between an element and
a boundary object is an `n_e × R_g` block holding at most `R_g` non-zeros in a single
column per constraint row. For the 384-element case that is a factor of ~11 in wasted
bandwidth and memory.

This is a caller-side shape choice as much as a library limitation, and grouping rows per
object already made it much better than one block per row. Still, if tall-thin blocks
were common in your other users, a **sparse or column-compressed block type** would remove
it entirely.

## 7. Usability — the live-view guard is a trap in long loops

**Concern: usability, performance.**

`decompose`, `reorder_blocks` and `eliminate_row` all refuse to run while any array from
`block_storage` is alive. That is a good safety property — it is what stops a stale
NumPy view from being written under a factorized system — but the failure mode is easy to
hit accidentally:

```python
views = [system.block_storage(i, i) for i in range(n)]  # decompose now refuses
```

**What would help:** a context manager (`with system.block_storage(i, i) as v:`) so the
lifetime is lexical, and including the offending block index in the error. The current
message naming the count of live views is good; the index would make it actionable.

Separately, the *semantics* are exactly right and worth keeping prominent in the docs:
first call creates zero-filled, re-fetch returns the same buffer without clearing, and
writing the diagonal invalidates the cached factorization. That let me pre-allocate a
whole pattern and fill it later, which is the shape a C core wants.

## 8. Documentation — the transpose ravel trap

**Concern: documentation, correctness.**

`from_blocks` takes row-major concatenated block data, and the validation
(`data holds N values, but the given rows and cols need M`) caught my error immediately —
that part is good. But one subtlety cost me real debugging time and is not stated
anywhere I could find:

> **A block and its transpose do not share a row-major ravel ordering** once a block has
> more than one column.

For an `n × 1` block `[a₀ … aₙ₋₁]` and its `1 × n` transpose, the ravels are identical —
which is exactly why a one-column-block assembly is *accidentally* correct. For
`m × n` versus `n × m` they are different permutations, and the transposed block needs its
own data. I shipped a silently non-symmetric matrix this way; `as_array()` not being
symmetric was the fastest tell, and `decompose` failed later with a zero pivot rather than
at assembly.

**What would help:** state the ravel order and this asymmetry in the `from_blocks` /
`add_blocks` docstrings, and — more usefully — consider a `add_transposed_blocks` or a
`symmetric=True` convenience that fills both directions, since symmetric storage is the
common case for this kind of solver.

## 9. Python API — some C capability is not surfaced

**Concern: API completeness, performance.**

`decompose.h` exposes `hybsol_fill_plan` and `hybsol_system_operation_bound`, both of
which answer "how expensive will this be?" *before* committing to a factorization.
Neither appears in `_mod.pyi`. Given that a caller has to choose block granularity
before seeing any numbers, being able to ask `system.fill_plan()` and
`system.operation_bound()` before assembling would be genuinely useful — it would have
told me to group per object much sooner.

Also not surfaced: `hybsol_system_apply_operations` (bulk replay without materializing),
`hybsol_system_solve_upper` and `hybsol_workspace_bytes` (the latter is exposed as
`workspace_bytes`).

## 10. Small things

- **`Precision` is defined twice** — as a `StrEnum` in `hybsol/__init__.py` and again in
  `_mod.pyi`. The runtime one is the `__init__` definition, so the stub can drift. Not
  important, but a single source would be nicer.
- **Single precision and tolerances.** `solve` always takes and returns doubles, which is
  convenient, but a caller checking a residual needs to know the achievable accuracy is
  `~cond·1e-7` in single. I tripped over exactly this (a `1e-6` guard firing on a
  legitimate `5.8e-6` single-precision residual). Worth a line in the docstring, or an
  accuracy hint returned from `solve`.
- **Error mapping** — everything data-related maps to `ValueError`, which is a reasonable
  default. Given §1 and §2, a distinct exception for "singular system" (as opposed to
  "malformed input") would let callers fall back deliberately rather than by string
  matching.

---

## What is working well

Worth saying, because these are the parts I built on:

- **Exactness.** On random symmetric systems with diagonal, chain, star and fully-coupled
  patterns (block sizes 2–4), residual `2e-16` to `5e-15` and agreement with a dense
  solve to the same. No accuracy concerns on the path where it applies.
- **OpenMP scaling is real.** Decompose + assemble, 384 elements: 1.11 s (1 thread) →
  0.92 (2) → 0.69 (4) → 0.65 s (default). Sublinear, no pathological spinning.
- **`from_blocks` as a single-pass assembly path** is the right shape for bulk
  construction, and the flat-data length check caught a genuine bug of mine immediately.
- **Zero-alloc-inside-parallel-region discipline** (fill pool sized by `hybsol_fill_plan`
  before any thread starts) is a careful design choice and clearly deliberate.
- **`block_storage` semantics** (create zero-filled, re-fetch untouched) are exactly
  right for pre-allocated assembly, which is the pattern a C core wants.

## Ranking, if you have time for only a few

1. **§5** — `compute_reordering` can silently break the factorization. Cheapest to fix
   (document, or validate in `reorder_blocks`) and removes a wrong-answer hazard.
2. **§1 + §2** — a `is_factorable()` predicate and an opt-in `check` on `solve`. Turns a
   silent wrong answer into a clear error, and lets callers choose a viable structure
   instead of guessing.
3. **§3** — worklist frontier. The clearest algorithmic win, and the one that would make
   `hybsol` competitive with a sparse pivoted LU at scale.
4. **§9** — surface `fill_plan` / `operation_bound` in Python. Small, and it makes the
   block-shape choice an informed one.
