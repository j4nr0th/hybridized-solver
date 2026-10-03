"""Factorize on a GPU with the OpenCL backend.

Everything a backend does has to stay behind an import, or every program that
merely mentions it pays for OpenCL: the runtime is loaded, the kernels are
built, the context is created. So ``import hybsol`` never touches it, and the
backend arrives only when a program asks for it:

.. code-block:: python

    import hybsol
    import hybsol.opencl          # this line is the opt-in

An installation built without the backend has no ``hybsol.opencl`` at all, and
the import says so. This example runs either way -- it reports that there is
nothing to run on, and returns -- because a machine without a GPU is a normal
machine, not a broken one.

The backend is worth reaching for when the system is large: the factorization
is one kernel launch per pass of the schedule, with the work inside a pass
spread over the device. The factors then stay on the device; the walk, the
assembly and every host-side query keep running on the CPU, and the
decomposition that comes back is used like any other, :meth:`solve
<hybsol.Decomposition.solve>` included.

Precision is worth a thought on a consumer GPU. The factors are usually the
heaviest thing in a run, and a GPU's single precision is far quicker than its
double; since a decomposition's precision need not match the system's, a
double system can be factorized in single. What that buys is speed, and what it
costs is accuracy with a floor at roughly ``eps_single * cond``: refinement
corrects a solve against the residual, but it cannot correct factors that were
narrow to begin with. A program that needs a smaller residual than the factors
allow has to factorize in double.
"""

import hybsol
import hybsol.opencl as ocl
import numpy as np

found = ocl.devices()

if not found:
    print("no OpenCL device: nothing to do here")
    raise SystemExit

for device in found:
    print(
        f"device {device['index']}: {device['name']} ({device['kind']}), "
        f"{device['double'] and 'can do double arithmetic' or 'single precision only'}, "
        f"{device['memory'] / 2**30:.1f} GiB"
    )

# The first device that can do double arithmetic; every one of these can take a
# decomposition, but the solver's vectors are double whatever the factors are.
device = next(info for info in found if info["double"])

# A block-sparse system, built the ordinary way: the backend does not change
# how a system is assembled.
rng = np.random.default_rng(23095)
n_blocks, block_size = 120, 6
eye = np.eye(block_size)
system = hybsol.BlockSystem(*([block_size] * n_blocks), precision=hybsol.Precision.SINGLE)
for row in range(n_blocks):
    # A diagonal block like a stiffness matrix's: dominant, and well conditioned.
    system.add_block(
        row, row, 2 * block_size * eye + rng.random((block_size, block_size)) * 0.05
    )
    # A band of couplings, so the factorization has real work to do.
    for col in (row + 1, row + 2):
        if col < n_blocks:
            coupling = rng.random((block_size, block_size)) * 0.05
            system.add_block(row, col, coupling)
            system.add_block(col, row, coupling.T.copy())

rhs = np.ones(n_blocks * block_size)

# The CPU path, for comparison. Both return a Decomposition; only one of them
# keeps its factors on the host.
cpu = system.decompose(n_threads=1)
cpu_solution = cpu.solve(rhs)

# The device path: the same walk, the same arithmetic, the factors uploaded
# instead of allocated here. Single-precision factors, since the device
# factorization is what we came for.
gpu = ocl.decompose(system, device=device["index"], precision=hybsol.Precision.SINGLE)
print(f"factors on device {gpu.device}, operations {gpu.n_operations}")
gpu_solution = gpu.solve(rhs)


# Both factorizations are of the same system in the same precision, so the
# residual against the matrix itself is what says whether they agree.
def residual(solution: np.ndarray) -> float:
    """How far a solution sits from solving the system, in the max norm."""
    return float(np.abs(system.matvec(solution) - rhs).max())


print(f"device: {device['name']}")
print(f"  max |cpu - opencl| {np.abs(cpu_solution - gpu_solution).max():.3e}")
print(f"  residual cpu       {residual(cpu_solution):.3e}")
print(f"  residual opencl    {residual(gpu_solution):.3e}")

# Refinement measures the residual against the system -- which stores doubles
# -- and corrects against the factors it has. That cannot rescue narrow
# factors: the floor is roughly eps_single * cond, and it is the factors
# themselves that carry the error. Asking for more than the factors can give
# raises rather than pretending, so the tolerance below is one they can reach.
refined = hybsol.refined_solve(system, gpu, rhs, tolerance=1e-6, max_iterations=5)
print(f"  after refinement   {residual(refined):.3e}  (floored by the factors)")
