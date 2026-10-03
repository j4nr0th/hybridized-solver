/**
 * @file src/python/opencl_module.c
 * The ``hybsol._mod_opencl`` extension: the Python face of the OpenCL backend.
 *
 * It links its own copy of the core, so the bindings' helpers reach it through
 * the capsule ``hybsol._mod`` publishes rather than by linking them. Raising,
 * which that capsule does not cover, is what is left to do here.
 */

#define PY_ARRAY_UNIQUE_SYMBOL _mod_opencl
#include "module.h"

#include "block_system_type.h"

#include <hybsol/opencl.h>

/** The table ``hybsol._mod`` publishes, read once on the first call that needs it. */
static hybsol_python_api_t *g_api = NULL;

/** The capsule ``hybsol._mod`` publishes the table in. */
static const char *const BACKEND_API_CAPSULE = "hybsol._mod.backend_api";

/**
 * Reject more arguments than the spec names, which cpyutl asserts on instead
 * of diagnosing. Returns 0, or -1 with a :exc:`TypeError` set.
 */
static int reject_extra_arguments(const char *const what, const Py_ssize_t nargs, PyObject *const kwnames,
                                  const Py_ssize_t accepted)
{
    const Py_ssize_t given = nargs + (kwnames != NULL ? PyTuple_GET_SIZE(kwnames) : 0);
    if (given <= accepted)
    {
        return 0;
    }
    PyErr_Format(PyExc_TypeError, "%s() takes at most %zd argument%s, but %zd were given.", what, accepted,
                 accepted == 1 ? "" : "s", given);
    return -1;
}

/**
 * Read the core module's backend API table, or leave an :exc:`ImportError`
 * saying why not. Returns 0 once the table is there.
 */
static int backend_api(const char *const what)
{
    if (g_api != NULL)
    {
        return 0;
    }

    PyObject *const core = PyImport_ImportModule("hybsol._mod");
    if (core == NULL)
    {
        PyErr_Clear();
        PyErr_Format(PyExc_ImportError,
                     "%s needs hybsol's core extension module, which could not be imported; the two halves "
                     "of this installation do not match.",
                     what);
        return -1;
    }

    PyObject *const capsule = PyObject_GetAttrString(core, "backend_api");
    Py_DECREF(core);
    if (capsule == NULL)
    {
        PyErr_Clear();
        PyErr_Format(PyExc_ImportError,
                     "%s needs hybsol's core extension module, which does not publish its backend API; the "
                     "two halves of this installation do not match.",
                     what);
        return -1;
    }

    void *const pointer = PyCapsule_GetPointer(capsule, BACKEND_API_CAPSULE);
    if (pointer == NULL)
    {
        Py_DECREF(capsule);
        return -1;
    }
    g_api = (hybsol_python_api_t *)pointer;
    Py_DECREF(capsule);
    return 0;
}

/** One attribute of ``hybsol._mod`` as a type, or ``NULL`` with an exception. */
static PyTypeObject *core_type(const char *const name)
{
    PyObject *const core = PyImport_ImportModule("hybsol._mod");
    if (core == NULL)
    {
        return NULL;
    }
    PyObject *const type = PyObject_GetAttrString(core, name);
    Py_DECREF(core);
    if (type == NULL)
    {
        return NULL;
    }
    if (!PyType_Check(type))
    {
        Py_DECREF(type);
        PyErr_Format(PyExc_TypeError, "hybsol._mod.%s is not a type.", name);
        return NULL;
    }
    return (PyTypeObject *)type; // New reference: the caller releases it.
}

/** Raise the ``hybsol`` exception named ``name`` with ``message``. */
static PyObject *raise_hybsol(const char *const name, PyObject *const message);

/** Raise ``hybsol.DeviceError``, with whatever detail the runtime last gave. */
static PyObject *raise_device_error(const char *const what, const hybsol_result_t res)
{
    const char *const detail = hybsol_opencl_last_error();
    if (detail == NULL || detail[0] == '\0')
    {
        return raise_hybsol("DeviceError", PyUnicode_FromFormat("%s: %s", what, hybsol_result_str(res)));
    }
    return raise_hybsol("DeviceError", PyUnicode_FromFormat("%s: %s (%s)", what, hybsol_result_str(res), detail));
}

/** Whether ``object`` is an instance of the core type named ``type_name``. */
static int is_core_type(PyObject *const object, const char *const type_name)
{
    PyTypeObject *const type = core_type(type_name);
    if (type == NULL)
    {
        return -1;
    }
    const int result = PyObject_IsInstance(object, (PyObject *)type);
    Py_DECREF(type);
    return result;
}

/** Raise the ``hybsol`` exception named ``name`` with ``message``. */
static PyObject *raise_hybsol(const char *const name, PyObject *const message)
{
    if (message == NULL)
    {
        return NULL;
    }
    PyObject *const hybsol_module = PyImport_ImportModule("hybsol");
    if (hybsol_module == NULL)
    {
        Py_DECREF(message);
        return NULL;
    }
    PyObject *const type = PyObject_GetAttrString(hybsol_module, name);
    Py_DECREF(hybsol_module);
    if (type == NULL)
    {
        Py_DECREF(message);
        return NULL;
    }
    PyErr_SetObject(type, message);
    Py_DECREF(message);
    Py_DECREF(type);
    return NULL;
}

/**
 * Raise whatever a result code means; the device codes carry the runtime's own
 * detail, which is where a kernel build failure becomes legible. Always
 * returns ``NULL``.
 */
static PyObject *raise_for_result(const hybsol_result_t res, const uint64_t failing_block)
{
    switch (res)
    {
    case HYBSOL_ERROR_SINGULAR:
        return raise_hybsol("SingularSystemError",
                            failing_block == UINT64_MAX
                                ? PyUnicode_FromFormat("decompose: %s", hybsol_result_str(res))
                                : PyUnicode_FromFormat("decompose: %s (block %llu)", hybsol_result_str(res),
                                                       (unsigned long long)failing_block));
    case HYBSOL_ERROR_NO_DEVICE:
    case HYBSOL_ERROR_DEVICE_CAPABILITY:
    case HYBSOL_ERROR_DEVICE:
        return raise_device_error("decompose", res);
    // The one code that is about the machine's memory rather than the data.
    case HYBSOL_ERROR_OUT_OF_MEMORY:
        return PyErr_NoMemory();
    default:
        PyErr_Format(PyExc_ValueError, "decompose: %s", hybsol_result_str(res));
        return NULL;
    }
}

/* ------------------------------------------------------------------------- */
/* Devices                                                                     */
/* ------------------------------------------------------------------------- */

PyDoc_STRVAR(opencl_device_count_docstring, "device_count() -> int\n"
                                            "\n"
                                            "How many devices the OpenCL runtime offers.\n"
                                            "\n"
                                            "Zero means there is no OpenCL runtime, no platform offers a\n"
                                            "device, or hybsol was built without the backend -- in every case\n"
                                            "``hybsol.opencl.decompose`` raises :exc:`hybsol.DeviceError`.\n"
                                            "\n"
                                            "Returns\n"
                                            "-------\n"
                                            "int\n"
                                            "    The number of devices, GPUs first.\n");

static PyObject *opencl_device_count(PyObject *const Py_UNUSED(self), PyObject *const Py_UNUSED(args))
{
    return PyLong_FromUnsignedLongLong(hybsol_opencl_device_count());
}

PyDoc_STRVAR(opencl_device_info_docstring, "device_info(index: int) -> dict\n"
                                           "\n"
                                           "What the OpenCL runtime reports about one device.\n"
                                           "\n"
                                           "Parameters\n"
                                           "----------\n"
                                           "index : int\n"
                                           "    A device index, below ``device_count()``. GPUs come first, so\n"
                                           "    ``0`` is a GPU wherever the machine has one.\n"
                                           "\n"
                                           "Returns\n"
                                           "-------\n"
                                           "dict\n"
                                           "    Keys ``index``, ``name``, ``vendor``, ``uuid``, ``kind``\n"
                                           "    (``\"gpu\"``, ``\"cpu\"``, ``\"accelerator\"`` or ``\"unknown\"``),\n"
                                           "    ``double`` (whether it can do double arithmetic, which every\n"
                                           "    decomposition needs), ``memory`` and ``compute_units``.\n"
                                           "\n"
                                           "Raises\n"
                                           "------\n"
                                           "hybsol.DeviceError\n"
                                           "    No device has that index.\n"
                                           "TypeError\n"
                                           "    ``index`` is not an integer, or was left out.\n"
                                           "ValueError\n"
                                           "    ``index`` is negative.\n");

static PyObject *opencl_device_info(PyObject *const Py_UNUSED(self), PyObject *const *const args,
                                    const Py_ssize_t nargs, PyObject *const kwnames)
{
    if (reject_extra_arguments("device_info", nargs, kwnames, 1) < 0)
    {
        return NULL;
    }
    Py_ssize_t index = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_SSIZE, .p_val = &index, .kwname = "index"},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }
    if (index < 0)
    {
        PyErr_SetString(PyExc_ValueError, "The device index must be non-negative.");
        return NULL;
    }

    hybsol_opencl_device_info_t info;
    if (hybsol_opencl_device_info((uint64_t)index, &info) != HYBSOL_SUCCESS)
    {
        return raise_hybsol("DeviceError", PyUnicode_FromFormat("No OpenCL device has index %zd.", index));
    }

    static const char *const kind_names[] = {"unknown", "cpu", "gpu", "accelerator"};
    const unsigned kind = (unsigned)info.kind < 4u ? (unsigned)info.kind : 0u;

    return cpyutl_output_create_check(
        CPYOUT_TYPE_DICT,
        (const cpyutl_output_t[]){
            {.type = CPYOUT_TYPE_PYINT, .value_int = index, .name = "index"},
            {.type = CPYOUT_TYPE_PYSTRING, .value_str = info.name, .name = "name"},
            {.type = CPYOUT_TYPE_PYSTRING, .value_str = info.vendor, .name = "vendor"},
            {.type = CPYOUT_TYPE_PYSTRING, .value_str = info.uuid, .name = "uuid"},
            {.type = CPYOUT_TYPE_PYSTRING, .value_str = kind_names[kind], .name = "kind"},
            {.type = CPYOUT_TYPE_PYBOOL, .value_bool = info.supports_double != 0, .name = "double"},
            {.type = CPYOUT_TYPE_PYINT, .value_int = (Py_ssize_t)info.global_memory_bytes, .name = "memory"},
            {.type = CPYOUT_TYPE_PYINT, .value_int = (Py_ssize_t)info.compute_units, .name = "compute_units"},
            {},
        });
}

PyDoc_STRVAR(opencl_decompose_docstring, "decompose(system: hybsol.BlockSystem, *, device: int = 0,\n"
                                         "           precision: hybsol.Precision | None = None,\n"
                                         "           n_threads: int = 0) -> hybsol.Decomposition\n"
                                         "\n"
                                         "Factorize a system with its factors on an OpenCL device.\n"
                                         "\n"
                                         "The symbolic walk and the assembly stay on the CPU; the\n"
                                         "factorization, the solve and the replay run as device kernels,\n"
                                         "and the factors never come back to the host. The result is used\n"
                                         "exactly as any other decomposition; only its ``device`` differs.\n"
                                         "\n"
                                         "Parameters\n"
                                         "----------\n"
                                         "system : hybsol.BlockSystem\n"
                                         "    The system to factorize. It is only read.\n"
                                         "device : int, default: 0\n"
                                         "    Which device of ``devices()`` to factorize on.\n"
                                         "precision : hybsol.Precision, optional\n"
                                         "    The precision of the factors, which need not match the\n"
                                         "    system's: a double system can produce single factors, the\n"
                                         "    fast path on a GPU with weak double arithmetic. Defaults\n"
                                         "    to the system's own precision.\n"
                                         "n_threads : int, default: 0\n"
                                         "    Accepted for symmetry with :meth:`hybsol.BlockSystem.decompose`\n"
                                         "    and ignored: the device schedules its own work.\n"
                                         "\n"
                                         "Returns\n"
                                         "-------\n"
                                         "hybsol.Decomposition\n"
                                         "    The factorized system, with its factors on the device.\n"
                                         "\n"
                                         "Raises\n"
                                         "------\n"
                                         "hybsol.DeviceError\n"
                                         "    No device has that index, it cannot do double arithmetic, or\n"
                                         "    the device runtime failed.\n"
                                         "ValueError\n"
                                         "    The system does not satisfy the solver's structural\n"
                                         "    assumptions; ``device`` or ``n_threads`` is negative; or\n"
                                         "    ``precision`` is not a :class:`hybsol.Precision` member.\n"
                                         "TypeError\n"
                                         "    ``system`` is not a :class:`hybsol.BlockSystem`, or an\n"
                                         "    argument is of the wrong type or shape.\n"
                                         "RuntimeError\n"
                                         "    ``system`` is a :class:`hybsol.BlockSystem` that was never\n"
                                         "    initialized.\n"
                                         "MemoryError\n"
                                         "    The device could not hold the factors, or the host could not\n"
                                         "    allocate the frame.\n"
                                         "hybsol.SingularSystemError\n"
                                         "    A diagonal block is singular, so the LU factorization hit a\n"
                                         "    zero pivot. The message names the block.\n");

static PyObject *opencl_decompose(PyObject *const Py_UNUSED(self), PyObject *const *const args, const Py_ssize_t nargs,
                                  PyObject *kwnames)
{
    if (reject_extra_arguments("decompose", nargs, kwnames, 4) < 0)
    {
        return NULL;
    }
    if (backend_api("hybsol.opencl") < 0)
    {
        return NULL;
    }

    PyObject *py_system = NULL;
    Py_ssize_t device_index = 0;
    PyObject *py_precision = Py_None;
    Py_ssize_t n_threads = 0;
    if (parse_arguments_check(
            (cpyutl_argument_t[]){
                {.type = CPYARG_TYPE_PYTHON, .p_val = &py_system},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &device_index, .kwname = "device", .optional = 1, .kw_only = 1},
                {.type = CPYARG_TYPE_PYTHON,
                 .p_val = &py_precision,
                 .kwname = "precision",
                 .optional = 1,
                 .kw_only = 1},
                {.type = CPYARG_TYPE_SSIZE, .p_val = &n_threads, .kwname = "n_threads", .optional = 1, .kw_only = 1},
                {},
            },
            args, nargs, kwnames) < 0)
    {
        return NULL;
    }

    const int is_system = is_core_type(py_system, "BlockSystem");
    if (is_system < 0)
    {
        return NULL;
    }
    if (is_system == 0)
    {
        PyErr_Format(PyExc_TypeError, "decompose() needs a hybsol.BlockSystem, got %R.", Py_TYPE(py_system));
        return NULL;
    }

    if (g_api->check_n_threads(n_threads) < 0)
    {
        return NULL;
    }
    if (device_index < 0)
    {
        PyErr_SetString(PyExc_ValueError, "The device index must be non-negative.");
        return NULL;
    }

    // The object was just checked to be one, and both modules read the same header.
    block_system_object *const system = (block_system_object *)py_system;
    if (system->system == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "decompose(): the BlockSystem was never initialized.");
        return NULL;
    }

    hybsol_precision_t precision;
    if (py_precision == Py_None)
    {
        precision = hybsol_system_precision(system->system);
    }
    else if (g_api->parse_precision(py_precision, &precision) < 0)
    {
        return NULL;
    }

    // The core asserts validity rather than reporting it, and raising is the only way out.
    if (g_api->require_valid_system(system->system, "decompose") < 0)
    {
        return NULL;
    }

    uint64_t failing_block = UINT64_MAX;
    hybsol_opencl_device_t *device = NULL;
    hybsol_elimination_t *graph = NULL;
    hybsol_decomposition_t *dec = NULL;
    hybsol_result_t res;

    Py_BEGIN_ALLOW_THREADS;
    res = hybsol_opencl_device_acquire((uint64_t)device_index, &device);
    if (res == HYBSOL_SUCCESS)
    {
        res = hybsol_elimination_create(system->system, &graph, &failing_block);
    }
    if (res == HYBSOL_SUCCESS)
    {
        res = hybsol_opencl_decomposition_create_on_device(system->system, graph, precision, device, &dec);
        // The decomposition copied the schedule it needs, so the graph is done.
        hybsol_elimination_destroy(graph);
        graph = NULL;
    }
    if (res == HYBSOL_SUCCESS)
    {
        res = hybsol_decomposition_factorize(dec, (uint64_t)n_threads);
    }
    Py_END_ALLOW_THREADS;

    hybsol_opencl_device_release(device);
    if (res != HYBSOL_SUCCESS)
    {
        const uint64_t failing = dec != NULL ? hybsol_decomposition_failing_block(dec) : failing_block;
        hybsol_elimination_destroy(graph);
        hybsol_decomposition_destroy(dec);
        return raise_for_result(res, failing);
    }

    PyTypeObject *const type = core_type("Decomposition");
    if (type == NULL)
    {
        hybsol_decomposition_destroy(dec);
        return NULL;
    }
    PyObject *const out = g_api->decomposition_alloc(type, dec);
    Py_DECREF(type);
    if (out == NULL)
    {
        hybsol_decomposition_destroy(dec);
        return NULL;
    }
    return out;
}

/* ------------------------------------------------------------------------- */
/* Module definition                                                           */
/* ------------------------------------------------------------------------- */

static PyMethodDef opencl_methods[] = {
    {"device_count", (void *)opencl_device_count, METH_NOARGS, opencl_device_count_docstring},
    {"device_info", (void *)opencl_device_info, METH_FASTCALL | METH_KEYWORDS, opencl_device_info_docstring},
    {"decompose", (void *)opencl_decompose, METH_FASTCALL | METH_KEYWORDS, opencl_decompose_docstring},
    {NULL, NULL, 0, NULL},
};

PyDoc_STRVAR(opencl_module_docstring, "The OpenCL backend, as an extension module.\n"
                                      "\n"
                                      "Reached through the ``hybsol.opencl`` wrapper, which is what a user\n"
                                      "imports; this module is its implementation.");

static struct PyModuleDef opencl_module_def = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "hybsol._mod_opencl",
    .m_doc = opencl_module_docstring,
    .m_size = 0,
    .m_slots =
        (PyModuleDef_Slot[]){
            {.slot = Py_mod_exec, .value = NULL},
            {.slot = Py_mod_multiple_interpreters, .value = Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED},
            {},
        },
};

static int opencl_module_exec(PyObject *const mod)
{
    return PyModule_AddFunctions(mod, opencl_methods);
}

PyMODINIT_FUNC PyInit__mod_opencl(void)
{
    opencl_module_def.m_slots[0].value = opencl_module_exec;
    return PyModuleDef_Init(&opencl_module_def);
}
