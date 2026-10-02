/**
 * @file core/result.c
 * Human-readable descriptions of the fallible API's result codes.
 */

#include "internal.h"

const char *hybsol_result_str(const hybsol_result_t result)
{
    switch (result)
    {
    case HYBSOL_SUCCESS:
        return "success";
    case HYBSOL_ERROR_OUT_OF_MEMORY:
        return "out of memory";
    case HYBSOL_ERROR_EMPTY_ROW:
        return "row has no entries";
    case HYBSOL_ERROR_NO_MORE_COLUMNS:
        return "row has no entries after the given column";
    case HYBSOL_ERROR_MAX_COLORS:
        return "maximum number of colors exceeded";
    case HYBSOL_ERROR_SINGULAR:
        return "zero pivot in LU decomposition";
    case HYBSOL_ERROR_INVALID_ORDERING:
        return "block ordering admits no factorization of this system";
    case HYBSOL_ERROR_NO_DEVICE:
        return "no such device";
    case HYBSOL_ERROR_DEVICE_CAPABILITY:
        return "device lacks a required capability";
    case HYBSOL_ERROR_DEVICE:
        return "device runtime failure";
    }
    return "unknown result code";
}
