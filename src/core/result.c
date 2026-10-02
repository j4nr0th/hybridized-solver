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
    case HYBSOL_ERROR_SYSTEM_INVALID:
        return "block system is not valid";
    case HYBSOL_ERROR_NOT_DECOMPOSED:
        return "decomposition has not been factorized";
    case HYBSOL_ERROR_ALREADY_DECOMPOSED:
        return "decomposition has already been factorized";
    case HYBSOL_ERROR_INVALID_ORDERING:
        return "block ordering admits no factorization of this system";
    case HYBSOL_ERROR_INTERNAL:
        return "internal error";
    }
    return "unknown result code";
}
