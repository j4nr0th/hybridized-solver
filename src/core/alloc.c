/**
 * @file core/alloc.c
 * Result descriptions, the default allocator and assertion reporting.
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
        return "system has not been decomposed";
    case HYBSOL_ERROR_ALREADY_DECOMPOSED:
        return "system has already been decomposed";
    case HYBSOL_ERROR_INTERNAL:
        return "internal error";
    }
    return "unknown result code";
}

static void *default_malloc(const size_t size)
{
    return malloc(size);
}

static void *default_realloc(void *const ptr, const size_t size)
{
    return realloc(ptr, size);
}

static void default_free(void *const ptr)
{
    free(ptr);
}

hybsol_allocator_t hybsol_current_allocator = {
    .malloc_fn = default_malloc,
    .realloc_fn = default_realloc,
    .free_fn = default_free,
};

void hybsol_set_allocator(const hybsol_allocator_t *const allocator)
{
    if (allocator == NULL)
    {
        hybsol_current_allocator = (hybsol_allocator_t){
            .malloc_fn = default_malloc,
            .realloc_fn = default_realloc,
            .free_fn = default_free,
        };
        return;
    }
    hybsol_current_allocator = *allocator;
}

const hybsol_allocator_t *hybsol_get_allocator(void)
{
    return &hybsol_current_allocator;
}

#if HYBSOL_ENABLE_ASSERTS
void hybsol_assert_fail(const char *const file, const int line, const char *const func, const char *const cond,
                        const char *const fmt, ...)
{
    va_list args;
    fprintf(stderr, "%s:%d (%s): assertion \"%s\" failed", file, line, func, cond);
    if (fmt != NULL && *fmt != '\0')
    {
        fputc(':', stderr);
        fputc(' ', stderr);
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
    }
    fputc('\n', stderr);
    abort();
}
#endif
