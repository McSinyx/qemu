#ifndef BINRADAR_E9_RANGES_H
#define BINRADAR_E9_RANGES_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Exact E9 exclusion intervals (loader + RESERVE + TRAMPOLINE maps of the
 * executing artifact), parsed from E9_EXCLUDE_RANGES.  Half-open
 * [start, end).  This module is plain C (no glib, no QEMU) so the parser
 * and membership checks are unit-testable without the tracer. */

typedef struct e9_exclude_region {
    uintptr_t start;
    uintptr_t end;
} e9_exclude_region;

/* Parse a canonical comma-separated interval list into *regions, growing
 * the caller's storage through realloc (*cap is updated).  Missing or
 * empty input yields 0 intervals.  On the first malformed token the
 * function returns -1 and *len is left untouched, so a partial list is
 * never observable.  The input string is never modified.
 *
 * Grammar per token: 0x<hex>-0x<hex> with no trailing data, start < end,
 * and checked load_bias addition (overflow is malformed). */
int e9_parse_exclude_ranges(const char *value, uintptr_t load_bias,
                            e9_exclude_region **regions, size_t *len,
                            size_t *cap);

/* Half-open membership: true iff start <= pc < end for some interval. */
bool e9_is_in_exclude_region(const e9_exclude_region *regions, size_t len,
                             uintptr_t pc);

#endif /* BINRADAR_E9_RANGES_H */
