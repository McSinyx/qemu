#include "e9-ranges.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

/* Parse one canonical E9 interval token "0x<hex>-0x<hex>".  The token
 * must be consumed entirely; start must be < end; the load_bias addition
 * is checked for overflow.  Returns 0 on success, -1 on malformed input. */
static int parse_e9_range_token(const char *token, uintptr_t load_bias,
                                e9_exclude_region *region) {
    if (token[0] != '0' || (token[1] != 'x' && token[1] != 'X')) {
        return -1;
    }
    const char *dash = strchr(token + 2, '-');
    if (dash == NULL) {
        return -1;
    }
    if (dash[1] != '0' || (dash[2] != 'x' && dash[2] != 'X')) {
        return -1;
    }
    char *end_start = NULL;
    char *end_end = NULL;
    errno = 0;
    unsigned long long start = strtoull(token + 2, &end_start, 16);
    if (errno != 0 || end_start != dash) {
        return -1;
    }
    errno = 0;
    unsigned long long end = strtoull(dash + 1, &end_end, 16);
    if (errno != 0 || end_end == dash + 1 || *end_end != '\0') {
        return -1;
    }
    if (start >= end) {
        return -1;
    }
    if (start > UINTPTR_MAX - load_bias || end > UINTPTR_MAX - load_bias) {
        return -1;
    }
    region->start = (uintptr_t)start + load_bias;
    region->end = (uintptr_t)end + load_bias;
    return 0;
}

int e9_parse_exclude_ranges(const char *value, uintptr_t load_bias,
                            e9_exclude_region **regions, size_t *len,
                            size_t *cap) {
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    char *copy = strdup(value);
    if (copy == NULL) {
        return -1;
    }
    char *saveptr = NULL;
    size_t parsed = 0;
    for (char *token = strtok_r(copy, ",", &saveptr); token != NULL;
         token = strtok_r(NULL, ",", &saveptr)) {
        e9_exclude_region region;
        if (parse_e9_range_token(token, load_bias, &region) != 0) {
            free(copy);
            return -1;
        }
        if (parsed == *cap) {
            size_t new_cap = *cap ? *cap * 2 : 8;
            e9_exclude_region *grown = realloc(
                *regions, new_cap * sizeof(**regions));
            if (grown == NULL) {
                free(copy);
                return -1;
            }
            *regions = grown;
            *cap = new_cap;
        }
        (*regions)[parsed++] = region;
    }
    free(copy);
    *len = parsed;
    return 0;
}

bool e9_is_in_exclude_region(const e9_exclude_region *regions, size_t len,
                             uintptr_t pc) {
    for (size_t i = 0; i < len; i++) {
        if (pc >= regions[i].start && pc < regions[i].end) {
            return true;
        }
    }
    return false;
}
