/* B1: bounded, facts-only OSPREY mutation advice. */
#include "qemu/osdep.h"
#include "osprey.h"
#include "osprey-internal.h"
#include <inttypes.h>
#include <limits.h>
#include <string.h>

void log_msg(const char *fmt, ...);

typedef int (*MutationCompare)(const void *, const void *);

typedef struct MutationTargetSummary {
    OspreyAddress target;
    uint64_t extent;
    uint8_t kind;
    uint8_t reserved[3];
    OspreyMutationAbstentionReason reason;
} MutationTargetSummary;

typedef struct MutationWork {
    OspreyPointsToFact *points;        /* cell,target order */
    OspreyPointsToFact *target_points; /* target,cell order */
    MutationTargetSummary *summaries;  /* target order */
    OspreyBaseFact *bases;
    OspreyMallocFact *allocs;
    OspreyMayArrayFact *arrays;
    uint8_t *scratch;
    uint32_t summary_count;
    uint64_t projected_bytes;
    uint64_t work_units;
} MutationWork;

static int cmp_u64(uint64_t a, uint64_t b) { return a < b ? -1 : a != b; }
static int cmp_i64(int64_t a, int64_t b) { return a < b ? -1 : a != b; }

static int region_cmp(const OspreyRegionId *a, const OspreyRegionId *b)
{
    int c = cmp_u64((uint64_t)a->kind, (uint64_t)b->kind);
    if (c != 0) return c;
    c = cmp_u64(a->code_image_id, b->code_image_id);
    return c != 0 ? c : cmp_u64(a->site_offset, b->site_offset);
}

static int address_cmp(const OspreyAddress *a, const OspreyAddress *b)
{
    int c = region_cmp(&a->region, &b->region);
    return c != 0 ? c : cmp_i64(a->offset, b->offset);
}

static int chunk_cmp(const OspreyChunk *a, const OspreyChunk *b)
{
    int c = address_cmp(&a->address, &b->address);
    return c != 0 ? c : cmp_u64(a->size, b->size);
}

static bool region_eq(const OspreyRegionId *a, const OspreyRegionId *b)
{ return region_cmp(a, b) == 0; }
static bool address_eq(const OspreyAddress *a, const OspreyAddress *b)
{ return address_cmp(a, b) == 0; }
static bool chunk_eq(const OspreyChunk *a, const OspreyChunk *b)
{ return chunk_cmp(a, b) == 0; }

static bool region_valid(const OspreyRegionId *r)
{
    return r != NULL && r->kind >= OSPREY_REGION_GLOBAL &&
           r->kind <= OSPREY_REGION_STACK_FUNCTION;
}

static bool chunk_valid(const OspreyChunk *c)
{
    uint64_t delta;
    if (c == NULL || !region_valid(&c->address.region) || c->size == 0) {
        return false;
    }
    delta = c->size - 1;
    if (delta > (uint64_t)INT64_MAX) return false;
    if (c->address.offset >= 0) {
        return (uint64_t)c->address.offset <= (uint64_t)INT64_MAX - delta;
    }
    return delta < (uint64_t)(-(c->address.offset + 1)) + 1;
}

static bool add_u64(uint64_t *value, uint64_t add)
{
    if (value == NULL || add > UINT64_MAX - *value) return false;
    *value += add;
    return true;
}

static bool mul_u64(uint64_t a, uint64_t b, uint64_t *out)
{ return osprey_check_mul(a, b, out); }

static bool bytes_for(uint64_t count, size_t size, uint64_t *out)
{ return mul_u64(count, (uint64_t)size, out); }

static int points_cmp(const void *ap, const void *bp)
{
    const OspreyPointsToFact *a = ap, *b = bp;
    int c = chunk_cmp(&a->pointer_chunk, &b->pointer_chunk);
    return c != 0 ? c : address_cmp(&a->target, &b->target);
}

static int target_points_cmp(const void *ap, const void *bp)
{
    const OspreyPointsToFact *a = ap, *b = bp;
    int c = address_cmp(&a->target, &b->target);
    return c != 0 ? c : chunk_cmp(&a->pointer_chunk, &b->pointer_chunk);
}

static int bases_cmp(const void *ap, const void *bp)
{
    const OspreyBaseFact *a = ap, *b = bp;
    int c = address_cmp(&a->base, &b->base);
    if (c != 0) return c;
    c = chunk_cmp(&a->chunk, &b->chunk);
    if (c != 0) return c;
    c = cmp_u64(a->pc, b->pc);
    if (c != 0) return c;
    c = cmp_u64(a->prov_object_id, b->prov_object_id);
    return c != 0 ? c : cmp_u64(a->prov_generation, b->prov_generation);
}

static int allocs_cmp(const void *ap, const void *bp)
{
    const OspreyMallocFact *a = ap, *b = bp;
    int c = cmp_u64(a->site_pc, b->site_pc);
    return c != 0 ? c : cmp_u64(a->requested_size, b->requested_size);
}

static int arrays_cmp(const void *ap, const void *bp)
{
    const OspreyMayArrayFact *a = ap, *b = bp;
    int c = address_cmp(&a->start, &b->start);
    if (c != 0) return c;
    c = cmp_u64(a->element_count, b->element_count);
    if (c != 0) return c;
    c = cmp_u64(a->element_size, b->element_size);
    return c != 0 ? c : cmp_u64(a->evidence_kind, b->evidence_kind);
}

static uint64_t sort_levels(size_t n)
{
    uint64_t levels = 0;
    size_t width = 1;
    while (width < n) {
        levels++;
        if (width > SIZE_MAX / 2) break;
        width *= 2;
    }
    return levels;
}

static bool charge_sort(uint64_t *work, size_t n)
{
    uint64_t comparisons, copies;
    if (!mul_u64(n, sort_levels(n), &comparisons) ||
        !mul_u64(comparisons, 2, &copies)) return false;
    return add_u64(work, copies);
}

static void merge_pass(uint8_t *src, uint8_t *dst, size_t n, size_t width,
                       size_t size, MutationCompare cmp)
{
    for (size_t left = 0; left < n; ) {
        size_t mid = left + width < n ? left + width : n;
        size_t right = mid + width < n ? mid + width : n;
        size_t i = left, j = mid, k = left;
        while (i < mid && j < right) {
            const void *a = src + i * size, *b = src + j * size;
            if (cmp(a, b) <= 0) memcpy(dst + k++ * size, a, size), i++;
            else memcpy(dst + k++ * size, b, size), j++;
        }
        while (i < mid) memcpy(dst + k++ * size, src + i++ * size, size);
        while (j < right) memcpy(dst + k++ * size, src + j++ * size, size);
        if (right == n) break;
        left = right;
    }
}

static bool stable_sort(void *array, size_t n, size_t size, void *scratch,
                        MutationCompare cmp)
{
    uint8_t *src = array, *dst = scratch;
    bool scratch_source = false;
    if (n < 2) return true;
    if (array == NULL || scratch == NULL || size == 0 || cmp == NULL) {
        return false;
    }
    for (size_t width = 1; width < n; ) {
        merge_pass(src, dst, n, width, size, cmp);
        uint8_t *tmp = src; src = dst; dst = tmp;
        scratch_source = !scratch_source;
        if (width > SIZE_MAX / 2) break;
        width *= 2;
    }
    if (scratch_source) memcpy(array, src, n * size);
    return true;
}

static bool input_count(const OspreyContext *ctx, uint64_t *out)
{
    const GArray *a[7] = { ctx->access_facts, ctx->base_facts,
        ctx->copy_facts, ctx->points_facts, ctx->alloc_facts,
        ctx->mayarray_facts, ctx->region_instances };
    uint64_t total = 0;
    if (ctx == NULL || out == NULL) return false;
    for (size_t i = 0; i < G_N_ELEMENTS(a); i++) {
        if (a[i] == NULL || !add_u64(&total, a[i]->len)) return false;
    }
    *out = total;
    return true;
}

static bool validate_input(const OspreyContext *ctx)
{
    if (ctx == NULL || ctx->access_facts == NULL || ctx->base_facts == NULL ||
        ctx->copy_facts == NULL || ctx->points_facts == NULL ||
        ctx->alloc_facts == NULL || ctx->mayarray_facts == NULL ||
        ctx->region_instances == NULL) return false;
    for (guint i = 0; i < ctx->access_facts->len; i++) {
        const OspreyAccessFact *f = &g_array_index(ctx->access_facts,
                                                    OspreyAccessFact, i);
        if (!chunk_valid(&f->chunk) || f->sample_support == 0) return false;
    }
    for (guint i = 0; i < ctx->base_facts->len; i++) {
        const OspreyBaseFact *f = &g_array_index(ctx->base_facts,
                                                  OspreyBaseFact, i);
        if (!chunk_valid(&f->chunk) || !region_valid(&f->base.region) ||
            f->sample_support == 0) return false;
    }
    for (guint i = 0; i < ctx->copy_facts->len; i++) {
        const OspreyCopyFact *f = &g_array_index(ctx->copy_facts,
                                                  OspreyCopyFact, i);
        if (!chunk_valid(&f->source) || !chunk_valid(&f->destination) ||
            f->sample_support == 0) return false;
    }
    for (guint i = 0; i < ctx->points_facts->len; i++) {
        const OspreyPointsToFact *f = &g_array_index(ctx->points_facts,
                                                      OspreyPointsToFact, i);
        if (!chunk_valid(&f->pointer_chunk) ||
            f->pointer_chunk.size != sizeof(target_ulong) ||
            !region_valid(&f->target.region) || f->sample_support == 0) {
            return false;
        }
    }
    for (guint i = 0; i < ctx->alloc_facts->len; i++) {
        const OspreyMallocFact *f = &g_array_index(ctx->alloc_facts,
                                                    OspreyMallocFact, i);
        if (f->requested_size > (uint64_t)INT64_MAX ||
            f->sample_support == 0) return false;
    }
    for (guint i = 0; i < ctx->mayarray_facts->len; i++) {
        const OspreyMayArrayFact *f = &g_array_index(ctx->mayarray_facts,
                                                      OspreyMayArrayFact, i);
        uint64_t product;
        if (!region_valid(&f->start.region) || f->element_count == 0 ||
            f->element_size == 0 || f->evidence_kind !=
            OSPREY_MAY_ARRAY_CALLOC_GEOMETRY || f->sample_support == 0 ||
            !mul_u64(f->element_count, f->element_size, &product) ||
            product > (uint64_t)INT64_MAX) return false;
    }
    for (guint i = 0; i < ctx->region_instances->len; i++) {
        const OspreyRegionInstance *f = &g_array_index(ctx->region_instances,
                                                        OspreyRegionInstance, i);
        if (!region_valid(&f->region) || f->raw_min > f->raw_max ||
            f->sample_support == 0) return false;
    }
    return true;
}

static bool prepare_work(OspreyContext *ctx, MutationWork *w,
                         uint64_t input, size_t *points_out,
                         size_t *bases_out, size_t *allocs_out,
                         size_t *arrays_out)
{
    size_t np = ctx->points_facts->len, nb = ctx->base_facts->len;
    size_t na = ctx->alloc_facts->len, nm = ctx->mayarray_facts->len;
    uint64_t bytes = 0, x, scratch = 0, entries;
    memset(w, 0, sizeof(*w));
    if (ctx->config.max_mutation_input != 0 &&
        input > ctx->config.max_mutation_input) return false;
    if (!bytes_for(np, sizeof(OspreyPointsToFact), &x) || !add_u64(&bytes, x) ||
        !add_u64(&bytes, x) || !bytes_for(np, sizeof(MutationTargetSummary), &x) ||
        !add_u64(&bytes, x) || !bytes_for(nb, sizeof(OspreyBaseFact), &x) ||
        !add_u64(&bytes, x) || !bytes_for(na, sizeof(OspreyMallocFact), &x) ||
        !add_u64(&bytes, x) || !bytes_for(nm, sizeof(OspreyMayArrayFact), &x) ||
        !add_u64(&bytes, x) || !bytes_for(np, sizeof(OspreyMutationEntry),
                                           &entries) ||
        !bytes_for(np, sizeof(OspreyPointsToFact), &scratch)) return false;
    if (!bytes_for(nb, sizeof(OspreyBaseFact), &x)) return false;
    if (x > scratch) scratch = x;
    if (!bytes_for(na, sizeof(OspreyMallocFact), &x)) return false;
    if (x > scratch) scratch = x;
    if (!bytes_for(nm, sizeof(OspreyMayArrayFact), &x)) return false;
    if (x > scratch) scratch = x;
    if (!add_u64(&bytes, scratch) || !add_u64(&bytes, entries) ||
        !add_u64(&bytes, sizeof(OspreyMutationModel)) || bytes > SIZE_MAX ||
        scratch > SIZE_MAX || entries > SIZE_MAX ||
        (ctx->config.max_mutation_bytes != 0 &&
         bytes > ctx->config.max_mutation_bytes)) return false;

    w->points = np == 0 ? NULL : g_try_malloc0(np * sizeof(*w->points));
    w->target_points = np == 0 ? NULL :
        g_try_malloc0(np * sizeof(*w->target_points));
    w->summaries = np == 0 ? NULL :
        g_try_malloc0(np * sizeof(*w->summaries));
    w->bases = nb == 0 ? NULL : g_try_malloc0(nb * sizeof(*w->bases));
    w->allocs = na == 0 ? NULL : g_try_malloc0(na * sizeof(*w->allocs));
    w->arrays = nm == 0 ? NULL : g_try_malloc0(nm * sizeof(*w->arrays));
    w->scratch = scratch == 0 ? NULL : g_try_malloc((size_t)scratch);
    w->projected_bytes = bytes;
    if ((np && (!w->points || !w->target_points || !w->summaries)) ||
        (nb && !w->bases) || (na && !w->allocs) || (nm && !w->arrays) ||
        (scratch && !w->scratch)) return false;
    if (np) {
        memcpy(w->points, ctx->points_facts->data, np * sizeof(*w->points));
        memcpy(w->target_points, ctx->points_facts->data,
               np * sizeof(*w->target_points));
    }
    if (nb) memcpy(w->bases, ctx->base_facts->data, nb * sizeof(*w->bases));
    if (na) memcpy(w->allocs, ctx->alloc_facts->data,
                   na * sizeof(*w->allocs));
    if (nm) memcpy(w->arrays, ctx->mayarray_facts->data,
                   nm * sizeof(*w->arrays));
    w->work_units = input;
    if (!charge_sort(&w->work_units, np) || !charge_sort(&w->work_units, np) ||
        !charge_sort(&w->work_units, nb) || !charge_sort(&w->work_units, na) ||
        !charge_sort(&w->work_units, nm) || !add_u64(&w->work_units, np) ||
        !add_u64(&w->work_units, np) || !add_u64(&w->work_units, nb) ||
        !add_u64(&w->work_units, na) || !add_u64(&w->work_units, nm) ||
        (ctx->config.max_mutation_work != 0 &&
         w->work_units > ctx->config.max_mutation_work)) return false;
    if (!stable_sort(w->points, np, sizeof(*w->points), w->scratch, points_cmp) ||
        !stable_sort(w->target_points, np, sizeof(*w->target_points), w->scratch,
                     target_points_cmp) ||
        !stable_sort(w->bases, nb, sizeof(*w->bases), w->scratch, bases_cmp) ||
        !stable_sort(w->allocs, na, sizeof(*w->allocs), w->scratch, allocs_cmp) ||
        !stable_sort(w->arrays, nm, sizeof(*w->arrays), w->scratch, arrays_cmp)) {
        return false;
    }
    *points_out = np; *bases_out = nb; *allocs_out = na; *arrays_out = nm;
    return true;
}

static void free_work(MutationWork *w)
{
    if (w == NULL) return;
    g_free(w->points); g_free(w->target_points); g_free(w->summaries);
    g_free(w->bases); g_free(w->allocs); g_free(w->arrays); g_free(w->scratch);
    memset(w, 0, sizeof(*w));
}

static size_t lb_address(const OspreyMayArrayFact *a, size_t n,
                         const OspreyAddress *key)
{
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t m = lo + (hi - lo) / 2;
        if (address_cmp(&a[m].start, key) < 0) lo = m + 1; else hi = m;
    }
    return lo;
}

static size_t lb_base(const OspreyBaseFact *a, size_t n,
                      const OspreyAddress *key)
{
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t m = lo + (hi - lo) / 2;
        if (address_cmp(&a[m].base, key) < 0) lo = m + 1; else hi = m;
    }
    return lo;
}

static size_t lb_site(const OspreyMallocFact *a, size_t n, uint64_t site)
{
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t m = lo + (hi - lo) / 2;
        if (a[m].site_pc < site) lo = m + 1; else hi = m;
    }
    return lo;
}

static bool array_shape(const MutationWork *w, size_t n,
                        const OspreyAddress *target, uint64_t *extent,
                        OspreyMutationAbstentionReason *reason)
{
    size_t i = lb_address(w->arrays, n, target);
    bool found = false;
    uint64_t selected_count = 0, selected_size = 0;
    while (i < n && address_eq(&w->arrays[i].start, target)) {
        uint64_t product;
        if (!mul_u64(w->arrays[i].element_count,
                     w->arrays[i].element_size, &product) || product == 0 ||
            product > OSPREY_MUTATION_MAX_EXTENT) {
            *reason = OSPREY_MUTATION_ABSTAIN_EXTENT_INVALID;
            return false;
        }
        if (!found) {
            found = true; selected_count = w->arrays[i].element_count;
            selected_size = w->arrays[i].element_size; *extent = product;
        } else if (selected_count != w->arrays[i].element_count ||
                   selected_size != w->arrays[i].element_size) {
            *reason = OSPREY_MUTATION_ABSTAIN_GEOMETRY_CONFLICT;
            return false;
        }
        i++;
    }
    return found;
}

static bool struct_shape(const MutationWork *w, size_t n,
                         const OspreyAddress *target, uint64_t *extent,
                         OspreyMutationAbstentionReason *reason)
{
    size_t i = lb_base(w->bases, n, target), distinct = 0;
    bool have_previous = false;
    int64_t previous_end = 0;
    uint64_t max_end = 0;
    while (i < n && address_eq(&w->bases[i].base, target)) {
        const OspreyBaseFact *f = &w->bases[i];
        int64_t relative, end_signed;
        uint64_t end;
        if (!chunk_valid(&f->chunk) ||
            !region_eq(&f->chunk.address.region, &target->region) ||
            !osprey_check_sub(f->chunk.address.offset, target->offset,
                              &relative) || relative < 0 ||
            !osprey_check_add(relative, (int64_t)f->chunk.size, &end_signed) ||
            end_signed <= 0) {
            *reason = OSPREY_MUTATION_ABSTAIN_EXTENT_INVALID; return false;
        }
        end = (uint64_t)end_signed;
        if (end > OSPREY_MUTATION_MAX_EXTENT) {
            *reason = OSPREY_MUTATION_ABSTAIN_EXTENT_INVALID; return false;
        }
        if (!have_previous || !chunk_eq(&f->chunk, &w->bases[i - 1].chunk)) {
            if (have_previous && relative < previous_end) {
                *reason = OSPREY_MUTATION_ABSTAIN_LAYOUT_CONFLICT; return false;
            }
            have_previous = true; previous_end = end_signed; distinct++;
            if (end > max_end) max_end = end;
        }
        i++;
    }
    if (distinct < 2 || max_end == 0) return false;
    *extent = max_end; return true;
}

static bool heap_capacity(const MutationWork *w, size_t n,
                          const OspreyAddress *target, uint64_t extent,
                          OspreyMutationAbstentionReason *reason)
{
    if (target->region.kind != OSPREY_REGION_HEAP_SITE) return true;
    if (target->region.code_image_id != 0) {
        *reason = OSPREY_MUTATION_ABSTAIN_F05_MISSING; return false;
    }
    size_t i = lb_site(w->allocs, n, target->region.site_offset);
    bool found = false; uint64_t selected = 0;
    while (i < n && w->allocs[i].site_pc == target->region.site_offset) {
        if (!found) { found = true; selected = w->allocs[i].requested_size; }
        else if (selected != w->allocs[i].requested_size) {
            *reason = OSPREY_MUTATION_ABSTAIN_F05_CONFLICT; return false;
        }
        i++;
    }
    if (!found) { *reason = OSPREY_MUTATION_ABSTAIN_F05_MISSING; return false; }
    if (extent > selected) {
        *reason = OSPREY_MUTATION_ABSTAIN_EXTENT_INVALID; return false;
    }
    return true;
}

static bool build_summaries(MutationWork *w, size_t np, size_t nb,
                            size_t na, size_t nm)
{
    size_t i = 0; uint32_t count = 0;
    while (i < np) {
        size_t end = i + 1;
        MutationTargetSummary *s = &w->summaries[count];
        OspreyMutationAbstentionReason reason = OSPREY_MUTATION_ABSTAIN_NONE;
        uint64_t ae = 0, se = 0;
        OspreyMutationAbstentionReason ar = OSPREY_MUTATION_ABSTAIN_NO_SHAPE;
        OspreyMutationAbstentionReason sr = OSPREY_MUTATION_ABSTAIN_NO_SHAPE;
        bool ha, hs;
        memset(s, 0, sizeof(*s)); s->target = w->target_points[i].target;
        while (end < np && address_eq(&w->target_points[end].target,
                                      &s->target)) end++;
        if (reason == OSPREY_MUTATION_ABSTAIN_NONE) {
            ha = array_shape(w, nm, &s->target, &ae, &ar);
            hs = struct_shape(w, nb, &s->target, &se, &sr);
            if (ar != OSPREY_MUTATION_ABSTAIN_NO_SHAPE) reason = ar;
            else if (sr != OSPREY_MUTATION_ABSTAIN_NO_SHAPE) reason = sr;
            else if (ha && hs) reason = OSPREY_MUTATION_ABSTAIN_KIND_CONFLICT;
            else if (!ha && !hs) reason = OSPREY_MUTATION_ABSTAIN_NO_SHAPE;
            else {
                s->extent = ha ? ae : se;
                s->kind = ha ? OSPREY_MUTATION_AGGREGATE_ARRAY
                             : OSPREY_MUTATION_AGGREGATE_STRUCT;
                (void)heap_capacity(w, na, &s->target, s->extent, &reason);
            }
        }
        s->reason = reason; count++; i = end;
    }
    w->summary_count = count; return true;
}

static size_t lb_summary(const MutationTargetSummary *a, size_t n,
                         const OspreyAddress *key)
{
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t m = lo + (hi - lo) / 2;
        if (address_cmp(&a[m].target, key) < 0) lo = m + 1; else hi = m;
    }
    return lo;
}

static void abstain(OspreyMutationStats *stats,
                    OspreyMutationAbstentionReason reason)
{
    if (reason > OSPREY_MUTATION_ABSTAIN_NONE &&
        reason < OSPREY_MUTATION_ABSTAIN_REASON_COUNT) stats->abstentions[reason]++;
}

const char *osprey_mutation_abstention_reason(
    OspreyMutationAbstentionReason reason)
{
    static const char *const names[OSPREY_MUTATION_ABSTAIN_REASON_COUNT] = {
        "none", "target-tie", "no-shape", "kind-conflict",
        "geometry-conflict", "f05-missing", "f05-conflict",
        "layout-conflict", "extent-invalid", "target-base-mismatch", "count"
    };
    return reason < OSPREY_MUTATION_ABSTAIN_REASON_COUNT ? names[reason] : "unknown";
}

bool osprey_mutation_model_validate(const OspreyMutationModel *m)
{
    if (m == NULL || m->version != OSPREY_MUTATION_MODEL_VERSION ||
        m->publication_valid != 1 || (m->entry_count && m->entries == NULL)) {
        return false;
    }
    for (uint32_t i = 0; i < m->entry_count; i++) {
        const OspreyMutationEntry *e = &m->entries[i];
        if (!chunk_valid(&e->cell) || e->cell.size != sizeof(target_ulong) ||
            !region_valid(&e->target_base.region) || e->extent == 0 ||
            e->extent > OSPREY_MUTATION_MAX_EXTENT || e->support == 0 ||
            (e->kind != OSPREY_MUTATION_AGGREGATE_ARRAY &&
             e->kind != OSPREY_MUTATION_AGGREGATE_STRUCT) || e->ordinal != i ||
            (i && chunk_cmp(&m->entries[i - 1].cell, &e->cell) >= 0)) return false;
    }
    return true;
}

const OspreyMutationEntry *osprey_mutation_lookup(
    const OspreyMutationModel *m, const OspreyChunk *cell)
{
    uint32_t lo = 0, hi;
    if (m == NULL || cell == NULL || !osprey_mutation_model_validate(m)) return NULL;
    hi = m->entry_count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (chunk_cmp(&m->entries[mid].cell, cell) < 0) lo = mid + 1; else hi = mid;
    }
    return lo < m->entry_count && chunk_eq(&m->entries[lo].cell, cell)
        ? &m->entries[lo] : NULL;
}

void osprey_mutation_model_clear(OspreyContext *ctx)
{
    if (ctx == NULL) return;
    if (ctx->mutation_model != NULL) {
        g_free(ctx->mutation_model->entries); g_free(ctx->mutation_model);
        ctx->mutation_model = NULL;
    }
    ctx->mutation_model_ready = false;
    ctx->mutation_runtime_ready = false;
}

static OspreyStatus mutation_reject(OspreyContext *ctx, const char *reason)
{
    if (ctx != NULL) {
        ctx->mutation_model_ready = false;
        ctx->mutation_runtime_ready = false;
        if (osprey_tx_ok(ctx)) osprey_tx_reject(ctx, OSPREY_LIMIT_EXCEEDED,
                                                "mutation", reason);
    }
    return OSPREY_LIMIT_EXCEEDED;
}

OspreyStatus osprey_mutation_model_build(OspreyContext *ctx)
{
    MutationWork w; OspreyMutationModel *m = NULL; OspreyMutationStats stats;
    uint64_t input; size_t np, nb, na, nm; size_t i = 0;
    memset(&w, 0, sizeof(w)); memset(&stats, 0, sizeof(stats));
    if (ctx == NULL || !ctx->config.enabled) return OSPREY_DISABLED;
    ctx->mutation_model_ready = false;
    ctx->mutation_runtime_ready = false;
    if (!input_count(ctx, &input)) return mutation_reject(ctx, "mutation-input-cap");
    stats.input_facts = input;
    if (ctx->config.max_mutation_input != 0 &&
        input > ctx->config.max_mutation_input) return mutation_reject(ctx, "mutation-input-cap");
    if (!validate_input(ctx)) return mutation_reject(ctx, "mutation-malformed-input");
    if (!prepare_work(ctx, &w, input, &np, &nb, &na, &nm)) {
        const char *reason = ctx->config.max_mutation_work != 0 &&
            w.work_units > ctx->config.max_mutation_work
            ? "mutation-work-budget" : "mutation-bytes-cap";
        free_work(&w); return mutation_reject(ctx, reason);
    }
    if (ctx->analysis_active && !osprey_budget_checkpoint(ctx, OSPREY_ANALYSIS_DECODE)) {
        free_work(&w); return mutation_reject(ctx, "analysis-deadline");
    }
    if (!build_summaries(&w, np, nb, na, nm)) {
        free_work(&w);
        return mutation_reject(ctx, "mutation-malformed-input");
    }
    m = g_try_malloc0(sizeof(*m));
    if (m == NULL) { free_work(&w); return mutation_reject(ctx, "mutation-bytes-cap"); }
    m->version = OSPREY_MUTATION_MODEL_VERSION; m->stats = stats;
    m->stats.work_units = w.work_units; m->stats.owned_bytes = w.projected_bytes;
    m->entries = np ? g_try_malloc0(np * sizeof(*m->entries)) : NULL;
    if (np && m->entries == NULL) { g_free(m); free_work(&w); return mutation_reject(ctx, "mutation-bytes-cap"); }
    while (i < np) {
        size_t end = i + 1; OspreyChunk cell = w.points[i].pointer_chunk;
        size_t selected = SIZE_MAX; uint64_t best = 0; bool tie = false;
        while (end < np && chunk_eq(&w.points[end].pointer_chunk, &cell)) end++;
        m->stats.eligible_cells++;
        for (size_t p = i; p < end; ) {
            size_t q = p + 1; const MutationTargetSummary *s;
            uint64_t cell_target_support = w.points[p].sample_support;
            s = &w.summaries[lb_summary(w.summaries, w.summary_count,
                                         &w.points[p].target)];
            while (q < end && address_eq(&w.points[q].target,
                                         &w.points[p].target)) {
                if (!add_u64(&cell_target_support,
                             w.points[q].sample_support)) {
                    free_work(&w);
                    g_free(m->entries);
                    g_free(m);
                    return mutation_reject(ctx, "mutation-malformed-input");
                }
                q++;
            }
            if (selected == SIZE_MAX || cell_target_support > best) {
                selected = (size_t)(s - w.summaries);
                best = cell_target_support;
                tie = false;
            } else if (cell_target_support == best) tie = true;
            p = q;
        }
        const MutationTargetSummary *s = selected == SIZE_MAX ? NULL :
            &w.summaries[selected];
        OspreyMutationAbstentionReason reason = tie
            ? OSPREY_MUTATION_ABSTAIN_TARGET_TIE
            : s == NULL ? OSPREY_MUTATION_ABSTAIN_COUNT : s->reason;
        if (reason != OSPREY_MUTATION_ABSTAIN_NONE) {
            abstain(&m->stats, reason);
            log_msg("[osprey] [mutation] [abstain] [reason %s]\n",
                    osprey_mutation_abstention_reason(reason));
            i = end; continue;
        }
        OspreyMutationEntry *e = &m->entries[m->entry_count++];
        memset(e, 0, sizeof(*e)); e->cell = cell; e->target_base = s->target;
        e->extent = s->extent; e->support = best; e->ordinal = m->entry_count - 1;
        e->kind = s->kind; m->stats.published_entries++; i = end;
    }
    m->publication_valid = 1;
    if (!osprey_mutation_model_validate(m)) {
        g_free(m->entries); g_free(m); free_work(&w);
        return mutation_reject(ctx, "mutation-malformed-input");
    }
    osprey_mutation_model_clear(ctx); ctx->mutation_model = m;
    ctx->mutation_model_ready = true;
    log_msg("[osprey] [mutation] [done] [input %" PRIu64 "] [eligible %" PRIu64
            "] [entries %" PRIu64 "] [work %" PRIu64 "] [bytes %" PRIu64 "]\n",
            m->stats.input_facts, m->stats.eligible_cells,
            m->stats.published_entries, m->stats.work_units, m->stats.owned_bytes);
    free_work(&w); return OSPREY_OK;
}

const OspreyMutationModel *osprey_mutation_model(const OspreyContext *ctx)
{
    return ctx != NULL && ctx->tx_status == OSPREY_OK &&
           ctx->mutation_model_ready && ctx->mutation_model != NULL &&
           osprey_mutation_model_validate(ctx->mutation_model)
        ? ctx->mutation_model : NULL;
}
