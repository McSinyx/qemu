/*
 * OSPREY Stage 7.2: instance-exact parent-side runtime resolution.
 *
 * The baseline region catalog is copied into one immutable, sorted index.
 * Resolver calls do not allocate, read guest memory, mutate the model, or
 * touch mutation queues.  A failed identity, bounds, or model check is a
 * typed-unavailable result for that access.
 */

#include "qemu/osdep.h"
#include "osprey-internal.h"

#include <limits.h>
#include <stdlib.h>

struct OspreyRuntimeIndex {
    OspreyRegionInstance *entries;
    uint32_t count;
    bool valid;
    OspreyRuntimeResolveStatus failure;
};

static OspreyRuntimeCounters runtime_counters;

void osprey_runtime_test_reset_counters(void)
{
    memset(&runtime_counters, 0, sizeof(runtime_counters));
}

void osprey_runtime_test_get_counters(OspreyRuntimeCounters *out)
{
    if (out != NULL) *out = runtime_counters;
}

static int runtime_cmp_u64(uint64_t a, uint64_t b)
{
    return a < b ? -1 : a != b;
}

static int runtime_cmp_i64(int64_t a, int64_t b)
{
    return a < b ? -1 : a != b;
}

static int runtime_region_compare(const OspreyRegionId *a,
                                  const OspreyRegionId *b)
{
    int c;

    if (a == NULL || b == NULL) return a == b ? 0 : (a == NULL ? -1 : 1);
    c = runtime_cmp_u64((uint64_t)a->kind, (uint64_t)b->kind);
    if (c != 0) return c;
    c = runtime_cmp_u64(a->code_image_id, b->code_image_id);
    if (c != 0) return c;
    return runtime_cmp_u64(a->site_offset, b->site_offset);
}

static int runtime_address_compare(const OspreyAddress *a,
                                   const OspreyAddress *b)
{
    int c = runtime_region_compare(&a->region, &b->region);
    return c != 0 ? c : runtime_cmp_i64(a->offset, b->offset);
}

static bool runtime_address_equal(const OspreyAddress *a,
                                  const OspreyAddress *b)
{
    return runtime_address_compare(a, b) == 0;
}

static int runtime_instance_key_compare(const OspreyRegionInstance *a,
                                        const OspreyRegionInstance *b)
{
    int c = runtime_region_compare(&a->region, &b->region);
    if (c != 0) return c;
    c = runtime_cmp_u64(a->instance_id, b->instance_id);
    if (c != 0) return c;
    c = runtime_cmp_u64(a->raw_base, b->raw_base);
    if (c != 0) return c;
    c = runtime_cmp_u64(a->prov_object_id, b->prov_object_id);
    if (c != 0) return c;
    return runtime_cmp_u64(a->prov_generation, b->prov_generation);
}

static int runtime_instance_qsort_compare(const void *ap, const void *bp)
{
    return runtime_instance_key_compare(ap, bp);
}

static bool runtime_instance_shape_valid(const OspreyRegionInstance *instance)
{
    if (instance == NULL || instance->region.kind > OSPREY_REGION_STACK_FUNCTION ||
        instance->raw_min > instance->raw_max ||
        instance->raw_base < instance->raw_min ||
        instance->raw_base > instance->raw_max) {
        return false;
    }
    switch (instance->region.kind) {
    case OSPREY_REGION_GLOBAL:
        return instance->instance_id == 0 &&
               instance->prov_object_id == 0 &&
               instance->prov_generation == 0;
    case OSPREY_REGION_HEAP_SITE:
        return instance->instance_id != 0 &&
               instance->prov_object_id != 0 &&
               instance->prov_generation != 0;
    case OSPREY_REGION_STACK_FUNCTION:
        return instance->instance_id != 0 &&
               instance->prov_object_id == 0 &&
               instance->prov_generation == 0;
    default:
        return false;
    }
}

static void runtime_index_free(OspreyRuntimeIndex *index)
{
    if (index == NULL) return;
    g_free(index->entries);
    g_free(index);
}

void osprey_runtime_index_clear(OspreyContext *ctx)
{
    if (ctx == NULL) return;
    runtime_index_free(ctx->runtime_index);
    ctx->runtime_index = NULL;
}

bool osprey_runtime_index_build(OspreyContext *ctx)
{
    OspreyRuntimeIndex *index;
    uint64_t count;

    if (ctx == NULL) return false;
    osprey_runtime_index_clear(ctx);

    index = g_try_new0(OspreyRuntimeIndex, 1);
    if (index == NULL) return false;
    index->failure = OSPREY_RUNTIME_INDEX_UNAVAILABLE;
    if (ctx->region_instances == NULL) {
        index->failure = OSPREY_RUNTIME_MALFORMED;
        ctx->runtime_index = index;
        return false;
    }
    count = ctx->region_instances->len;
    if (count > UINT32_MAX ||
        count > SIZE_MAX / sizeof(*index->entries)) {
        index->failure = OSPREY_RUNTIME_MALFORMED;
        ctx->runtime_index = index;
        return false;
    }
    index->count = (uint32_t)count;
    if (index->count != 0) {
        index->entries = g_try_malloc0((size_t)index->count *
                                       sizeof(*index->entries));
        if (index->entries == NULL) {
            g_free(index);
            return false;
        }
        memcpy(index->entries, ctx->region_instances->data,
               (size_t)index->count * sizeof(*index->entries));
        qsort(index->entries, index->count, sizeof(*index->entries),
              runtime_instance_qsort_compare);
    }

    for (uint32_t i = 0; i < index->count; i++) {
        if (!runtime_instance_shape_valid(&index->entries[i])) {
            index->failure = OSPREY_RUNTIME_MALFORMED;
            ctx->runtime_index = index;
            return false;
        }
        if (i != 0 && runtime_instance_key_compare(
                &index->entries[i - 1], &index->entries[i]) == 0) {
            /* Equal complete identity with either equal or conflicting
             * bounds is never a resolvable runtime instance. */
            index->failure = OSPREY_RUNTIME_AMBIGUOUS_INSTANCE;
            ctx->runtime_index = index;
            return false;
        }
    }

    index->valid = true;
    index->failure = OSPREY_RUNTIME_RESOLVED;
    ctx->runtime_index = index;
    return true;
}

bool osprey_runtime_index_ready(const OspreyContext *ctx)
{
    return ctx != NULL && ctx->runtime_index != NULL &&
           ctx->runtime_index->valid;
}

static bool runtime_add_signed(uint64_t base, int64_t offset, uint64_t *out)
{
    uint64_t magnitude;

    if (out == NULL) return false;
    if (offset >= 0) {
        magnitude = (uint64_t)offset;
        if (base > UINT64_MAX - magnitude) return false;
        *out = base + magnitude;
        return true;
    }
    magnitude = (uint64_t)(-(offset + 1)) + 1;
    if (base < magnitude) return false;
    *out = base - magnitude;
    return true;
}

static bool runtime_derive_base(const OspreyRuntimeAddressRef *ref,
                                 uint64_t *base_out)
{
    uint64_t magnitude;

    if (ref == NULL || base_out == NULL) return false;
    if (ref->address.offset >= 0) {
        magnitude = (uint64_t)ref->address.offset;
        if (ref->raw < magnitude) return false;
        *base_out = ref->raw - magnitude;
        return true;
    }
    magnitude = (uint64_t)(-(ref->address.offset + 1)) + 1;
    if (ref->raw > UINT64_MAX - magnitude) return false;
    *base_out = ref->raw + magnitude;
    return true;
}

static OspreyRegionInstance runtime_probe_from_ref(
    const OspreyRuntimeAddressRef *ref, uint64_t raw_base)
{
    OspreyRegionInstance probe;
    memset(&probe, 0, sizeof(probe));
    probe.region = ref->address.region;
    probe.instance_id = ref->instance_id;
    probe.raw_base = raw_base;
    probe.prov_object_id = ref->prov_object_id;
    probe.prov_generation = ref->prov_generation;
    return probe;
}

static int runtime_instance_search_compare(
    const OspreyRegionInstance *a, const OspreyRegionInstance *b)
{
    runtime_counters.runtime_instance_comparisons++;
    return runtime_instance_key_compare(a, b);
}

static OspreyRuntimeResolveStatus runtime_find_instance(
    const OspreyContext *ctx, const OspreyRuntimeAddressRef *ref,
    OspreyRegionInstance *out)
{
    const OspreyRuntimeIndex *index;
    OspreyRegionInstance probe;
    uint32_t lo, hi;
    uint64_t raw_base, translated;

    if (ctx == NULL || ref == NULL || out == NULL) {
        return OSPREY_RUNTIME_MALFORMED;
    }
    if (ref->valid == 0) return OSPREY_RUNTIME_NO_LOCATOR;
    if (ref->valid != 1 || ref->reserved[0] != 0 ||
        ref->reserved[1] != 0 || ref->reserved[2] != 0 ||
        ref->address.region.kind > OSPREY_REGION_STACK_FUNCTION) {
        return OSPREY_RUNTIME_MALFORMED;
    }
    index = ctx->runtime_index;
    if (index == NULL) return OSPREY_RUNTIME_INDEX_UNAVAILABLE;
    if (!index->valid) return index->failure;
    if (!runtime_derive_base(ref, &raw_base)) {
        return OSPREY_RUNTIME_STALE_INSTANCE;
    }
    probe = runtime_probe_from_ref(ref, raw_base);

    /* The full identity includes the recovered raw anchor. */
    lo = 0;
    hi = index->count;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        int c = runtime_instance_search_compare(&index->entries[mid], &probe);
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    if (lo == index->count ||
        runtime_instance_search_compare(&index->entries[lo], &probe) != 0) {
        return OSPREY_RUNTIME_STALE_INSTANCE;
    }
    if (lo + 1 < index->count &&
        runtime_instance_key_compare(&index->entries[lo],
                                      &index->entries[lo + 1]) == 0) {
        return OSPREY_RUNTIME_AMBIGUOUS_INSTANCE;
    }

    const OspreyRegionInstance *entry = &index->entries[lo];
    if (!runtime_add_signed(entry->raw_base, ref->address.offset,
                            &translated) || translated != ref->raw) {
        return OSPREY_RUNTIME_STALE_INSTANCE;
    }
    if (entry->raw_min > entry->raw_max || ref->raw < entry->raw_min ||
        ref->raw >= entry->raw_max) {
        return OSPREY_RUNTIME_OUT_OF_BOUNDS;
    }
    *out = *entry;
    return OSPREY_RUNTIME_RESOLVED;
}

static bool runtime_canonical_chunk_valid(const OspreyChunk *chunk)
{
    uint64_t delta;

    if (chunk == NULL || chunk->size == 0 ||
        chunk->address.region.kind > OSPREY_REGION_STACK_FUNCTION) {
        return false;
    }
    delta = chunk->size - 1;
    if (delta > INT64_MAX) return false;
    if (chunk->address.offset >= 0) {
        return (uint64_t)chunk->address.offset <=
               (uint64_t)INT64_MAX - delta;
    }
    /* Keep a negative stack span on the same side of its anchor.  Express
     * the check through magnitude so INT64_MIN never enters signed math. */
    uint64_t magnitude =
        (uint64_t)(-(chunk->address.offset + 1)) + 1;
    return delta < magnitude;
}

static bool runtime_ref_is_zero(const OspreyRuntimeAddressRef *ref)
{
    OspreyRuntimeAddressRef zero;
    if (ref == NULL) return true;
    memset(&zero, 0, sizeof(zero));
    return memcmp(ref, &zero, sizeof(zero)) == 0;
}

static bool runtime_mutation_model_shape_valid(
    const OspreyMutationModel *model)
{
    if (model == NULL || !osprey_mutation_model_validate(model)) return false;
    for (uint32_t i = 0; i < model->entry_count; i++) {
        const OspreyMutationEntry *entry = &model->entries[i];
        if (entry->target_base.region.kind > OSPREY_REGION_STACK_FUNCTION ||
            entry->extent == 0 || entry->extent > OSPREY_MUTATION_MAX_EXTENT ||
            (entry->kind != OSPREY_MUTATION_AGGREGATE_ARRAY &&
             entry->kind != OSPREY_MUTATION_AGGREGATE_STRUCT)) {
            return false;
        }
    }
    return true;
}

bool osprey_runtime_mutation_prepare(OspreyContext *ctx)
{
    if (ctx == NULL || ctx->config.analysis_mode !=
            OSPREY_ANALYSIS_MODE_MUTATION || !ctx->mutation_model_ready ||
        ctx->mutation_model == NULL || !osprey_tx_ok(ctx)) {
        return false;
    }
    if (ctx->mutation_runtime_ready) return true;
    runtime_counters.reception_validation_passes++;
    if (!runtime_mutation_model_shape_valid(ctx->mutation_model)) {
        ctx->mutation_model_ready = false;
        ctx->mutation_runtime_ready = false;
        return false;
    }
    ctx->mutation_runtime_ready = true;
    return true;
}

const OspreyMutationModel *osprey_runtime_mutation_model(
    const OspreyContext *ctx)
{
    return ctx != NULL && ctx->config.analysis_mode ==
               OSPREY_ANALYSIS_MODE_MUTATION &&
           ctx->mutation_model_ready && ctx->mutation_runtime_ready &&
           osprey_tx_ok(ctx) && ctx->mutation_model != NULL
        ? ctx->mutation_model : NULL;
}

static int runtime_compact_chunk_compare(const OspreyChunk *a,
                                         const OspreyChunk *b)
{
    int c = runtime_address_compare(&a->address, &b->address);
    if (c != 0) return c;
    return runtime_cmp_u64(a->size, b->size);
}

static const OspreyMutationEntry *runtime_compact_entry_find(
    const OspreyMutationModel *model, const OspreyChunk *chunk)
{
    uint32_t lo = 0;
    uint32_t hi = model == NULL ? 0 : model->entry_count;

    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        int c = runtime_compact_chunk_compare(&model->entries[mid].cell,
                                               chunk);
        runtime_counters.cell_key_comparisons++;
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    if (lo >= model->entry_count) return NULL;
    runtime_counters.cell_key_comparisons++;
    if (runtime_compact_chunk_compare(&model->entries[lo].cell, chunk) != 0) {
        return NULL;
    }
    return &model->entries[lo];
}

OspreyRuntimeResolveStatus osprey_runtime_resolve_cell(
    const OspreyContext *ctx, const OspreyMutationModel *model,
    const OspreyRuntimeChunkRef *locator,
    OspreyRuntimeCellResolution *out)
{
    OspreyRegionInstance instance;
    OspreyChunk chunk;
    const OspreyMutationEntry *entry;
    OspreyRuntimeResolveStatus status;
    uint64_t raw_end;

    if (out == NULL) return OSPREY_RUNTIME_MALFORMED;
    memset(out, 0, sizeof(*out));
    out->status = OSPREY_RUNTIME_MALFORMED;
    out->entry_ordinal = UINT32_MAX;
    if (locator == NULL || locator->start.valid == 0) {
        out->status = OSPREY_RUNTIME_NO_LOCATOR;
        return out->status;
    }
    if (locator->start.valid != 1 || locator->size == 0) {
        return out->status;
    }
    if (model == NULL || ctx == NULL || !ctx->mutation_runtime_ready ||
        !ctx->mutation_model_ready || model != ctx->mutation_model) {
        out->status = OSPREY_RUNTIME_INDEX_UNAVAILABLE;
        return out->status;
    }
    chunk.address = locator->start.address;
    chunk.size = locator->size;
    if (!runtime_canonical_chunk_valid(&chunk)) return out->status;
    status = runtime_find_instance(ctx, &locator->start, &instance);
    if (status != OSPREY_RUNTIME_RESOLVED) {
        out->status = status;
        return status;
    }
    if (locator->start.raw > UINT64_MAX - locator->size) {
        out->status = OSPREY_RUNTIME_OUT_OF_BOUNDS;
        return out->status;
    }
    raw_end = locator->start.raw + locator->size;
    if (raw_end <= locator->start.raw || raw_end > instance.raw_max) {
        out->status = OSPREY_RUNTIME_OUT_OF_BOUNDS;
        return out->status;
    }
    entry = runtime_compact_entry_find(model, &chunk);
    if (entry == NULL) {
        out->status = OSPREY_RUNTIME_NO_CELL_OBJECT;
        return out->status;
    }
    out->cell = *locator;
    out->instance_valid = 1;
    out->entry_ordinal = entry->ordinal;
    out->aggregate_kind = entry->kind;
    out->target_base = entry->target_base;
    out->target_extent = entry->extent;
    out->status = OSPREY_RUNTIME_RESOLVED;
    return out->status;
}

OspreyRuntimeResolveStatus osprey_runtime_resolve_pointer(
    const OspreyContext *ctx, const OspreyMutationModel *model,
    const OspreyRuntimeChunkRef *cell_locator, target_ulong concrete_value,
    const OspreyRuntimeAddressRef *target_locator,
    OspreyRuntimePointerResolution *out)
{
    OspreyRuntimeResolveStatus status;
    OspreyRegionInstance target_instance;
    uint64_t target_end;

    if (out == NULL) return OSPREY_RUNTIME_MALFORMED;
    memset(out, 0, sizeof(*out));
    out->status = OSPREY_RUNTIME_MALFORMED;
    out->entry_ordinal = UINT32_MAX;
    status = osprey_runtime_resolve_cell(ctx, model, cell_locator,
                                         &out->cell);
    if (status != OSPREY_RUNTIME_RESOLVED) {
        out->status = status;
        return status;
    }
    if (out->cell.cell.size != sizeof(target_ulong)) {
        out->status = OSPREY_RUNTIME_NON_POINTER;
        return out->status;
    }
    if (out->cell.aggregate_kind != OSPREY_MUTATION_AGGREGATE_ARRAY &&
        out->cell.aggregate_kind != OSPREY_MUTATION_AGGREGATE_STRUCT) {
        out->status = OSPREY_RUNTIME_MISSING_AGGREGATE;
        return out->status;
    }
    if (out->cell.target_extent == 0 ||
        out->cell.target_extent > OSPREY_MUTATION_MAX_EXTENT) {
        out->status = OSPREY_RUNTIME_OUT_OF_BOUNDS;
        return out->status;
    }
    out->entry_ordinal = out->cell.entry_ordinal;
    out->aggregate_kind = out->cell.aggregate_kind;
    out->target_extent = out->cell.target_extent;
    out->target_base = out->cell.target_base;

    if (concrete_value == 0) {
        if (target_locator != NULL && !runtime_ref_is_zero(target_locator)) {
            out->status = OSPREY_RUNTIME_MALFORMED;
            return out->status;
        }
        out->status = OSPREY_RUNTIME_RESOLVED;
        return out->status;
    }
    if (target_locator == NULL || target_locator->valid == 0) {
        out->status = OSPREY_RUNTIME_NO_LOCATOR;
        return out->status;
    }
    if (target_locator->valid != 1 ||
        target_locator->raw != (uint64_t)concrete_value) {
        out->status = target_locator->valid != 1
            ? OSPREY_RUNTIME_MALFORMED
            : OSPREY_RUNTIME_STALE_INSTANCE;
        return out->status;
    }
    if (!runtime_address_equal(&target_locator->address,
                               &out->target_base)) {
        out->status = OSPREY_RUNTIME_STALE_INSTANCE;
        return out->status;
    }
    status = runtime_find_instance(ctx, target_locator, &target_instance);
    if (status != OSPREY_RUNTIME_RESOLVED) {
        out->status = status;
        return status;
    }
    if (target_locator->raw > UINT64_MAX - out->target_extent) {
        out->status = OSPREY_RUNTIME_OUT_OF_BOUNDS;
        return out->status;
    }
    target_end = target_locator->raw + out->target_extent;
    if (target_end <= target_locator->raw ||
        target_end > target_instance.raw_max) {
        out->status = OSPREY_RUNTIME_OUT_OF_BOUNDS;
        return out->status;
    }
    out->target = *target_locator;
    out->target_valid = 1;
    out->has_runtime_target = 1;
    out->status = OSPREY_RUNTIME_RESOLVED;
    return out->status;
}
