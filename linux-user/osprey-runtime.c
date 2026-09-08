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
        int c = runtime_instance_key_compare(&index->entries[mid], &probe);
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    if (lo == index->count ||
        runtime_instance_key_compare(&index->entries[lo], &probe) != 0) {
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

static int runtime_key_compare(const OspreyKey *a, const OspreyKey *b)
{
    int c = runtime_cmp_u64(a->tag, b->tag);
    if (c != 0) return c;
    for (size_t i = 0; i < G_N_ELEMENTS(a->w); i++) {
        c = runtime_cmp_u64(a->w[i], b->w[i]);
        if (c != 0) return c;
    }
    return 0;
}

static bool runtime_chunk_equal(const OspreyChunk *a, const OspreyChunk *b)
{
    return a != NULL && b != NULL &&
           runtime_address_equal(&a->address, &b->address) &&
           a->size == b->size;
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

static bool runtime_model_shape_valid(const OspreyModel *model)
{
    if (model == NULL || model->version != OSPREY_MODEL_VERSION ||
        model->chunk_index_count != model->object_count ||
        (model->object_count != 0 &&
         (model->objects == NULL || model->chunk_index == NULL))) {
        return false;
    }
    for (uint32_t i = 0; i < model->chunk_index_count; i++) {
        const OspreyModelIndexEntry *entry = &model->chunk_index[i];
        OspreyKey expected;
        if (entry->ordinal >= model->object_count) return false;
        expected = osprey_chunk_key(&model->objects[entry->ordinal].chunk);
        if (!osprey_key_equal(&entry->key, &expected)) return false;
        if (i != 0 && runtime_key_compare(&model->chunk_index[i - 1].key,
                                          &entry->key) >= 0) {
            return false;
        }
    }
    return true;
}

static uint32_t runtime_model_object_index(const OspreyModel *model,
                                           const OspreyDecodedObject *object)
{
    if (model == NULL || object == NULL) return UINT32_MAX;
    for (uint32_t i = 0; i < model->object_count; i++) {
        if (&model->objects[i] == object) return i;
    }
    return UINT32_MAX;
}

static bool runtime_ref_is_zero(const OspreyRuntimeAddressRef *ref)
{
    OspreyRuntimeAddressRef zero;
    if (ref == NULL) return true;
    memset(&zero, 0, sizeof(zero));
    return memcmp(ref, &zero, sizeof(zero)) == 0;
}

OspreyRuntimeResolveStatus osprey_runtime_resolve_cell(
    const OspreyContext *ctx, const OspreyModel *model,
    const OspreyRuntimeChunkRef *locator,
    OspreyRuntimeCellResolution *out)
{
    OspreyRegionInstance instance;
    OspreyChunk chunk;
    const OspreyDecodedObject *object;
    OspreyRuntimeResolveStatus status;
    uint64_t raw_end;

    if (out == NULL) return OSPREY_RUNTIME_MALFORMED;
    memset(out, 0, sizeof(*out));
    out->status = OSPREY_RUNTIME_MALFORMED;
    out->object_index = UINT32_MAX;
    if (locator == NULL || locator->start.valid == 0) {
        out->status = OSPREY_RUNTIME_NO_LOCATOR;
        return out->status;
    }
    if (locator->start.valid != 1 || locator->size == 0) {
        out->status = OSPREY_RUNTIME_MALFORMED;
        return out->status;
    }
    if (!runtime_model_shape_valid(model)) {
        out->status = OSPREY_RUNTIME_MALFORMED;
        return out->status;
    }
    chunk.address = locator->start.address;
    chunk.size = locator->size;
    if (!runtime_canonical_chunk_valid(&chunk)) {
        out->status = OSPREY_RUNTIME_MALFORMED;
        return out->status;
    }
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

    out->cell = *locator;
    out->instance_valid = 1;
    object = osprey_lookup_chunk(model, &chunk);
    if (object == NULL) {
        out->status = OSPREY_RUNTIME_NO_CELL_OBJECT;
        return out->status;
    }
    if (!runtime_chunk_equal(&object->chunk, &chunk)) {
        out->status = OSPREY_RUNTIME_MALFORMED;
        return out->status;
    }
    out->object = object;
    out->object_index = runtime_model_object_index(model, object);
    if (out->object_index == UINT32_MAX) {
        out->status = OSPREY_RUNTIME_MALFORMED;
        return out->status;
    }
    out->status = OSPREY_RUNTIME_RESOLVED;
    return out->status;
}

OspreyRuntimeResolveStatus osprey_runtime_resolve_pointer(
    const OspreyContext *ctx, const OspreyModel *model,
    const OspreyRuntimeChunkRef *cell_locator, target_ulong concrete_value,
    const OspreyRuntimeAddressRef *target_locator,
    OspreyRuntimePointerResolution *out)
{
    const OspreyDecodedObject *object;
    const OspreyDecodedType *pointer_type;
    const OspreyDecodedType *target_type;
    OspreyRuntimeResolveStatus status;
    OspreyRegionInstance target_instance;
    uint64_t target_end;

    if (out == NULL) return OSPREY_RUNTIME_MALFORMED;
    memset(out, 0, sizeof(*out));
    out->status = OSPREY_RUNTIME_MALFORMED;
    out->target_type_id = UINT32_MAX;

    status = osprey_runtime_resolve_cell(ctx, model, cell_locator,
                                         &out->cell);
    if (status != OSPREY_RUNTIME_RESOLVED) {
        out->status = status;
        return status;
    }
    object = out->cell.object;
    if (object == NULL || model == NULL || model->types == NULL) {
        out->status = OSPREY_RUNTIME_MISSING_TYPE;
        return out->status;
    }
    if (object->has_pointer_target > 1) {
        out->status = OSPREY_RUNTIME_MALFORMED;
        return out->status;
    }
    if (object->value_type_id >= model->type_count) {
        out->status = OSPREY_RUNTIME_MISSING_TYPE;
        return out->status;
    }
    pointer_type = &model->types[object->value_type_id];
    if (pointer_type->kind != OSPREY_TYPE_POINTER ||
        pointer_type->size != sizeof(target_ulong) ||
        object->chunk.size != sizeof(target_ulong)) {
        out->status = OSPREY_RUNTIME_NON_POINTER;
        return out->status;
    }
    if (pointer_type->target_is_void > 1) {
        out->status = OSPREY_RUNTIME_MALFORMED;
        return out->status;
    }
    if (object->has_pointer_target == 0) {
        out->status = OSPREY_RUNTIME_NO_POINTER_TARGET;
        return out->status;
    }
    if (pointer_type->target_is_void != 0) {
        out->status = OSPREY_RUNTIME_VOID_TARGET;
        return out->status;
    }
    if (pointer_type->target_type_id >= model->type_count) {
        out->status = OSPREY_RUNTIME_MISSING_TYPE;
        return out->status;
    }
    target_type = &model->types[pointer_type->target_type_id];
    if (target_type->kind != OSPREY_TYPE_ARRAY &&
        target_type->kind != OSPREY_TYPE_STRUCT) {
        out->status = OSPREY_RUNTIME_MISSING_AGGREGATE;
        return out->status;
    }
    if (target_type->size == 0 ||
        !runtime_address_equal(&target_type->canonical_base,
                               &object->pointer_target)) {
        out->status = OSPREY_RUNTIME_MALFORMED;
        return out->status;
    }
    out->target_type_id = pointer_type->target_type_id;
    out->target_extent = target_type->size;
    out->target_base = object->pointer_target;

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
                               &object->pointer_target)) {
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
