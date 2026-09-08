#include "osprey-internal.h"
#include "stage7_mutation_reference.h"

#include <limits.h>

static int reference_region_compare(const OspreyRegionId *a,
                                    const OspreyRegionId *b)
{
    if ((uint64_t)a->kind != (uint64_t)b->kind)
        return (uint64_t)a->kind < (uint64_t)b->kind ? -1 : 1;
    if (a->code_image_id != b->code_image_id)
        return a->code_image_id < b->code_image_id ? -1 : 1;
    if (a->site_offset != b->site_offset)
        return a->site_offset < b->site_offset ? -1 : 1;
    return 0;
}

static bool reference_address_equal(const OspreyAddress *a,
                                    const OspreyAddress *b)
{
    return reference_region_compare(&a->region, &b->region) == 0 &&
           a->offset == b->offset;
}

static bool reference_derive_base(const OspreyRuntimeAddressRef *ref,
                                  uint64_t *out)
{
    uint64_t magnitude;
    if (ref->address.offset >= 0) {
        magnitude = (uint64_t)ref->address.offset;
        if (ref->raw < magnitude) return false;
        *out = ref->raw - magnitude;
        return true;
    }
    magnitude = (uint64_t)(-(ref->address.offset + 1)) + 1;
    if (ref->raw > UINT64_MAX - magnitude) return false;
    *out = ref->raw + magnitude;
    return true;
}

static bool reference_chunk_equal(const OspreyChunk *a,
                                  const OspreyChunk *b)
{
    return reference_address_equal(&a->address, &b->address) &&
           a->size == b->size;
}

static bool reference_ref_is_zero(const OspreyRuntimeAddressRef *ref)
{
    OspreyRuntimeAddressRef zero;
    if (ref == NULL) return true;
    memset(&zero, 0, sizeof(zero));
    return memcmp(ref, &zero, sizeof(zero)) == 0;
}

static OspreyRuntimeResolveStatus reference_find_instance(
    const OspreyContext *ctx, const OspreyRuntimeAddressRef *ref,
    OspreyRegionInstance *out)
{
    const OspreyRegionInstance *instance = NULL;
    uint64_t raw_base;
    unsigned matches = 0;

    if (ctx == NULL || ctx->region_instances == NULL || ref == NULL ||
        out == NULL) {
        return OSPREY_RUNTIME_MALFORMED;
    }
    if (ref->valid == 0) return OSPREY_RUNTIME_NO_LOCATOR;
    if (ref->valid != 1 || ref->reserved[0] != 0 ||
        ref->reserved[1] != 0 || ref->reserved[2] != 0 ||
        ref->address.region.kind > OSPREY_REGION_STACK_FUNCTION) {
        return OSPREY_RUNTIME_MALFORMED;
    }
    if (!reference_derive_base(ref, &raw_base)) {
        return OSPREY_RUNTIME_STALE_INSTANCE;
    }
    for (guint i = 0; i < ctx->region_instances->len; i++) {
        const OspreyRegionInstance *candidate = &g_array_index(
            ctx->region_instances, OspreyRegionInstance, i);
        if (reference_region_compare(&candidate->region,
                                     &ref->address.region) != 0 ||
            candidate->instance_id != ref->instance_id ||
            candidate->raw_base != raw_base ||
            candidate->prov_object_id != ref->prov_object_id ||
            candidate->prov_generation != ref->prov_generation) {
            continue;
        }
        instance = candidate;
        matches++;
    }
    if (matches == 0) return OSPREY_RUNTIME_STALE_INSTANCE;
    if (matches > 1) return OSPREY_RUNTIME_AMBIGUOUS_INSTANCE;
    if (instance->raw_min > instance->raw_max ||
        ref->raw < instance->raw_min || ref->raw >= instance->raw_max) {
        return OSPREY_RUNTIME_OUT_OF_BOUNDS;
    }
    *out = *instance;
    return OSPREY_RUNTIME_RESOLVED;
}

OspreyRuntimeResolveStatus stage7_reference_resolve_cell(
    const OspreyContext *ctx, const OspreyModel *model,
    const OspreyRuntimeChunkRef *locator,
    OspreyRuntimeCellResolution *out)
{
    OspreyRegionInstance instance;
    OspreyChunk chunk;
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
    if (locator->start.valid != 1 || locator->size == 0 || model == NULL ||
        model->version != OSPREY_MODEL_VERSION ||
        (model->object_count != 0 && model->objects == NULL)) {
        return out->status;
    }
    chunk.address = locator->start.address;
    chunk.size = locator->size;
    if (chunk.address.region.kind > OSPREY_REGION_STACK_FUNCTION ||
        chunk.size - 1 > INT64_MAX) {
        return out->status;
    }
    if (chunk.address.offset >= 0) {
        if ((uint64_t)chunk.address.offset >
            (uint64_t)INT64_MAX - (chunk.size - 1)) {
            return out->status;
        }
    } else {
        uint64_t magnitude =
            (uint64_t)(-(chunk.address.offset + 1)) + 1;
        if (chunk.size - 1 >= magnitude) return out->status;
    }

    status = reference_find_instance(ctx, &locator->start, &instance);
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
    for (uint32_t i = 0; i < model->object_count; i++) {
        if (!reference_chunk_equal(&model->objects[i].chunk, &chunk)) continue;
        out->object = &model->objects[i];
        out->object_index = i;
        out->status = OSPREY_RUNTIME_RESOLVED;
        return out->status;
    }
    out->status = OSPREY_RUNTIME_NO_CELL_OBJECT;
    return out->status;
}

OspreyRuntimeResolveStatus stage7_reference_resolve_pointer(
    const OspreyContext *ctx, const OspreyModel *model,
    const OspreyRuntimeChunkRef *cell_locator, target_ulong concrete_value,
    const OspreyRuntimeAddressRef *target_locator,
    OspreyRuntimePointerResolution *out)
{
    const OspreyDecodedObject *object;
    const OspreyDecodedType *pointer_type;
    const OspreyDecodedType *target_type;
    OspreyRegionInstance target_instance;
    OspreyRuntimeResolveStatus status;
    uint64_t target_end;

    if (out == NULL) return OSPREY_RUNTIME_MALFORMED;
    memset(out, 0, sizeof(*out));
    out->status = OSPREY_RUNTIME_MALFORMED;
    out->target_type_id = UINT32_MAX;
    status = stage7_reference_resolve_cell(ctx, model, cell_locator,
                                           &out->cell);
    if (status != OSPREY_RUNTIME_RESOLVED) {
        out->status = status;
        return status;
    }
    object = out->cell.object;
    if (object == NULL || model == NULL || model->types == NULL ||
        object->value_type_id >= model->type_count) {
        out->status = OSPREY_RUNTIME_MISSING_TYPE;
        return out->status;
    }
    if (object->has_pointer_target > 1) {
        out->status = OSPREY_RUNTIME_MALFORMED;
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
        !reference_address_equal(&target_type->canonical_base,
                                 &object->pointer_target)) {
        return out->status;
    }
    out->target_type_id = pointer_type->target_type_id;
    out->target_extent = target_type->size;
    out->target_base = object->pointer_target;

    if (concrete_value == 0) {
        if (target_locator != NULL && !reference_ref_is_zero(target_locator)) {
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
    if (!reference_address_equal(&target_locator->address,
                                 &object->pointer_target)) {
        out->status = OSPREY_RUNTIME_STALE_INSTANCE;
        return out->status;
    }
    status = reference_find_instance(ctx, target_locator, &target_instance);
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
