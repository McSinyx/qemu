#ifndef BINRADAR_STAGE7_MUTATION_REFERENCE_H
#define BINRADAR_STAGE7_MUTATION_REFERENCE_H

#include "osprey.h"

/* Deliberately slow, index-independent cell and pointer oracles for
 * Stage 7.2 tests.  They scan the immutable source records directly and
 * never call the production runtime index or resolvers. */
OspreyRuntimeResolveStatus stage7_reference_resolve_cell(
    const OspreyContext *ctx, const OspreyMutationModel *model,
    const OspreyRuntimeChunkRef *locator,
    OspreyRuntimeCellResolution *out);
OspreyRuntimeResolveStatus stage7_reference_resolve_pointer(
    const OspreyContext *ctx, const OspreyMutationModel *model,
    const OspreyRuntimeChunkRef *cell_locator, target_ulong concrete_value,
    const OspreyRuntimeAddressRef *target_locator,
    OspreyRuntimePointerResolution *out);

/* Independent Stage 7.5 planner description.  These values deliberately
 * duplicate the serialized mutation kinds without including snapshot.c, so
 * the oracle cannot call or observe production queue/application helpers. */
typedef enum Stage7ReferencePointerSource {
    STAGE7_REFERENCE_FROM_PRIMITIVE = 0,
    STAGE7_REFERENCE_FROM_ACCESS = 1,
} Stage7ReferencePointerSource;

typedef enum Stage7ReferenceMutationKind {
    STAGE7_REFERENCE_BYTES = 0,
    STAGE7_REFERENCE_POINTER_NULL = 1,
    STAGE7_REFERENCE_POINTER_OOB = 2,
    STAGE7_REFERENCE_POINTER_FRESH = 3,
} Stage7ReferenceMutationKind;

typedef struct Stage7ReferenceCandidate {
    target_ulong addr;
    uint32_t size;
    int64_t expr_index;
    uint8_t value[sizeof(target_ulong)];
} Stage7ReferenceCandidate;

typedef struct Stage7ReferenceVariant {
    Stage7ReferenceMutationKind kind;
    target_ulong addr;
    uint32_t size;
    int64_t expr_index;
    uint8_t value[sizeof(target_ulong)];
    uint64_t target_extent;
    uint8_t *target_bytes;
    uint64_t resolved_target_raw;
    uint64_t resolved_target_end;
} Stage7ReferenceVariant;

typedef struct Stage7ReferencePlan {
    bool typed;
    OspreyRuntimeResolveStatus resolver_status;
    uint32_t count;
    Stage7ReferenceVariant variants[3];
} Stage7ReferencePlan;

void stage7_reference_plan_clear(Stage7ReferencePlan *plan);
bool stage7_reference_plan_build(
    const OspreyContext *ctx, const OspreyMutationModel *model,
    const Stage7ReferenceCandidate *candidate,
    const OspreyRuntimeChunkRef *cell_locator,
    const OspreyRuntimeAddressRef *target_locator,
    bool typed_allowed, Stage7ReferencePointerSource source_kind,
    uint64_t fresh_target_cap, uint64_t oob_delta,
    Stage7ReferencePlan *out);

#endif
