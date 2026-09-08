#ifndef BINRADAR_STAGE7_MUTATION_REFERENCE_H
#define BINRADAR_STAGE7_MUTATION_REFERENCE_H

#include "osprey.h"

/* Deliberately slow, index-independent cell and pointer oracles for
 * Stage 7.2 tests.  They scan the immutable source records directly and
 * never call the production runtime index or resolvers. */
OspreyRuntimeResolveStatus stage7_reference_resolve_cell(
    const OspreyContext *ctx, const OspreyModel *model,
    const OspreyRuntimeChunkRef *locator,
    OspreyRuntimeCellResolution *out);
OspreyRuntimeResolveStatus stage7_reference_resolve_pointer(
    const OspreyContext *ctx, const OspreyModel *model,
    const OspreyRuntimeChunkRef *cell_locator, target_ulong concrete_value,
    const OspreyRuntimeAddressRef *target_locator,
    OspreyRuntimePointerResolution *out);

#endif
