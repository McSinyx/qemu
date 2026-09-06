#ifndef BINRADAR_STAGE6_DECODE_REFERENCE_H
#define BINRADAR_STAGE6_DECODE_REFERENCE_H

/*
 * Declarative Stage 6.5 decoder oracle.
 *
 * This file deliberately contains no production decoder state.  A reference
 * case owns plain candidate payloads, beliefs, evidence, and extents.  The
 * implementation performs selection, checked geometry, score arithmetic, and
 * canonical model construction independently; the test harness compares its
 * result with the production model.
 */

#include "osprey.h"
#include "osprey-internal.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define OSPREY_REFERENCE_MAX_CANDIDATE_KIND 16u

typedef struct OspreyReferenceCandidate {
    uint8_t kind;
    uint8_t belief_valid;
    uint8_t hard_false;
    uint8_t reserved;
    OspreyVarPayload payload;
    double posterior;
    uint64_t direct_support;
    uint64_t source_rule_bits;
} OspreyReferenceCandidate;

typedef struct OspreyReferenceCase {
    const OspreyReferenceCandidate *candidates;
    uint32_t candidate_count;
    const OspreyRegionExtent *extents;
    uint32_t extent_count;
    double report_threshold;
} OspreyReferenceCase;

typedef struct OspreyReferenceResult {
    OspreyStatus status;
    OspreyModelValidationError error;
    uint64_t discarded_hard_false;
    uint64_t discarded_threshold;
    uint64_t discarded_role;
    uint64_t discarded_layout;

    uint32_t object_count;
    uint32_t type_count;
    uint32_t field_count;
    uint32_t chunk_index_count;
    uint32_t type_name_count;
    OspreyDecodedObject *objects;
    OspreyDecodedType *types;
    OspreyDecodedField *fields;
    char **type_names;
    OspreyModelIndexEntry *chunk_index;
    OspreyModelIndexEntry *aggregate_index;
    OspreyModelIndexEntry *type_index;
    uint32_t aggregate_index_count;
    uint32_t type_index_count;
} OspreyReferenceResult;

/* Decode one declarative case.  The result owns all returned arrays and must
 * be released with osprey_reference_result_free(). */
bool osprey_reference_decode(const OspreyReferenceCase *input,
                             OspreyReferenceResult *result);
void osprey_reference_result_free(OspreyReferenceResult *result);

/* Compare every semantic model record and canonical index entry.  Runtime
 * spans are intentionally outside the oracle's semantic result. */
bool osprey_reference_model_equal(const OspreyReferenceResult *reference,
                                  const OspreyModel *model);
bool osprey_reference_model_dump_file(const OspreyReferenceResult *reference,
                                      FILE *out);

#endif /* BINRADAR_STAGE6_DECODE_REFERENCE_H */
