/*
 * OSPREY posterior decoder.
 *
 * Stage 6.1 adds the canonical owned belief projection below without changing
 * the legacy production selection path.  Stage 6.2 selects provisional
 * scalar/field/pointer roles; Stage 6.3 validates and schedules arrays before
 * finalizing storage ownership.  Stage 6.4 owns the production clean cutover.
 * The complete decoder design is reference §10:
 *
 * Plan §10 (design choice):
 *  1. Discard hard-false candidates and posterior < report_threshold.
 *  2. Per chunk, choose scalar or field storage subject to exclusivity and
 *     retain pointer targets independently; arrays finalize eligible members.
 *  3. Group FieldOf(v,a) by base a, sort by offset, retain
 *     non-overlapping field layouts.
 *  4. Validate arrays per (region,stride), then run one region-wide weighted
 *     interval schedule across all strides using adjusted logit scores.
 *  5. At most one target base per pointer chunk.
 *  6. Deterministic names: struct_H_<site>, array_H_<site>, etc.
 *  7. Emit width-preserving placeholders (uint64_t/byte[8]/void *)
 *     for primitive chunks; the pointer/spatial decoding is what the
 *     binradar consumer needs (pointer -> target allocation, size).
 *  8. Every output carries its posterior.
 */

#include "osprey.h"
#include "osprey-internal.h"

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Diagnostic sink (snapshot.c). */
void log_msg(const char *fmt, ...);

#define OSPREY_DECODE_MAX_FIELDS_PER_BASE 256u
#define OSPREY_DECODE_MAX_ARRAYS_PER_SIDE 512u

/* ------------------------------------------------------------------ */
/* Stage 6.1 canonical decoder input                                   */
/* ------------------------------------------------------------------ */

static int64_t osprey_decode_alloc_fail_after = -1;

typedef struct OspreyDecodeAllocator {
    size_t total_bytes;
} OspreyDecodeAllocator;

void osprey_decode_test_set_alloc_fail_after(int64_t allocations)
{
    osprey_decode_alloc_fail_after = allocations;
}

static bool decode_size_mul(size_t count, size_t element_size,
                            size_t *bytes_out)
{
    if (bytes_out == NULL) return false;
    if (element_size != 0 && count > SIZE_MAX / element_size) return false;
    *bytes_out = count * element_size;
    return true;
}

static bool decode_size_add(size_t left, size_t right, size_t *out)
{
    if (out == NULL || right > SIZE_MAX - left) return false;
    *out = left + right;
    return true;
}

static void *decode_alloc(OspreyDecodeAllocator *allocator, size_t count,
                          size_t element_size)
{
    size_t bytes;

    if (allocator == NULL || !decode_size_mul(count, element_size, &bytes)) {
        return NULL;
    }
    if (bytes == 0) return NULL;
    if (allocator->total_bytes > SIZE_MAX - bytes) return NULL;
    if (osprey_decode_alloc_fail_after == 0) return NULL;
    if (osprey_decode_alloc_fail_after > 0) {
        osprey_decode_alloc_fail_after--;
    }
    allocator->total_bytes += bytes;
    return g_try_malloc0(bytes);
}

static int decode_cmp_u64(uint64_t a, uint64_t b)
{
    return a < b ? -1 : a != b;
}

static int decode_cmp_i64(int64_t a, int64_t b)
{
    return a < b ? -1 : a != b;
}

static int decode_region_compare(const OspreyRegionId *a,
                                 const OspreyRegionId *b)
{
    int c = decode_cmp_u64((uint64_t)a->kind, (uint64_t)b->kind);
    if (c != 0) return c;
    c = decode_cmp_u64(a->code_image_id, b->code_image_id);
    if (c != 0) return c;
    return decode_cmp_u64(a->site_offset, b->site_offset);
}

static int decode_address_compare(const OspreyAddress *a,
                                  const OspreyAddress *b)
{
    int c = decode_region_compare(&a->region, &b->region);
    return c != 0 ? c : decode_cmp_i64(a->offset, b->offset);
}

static int decode_extent_compare(const void *ap, const void *bp)
{
    const OspreyRegionExtent *a = ap;
    const OspreyRegionExtent *b = bp;
    return decode_region_compare(&a->region, &b->region);
}

static int decode_chunk_compare(const OspreyChunk *a, const OspreyChunk *b)
{
    int c = decode_address_compare(&a->address, &b->address);
    return c != 0 ? c : decode_cmp_u64(a->size, b->size);
}

static int decode_key_compare(const OspreyKey *a, const OspreyKey *b)
{
    int c = decode_cmp_u64(a->tag, b->tag);
    if (c != 0) return c;
    for (size_t i = 0; i < G_N_ELEMENTS(a->w); i++) {
        c = decode_cmp_u64(a->w[i], b->w[i]);
        if (c != 0) return c;
    }
    return 0;
}

static int decode_candidate_compare(const void *ap, const void *bp)
{
    const OspreyDecodeCandidate *a = ap;
    const OspreyDecodeCandidate *b = bp;
    int c = decode_cmp_u64(a->predicate_kind, b->predicate_kind);
    if (c != 0) return c;
    c = osprey_var_payload_compare(a->predicate_kind, &a->payload,
                                   &b->payload);
    if (c != 0) return c;
    return decode_key_compare(&a->key, &b->key);
}

static bool decode_score_valid(const OspreyDecodeScore *score)
{
    return score != NULL && score->negative_infinite <= 1 &&
           isfinite(score->finite);
}

bool osprey_decode_score_add(OspreyDecodeScore *score,
                             const OspreyDecodeScore *term)
{
    int64_t balance;

    if (!decode_score_valid(score) || !decode_score_valid(term)) return false;
    if (term->negative_infinite) {
        score->negative_infinite = 1;
        return true;
    }
    if (score->negative_infinite) return true;
    if (!osprey_check_add(score->infinity_balance,
                          term->infinity_balance, &balance)) {
        return false;
    }
    double finite = score->finite + term->finite;
    if (!isfinite(finite)) return false;
    score->infinity_balance = balance;
    score->finite = finite == 0.0 ? 0.0 : finite;
    return true;
}

int osprey_decode_score_compare(const OspreyDecodeScore *left,
                                const OspreyDecodeScore *right)
{
    if (left == NULL || right == NULL) return 0;
    if (left->negative_infinite != right->negative_infinite) {
        return left->negative_infinite ? -1 : 1;
    }
    if (left->negative_infinite) return 0;
    if (left->infinity_balance != right->infinity_balance) {
        return left->infinity_balance < right->infinity_balance ? -1 : 1;
    }
    return left->finite < right->finite ? -1 :
        (left->finite != right->finite ? 1 : 0);
}

typedef struct DecodeVarRef {
    OspreyKey key;
    uint32_t graph_id;
} DecodeVarRef;

static int decode_var_ref_compare(const void *ap, const void *bp)
{
    const DecodeVarRef *a = ap;
    const DecodeVarRef *b = bp;
    int c = decode_key_compare(&a->key, &b->key);
    return c != 0 ? c : decode_cmp_u64(a->graph_id, b->graph_id);
}

typedef struct DecodeChunkSortRef {
    OspreyChunk chunk;
    uint32_t ordinal;
    uint8_t family;
} DecodeChunkSortRef;

static int decode_chunk_sort_ref_compare(const void *ap, const void *bp)
{
    const DecodeChunkSortRef *a = ap;
    const DecodeChunkSortRef *b = bp;
    int c = decode_chunk_compare(&a->chunk, &b->chunk);
    if (c != 0) return c;
    c = decode_cmp_u64(a->family, b->family);
    return c != 0 ? c : decode_cmp_u64(a->ordinal, b->ordinal);
}

typedef struct DecodeFieldSortRef {
    OspreyAddress base;
    uint32_t ordinal;
} DecodeFieldSortRef;

static int decode_field_sort_ref_compare(const void *ap, const void *bp)
{
    const DecodeFieldSortRef *a = ap;
    const DecodeFieldSortRef *b = bp;
    int c = decode_address_compare(&a->base, &b->base);
    return c != 0 ? c : decode_cmp_u64(a->ordinal, b->ordinal);
}

typedef struct DecodeArraySortRef {
    OspreyRegionId region;
    uint64_t stride;
    uint32_t ordinal;
} DecodeArraySortRef;

static int decode_array_region_sort_ref_compare(const void *ap,
                                                const void *bp)
{
    const DecodeArraySortRef *a = ap;
    const DecodeArraySortRef *b = bp;
    int c = decode_region_compare(&a->region, &b->region);
    return c != 0 ? c : decode_cmp_u64(a->ordinal, b->ordinal);
}

static int decode_array_stride_sort_ref_compare(const void *ap,
                                                const void *bp)
{
    const DecodeArraySortRef *a = ap;
    const DecodeArraySortRef *b = bp;
    int c = decode_region_compare(&a->region, &b->region);
    if (c != 0) return c;
    c = decode_cmp_u64(a->stride, b->stride);
    return c != 0 ? c : decode_cmp_u64(a->ordinal, b->ordinal);
}

static bool decode_region_valid(const OspreyRegionId *region)
{
    return region != NULL && region->kind >= OSPREY_REGION_GLOBAL &&
           region->kind <= OSPREY_REGION_STACK_FUNCTION;
}

static bool decode_projected_kind(uint8_t kind)
{
    return kind == OSPREY_PRED_PRIMITIVE_VAR ||
           kind == OSPREY_PRED_SCALAR || kind == OSPREY_PRED_ARRAY ||
           kind == OSPREY_PRED_FIELD_OF || kind == OSPREY_PRED_POINTER;
}

static bool decode_u32_add(uint32_t left, uint32_t right, uint32_t *out)
{
    if (out == NULL || right > UINT32_MAX - left) return false;
    *out = left + right;
    return true;
}

static bool decode_index_matches(gpointer indexed, uint32_t expected)
{
    uintptr_t raw = (uintptr_t)indexed;
    return raw != 0 && raw - 1u <= UINT32_MAX &&
           (uint32_t)(raw - 1u) == expected;
}

static void decode_factor_key(const OspreyFactor *factor,
                              OspreyFactorKey *key)
{
    memset(key, 0, sizeof(*key));
    key->rule = factor->rule;
    key->stage = factor->stage;
    key->potential_kind = factor->potential_kind;
    key->negative = factor->negative;
    key->head_idx = factor->head_idx;
    memcpy(&key->p_bits, &factor->p, sizeof(key->p_bits));
    key->num_vars = factor->num_vars;
    for (uint32_t i = 0; i < factor->num_vars &&
         i < OSPREY_FACTOR_MAX_ARITY; i++) {
        key->var_ids[i] = factor->var_ids[i];
    }
}

static bool decode_rule_stage_valid(const OspreyFactor *factor)
{
    bool base_rule;

    if (factor == NULL) return false;
    base_rule = factor->rule >= OSPREY_RULE_CA01 &&
                factor->rule <= OSPREY_RULE_CA08;
    return (factor->stage == OSPREY_GRAPH_BASE_CA) == base_rule;
}

static bool decode_factor_valid(const OspreyGraph *graph,
                                const OspreyFactor *factor,
                                uint32_t factor_id, uint32_t variable_count,
                                uint32_t *hard_counts)
{
    if (factor == NULL || factor->id != factor_id ||
        !decode_rule_stage_valid(factor) ||
        factor->rule <= OSPREY_RULE_NONE ||
        factor->rule >= OSPREY_RULE_COUNT ||
        (factor->stage != OSPREY_GRAPH_BASE_CA &&
         factor->stage != OSPREY_GRAPH_SECONDARY) ||
        (factor->potential_kind != OSPREY_POTENTIAL_IMPLICATION &&
         factor->potential_kind != OSPREY_POTENTIAL_PRIOR &&
         factor->potential_kind != OSPREY_POTENTIAL_HARD_FALSE) ||
        factor->negative > 1 || !isfinite(factor->p) || factor->p < 0.0 ||
        factor->p > 1.0 || factor->num_vars == 0 ||
        factor->num_vars > OSPREY_FACTOR_MAX_ARITY ||
        factor->var_ids == NULL || graph->factor_index == NULL) {
        return false;
    }

    bool hard_false = factor->potential_kind == OSPREY_POTENTIAL_HARD_FALSE;
    if (hard_false) {
        uint64_t p_bits = 0;
        memcpy(&p_bits, &factor->p, sizeof(p_bits));
        if (factor->rule != OSPREY_RULE_CB06 ||
            factor->stage != OSPREY_GRAPH_SECONDARY ||
            factor->num_vars != 1 || factor->head_idx != UINT16_MAX ||
            factor->negative || p_bits != 0) {
            return false;
        }
    } else if (factor->rule == OSPREY_RULE_CB06) {
        return false;
    }

    switch (factor->potential_kind) {
    case OSPREY_POTENTIAL_PRIOR:
        if (factor->num_vars != 1 || factor->head_idx != 0) return false;
        break;
    case OSPREY_POTENTIAL_IMPLICATION:
        if (factor->num_vars < 2 || factor->head_idx >= factor->num_vars) {
            return false;
        }
        break;
    case OSPREY_POTENTIAL_HARD_FALSE:
        break;
    default:
        return false;
    }

    for (uint32_t i = 0; i < factor->num_vars; i++) {
        uint32_t id = factor->var_ids[i];
        if (id >= variable_count) return false;
        for (uint32_t j = 0; j < i; j++) {
            if (factor->var_ids[j] == id) return false;
        }
    }

    if (hard_false) {
        const OspreyVar *variable = &g_array_index(
            graph->vars, OspreyVar, factor->var_ids[0]);
        if (variable->kind != OSPREY_PRED_ARRAY ||
            hard_counts[factor->var_ids[0]] == UINT32_MAX) {
            return false;
        }
        hard_counts[factor->var_ids[0]]++;
    }

    OspreyFactorKey key;
    decode_factor_key(factor, &key);
    gpointer indexed = g_hash_table_lookup(graph->factor_index, &key);
    return decode_index_matches(indexed, factor_id);
}

static bool decode_validate_graph(const OspreyContext *ctx,
                                  OspreyDecodeAllocator *allocator,
                                  DecodeVarRef **refs_out,
                                  uint32_t **hard_counts_out,
                                  uint32_t *variable_count_out)
{
    const OspreyGraph *graph;
    uint32_t variable_count;
    uint32_t factor_count;
    DecodeVarRef *refs = NULL;
    uint32_t *hard_counts = NULL;

    if (refs_out == NULL || hard_counts_out == NULL ||
        variable_count_out == NULL) return false;
    *refs_out = NULL;
    *hard_counts_out = NULL;
    *variable_count_out = 0;
    if (ctx == NULL || allocator == NULL ||
        !isfinite(ctx->config.report_threshold) ||
        ctx->config.report_threshold < 0.0 ||
        ctx->config.report_threshold > 1.0 ||
        ctx->graph == NULL) return false;
    graph = ctx->graph;
    if (graph->vars == NULL || graph->extents == NULL ||
        !graph->extents_built || graph->factors == NULL ||
        graph->var_index == NULL ||
        graph->factor_index == NULL || graph->vars->len > UINT32_MAX ||
        graph->factors->len > UINT32_MAX ||
        graph->vars->len > ctx->config.max_variables ||
        graph->factors->len > ctx->config.max_factors ||
        (graph->vars->len != 0 && graph->vars->data == NULL) ||
        (graph->factors->len != 0 && graph->factors->data == NULL) ||
        (graph->extents->len != 0 && graph->extents->data == NULL) ||
        g_hash_table_size(graph->var_index) != graph->vars->len ||
        g_hash_table_size(graph->factor_index) != graph->factors->len) {
        return false;
    }

    variable_count = (uint32_t)graph->vars->len;
    factor_count = (uint32_t)graph->factors->len;
    refs = decode_alloc(allocator, variable_count, sizeof(*refs));
    hard_counts = decode_alloc(allocator, variable_count, sizeof(*hard_counts));
    if ((variable_count != 0 && (refs == NULL || hard_counts == NULL))) {
        g_free(refs);
        g_free(hard_counts);
        return false;
    }

    for (uint32_t i = 0; i < variable_count; i++) {
        const OspreyVar *variable = &g_array_index(graph->vars, OspreyVar, i);
        if (variable->id != i || variable->kind <= OSPREY_PRED_NONE ||
            variable->kind >= OSPREY_PRED_COUNT ||
            !osprey_var_payload_valid(variable->kind, &variable->payload) ||
            variable->belief_valid != 1 || !isfinite(variable->belief) ||
            variable->belief < 0.0 || variable->belief > 1.0 ||
            variable->hard_false > 1) {
            g_free(refs);
            g_free(hard_counts);
            return false;
        }
        refs[i].key = osprey_var_key(variable->kind, &variable->payload);
        refs[i].graph_id = i;
        gpointer indexed = g_hash_table_lookup(graph->var_index, &refs[i].key);
        if (!decode_index_matches(indexed, i)) {
            g_free(refs);
            g_free(hard_counts);
            return false;
        }
    }

    if (variable_count > 1) {
        qsort(refs, variable_count, sizeof(*refs), decode_var_ref_compare);
        for (uint32_t i = 1; i < variable_count; i++) {
            if (decode_key_compare(&refs[i - 1].key, &refs[i].key) == 0) {
                g_free(refs);
                g_free(hard_counts);
                return false;
            }
        }
    }

    for (uint32_t i = 0; i < factor_count; i++) {
        const OspreyFactor *factor = g_array_index(graph->factors,
                                                    OspreyFactor *, i);
        if (!decode_factor_valid(graph, factor, i, variable_count,
                                 hard_counts)) {
            g_free(refs);
            g_free(hard_counts);
            return false;
        }
    }
    for (uint32_t i = 0; i < variable_count; i++) {
        const OspreyVar *variable = &g_array_index(graph->vars, OspreyVar, i);
        bool expected_hard_false = variable->kind == OSPREY_PRED_ARRAY &&
                                   hard_counts[i] == 1;
        if (hard_counts[i] > 1 || variable->hard_false != expected_hard_false) {
            g_free(refs);
            g_free(hard_counts);
            return false;
        }
    }

    *refs_out = refs;
    *hard_counts_out = hard_counts;
    *variable_count_out = variable_count;
    return true;
}

static bool decode_copy_extents(const OspreyGraph *graph,
                                OspreyDecodeInput *input,
                                OspreyDecodeAllocator *allocator)
{
    if (graph == NULL || input == NULL || allocator == NULL ||
        graph->extents == NULL || graph->extents->len > UINT32_MAX) {
        return false;
    }
    input->extent_count = (uint32_t)graph->extents->len;
    input->extents = decode_alloc(allocator, input->extent_count,
                                  sizeof(*input->extents));
    if (input->extent_count != 0 && input->extents == NULL) return false;
    for (uint32_t i = 0; i < input->extent_count; i++) {
        const OspreyRegionExtent *extent = &g_array_index(
            graph->extents, OspreyRegionExtent, i);
        if (!decode_region_valid(&extent->region) || extent->lo > extent->hi) {
            return false;
        }
        input->extents[i] = *extent;
    }
    if (input->extent_count > 1) {
        qsort(input->extents, input->extent_count, sizeof(*input->extents),
              decode_extent_compare);
    }
    for (uint32_t i = 1; i < input->extent_count; i++) {
        if (decode_region_compare(&input->extents[i - 1].region,
                                  &input->extents[i].region) == 0) {
            return false;
        }
    }
    return true;
}

static uint32_t *decode_family_count(OspreyDecodeInput *input,
                                     uint8_t kind)
{
    switch (kind) {
    case OSPREY_PRED_PRIMITIVE_VAR: return &input->primitive_count;
    case OSPREY_PRED_SCALAR: return &input->scalar_count;
    case OSPREY_PRED_ARRAY: return &input->array_count;
    case OSPREY_PRED_FIELD_OF: return &input->field_count;
    case OSPREY_PRED_POINTER: return &input->pointer_count;
    default: return NULL;
    }
}

static OspreyDecodeCandidate *decode_family_data(OspreyDecodeInput *input,
                                                  uint8_t kind)
{
    switch (kind) {
    case OSPREY_PRED_PRIMITIVE_VAR: return input->primitive_candidates;
    case OSPREY_PRED_SCALAR: return input->scalar_candidates;
    case OSPREY_PRED_ARRAY: return input->array_candidates;
    case OSPREY_PRED_FIELD_OF: return input->field_candidates;
    case OSPREY_PRED_POINTER: return input->pointer_candidates;
    default: return NULL;
    }
}

static bool decode_count_candidates(const OspreyGraph *graph,
                                    OspreyDecodeInput *input,
                                    uint32_t variable_count,
                                    double threshold)
{
    for (uint32_t i = 0; i < variable_count; i++) {
        const OspreyVar *variable = &g_array_index(graph->vars, OspreyVar, i);
        if (!decode_projected_kind(variable->kind)) continue;
        if (variable->hard_false) {
            input->discarded_hard_false++;
            continue;
        }
        if (variable->belief < threshold) {
            input->discarded_threshold++;
            continue;
        }
        uint32_t *count = decode_family_count(input, variable->kind);
        if (count == NULL || *count == UINT32_MAX) return false;
        (*count)++;
    }
    return true;
}

static void decode_set_candidate(OspreyDecodeCandidate *candidate,
                                 const OspreyVar *variable)
{
    memset(candidate, 0, sizeof(*candidate));
    candidate->key = osprey_var_key(variable->kind, &variable->payload);
    candidate->payload = variable->payload;
    candidate->source_graph_id = variable->id;
    candidate->predicate_kind = variable->kind;
    candidate->posterior = variable->belief;
    memcpy(&candidate->posterior_bits, &variable->belief,
           sizeof(candidate->posterior_bits));
    candidate->direct_support = variable->direct_support;
    candidate->source_rule_bits = variable->source_rule_bits;
}

static bool decode_fill_candidates(const OspreyGraph *graph,
                                   OspreyDecodeInput *input,
                                   uint32_t variable_count,
                                   double threshold)
{
    uint32_t positions[OSPREY_PRED_COUNT];
    memset(positions, 0, sizeof(positions));
    for (uint32_t i = 0; i < variable_count; i++) {
        const OspreyVar *variable = &g_array_index(graph->vars, OspreyVar, i);
        if (!decode_projected_kind(variable->kind) || variable->hard_false ||
            variable->belief < threshold) continue;
        uint32_t *count = decode_family_count(input, variable->kind);
        OspreyDecodeCandidate *family = decode_family_data(input,
                                                            variable->kind);
        if (count == NULL || family == NULL ||
            positions[variable->kind] >= *count) return false;
        decode_set_candidate(&family[positions[variable->kind]++], variable);
    }
    if (positions[OSPREY_PRED_PRIMITIVE_VAR] != input->primitive_count ||
        positions[OSPREY_PRED_SCALAR] != input->scalar_count ||
        positions[OSPREY_PRED_ARRAY] != input->array_count ||
        positions[OSPREY_PRED_FIELD_OF] != input->field_count ||
        positions[OSPREY_PRED_POINTER] != input->pointer_count) {
        return false;
    }

    OspreyDecodeCandidate *families[] = {
        input->primitive_candidates, input->scalar_candidates,
        input->array_candidates, input->field_candidates,
        input->pointer_candidates,
    };
    uint32_t counts[] = {
        input->primitive_count, input->scalar_count, input->array_count,
        input->field_count, input->pointer_count,
    };
    for (size_t i = 0; i < G_N_ELEMENTS(families); i++) {
        if (counts[i] > 1) {
            qsort(families[i], counts[i], sizeof(*families[i]),
                  decode_candidate_compare);
        }
        for (uint32_t j = 1; j < counts[i]; j++) {
            if (decode_key_compare(&families[i][j - 1].key,
                                   &families[i][j].key) == 0) return false;
        }
    }
    return true;
}

static bool decode_build_indexes(OspreyDecodeInput *input,
                                 OspreyDecodeAllocator *allocator)
{
    uint32_t chunk_count = 0;
    if (!decode_u32_add(input->primitive_count, input->scalar_count,
                        &chunk_count) ||
        !decode_u32_add(chunk_count, input->field_count, &chunk_count) ||
        !decode_u32_add(chunk_count, input->pointer_count, &chunk_count)) {
        return false;
    }
    input->chunk_candidate_count = chunk_count;
    input->chunk_candidates = decode_alloc(allocator, chunk_count,
                                           sizeof(*input->chunk_candidates));
    input->chunk_ranges = decode_alloc(allocator, chunk_count,
                                       sizeof(*input->chunk_ranges));
    if (chunk_count != 0 && (input->chunk_candidates == NULL ||
                             input->chunk_ranges == NULL)) return false;

    DecodeChunkSortRef *chunk_refs = decode_alloc(allocator, chunk_count,
                                                   sizeof(*chunk_refs));
    if (chunk_count != 0 && chunk_refs == NULL) return false;
    uint32_t chunk_pos = 0;
#define ADD_CHUNK_REFS(_data, _count, _family)                              \
    for (uint32_t _i = 0; _i < (_count); _i++) {                           \
        chunk_refs[chunk_pos].chunk = (_data)[_i].predicate_kind ==          \
            OSPREY_PRED_FIELD_OF || (_data)[_i].predicate_kind ==           \
            OSPREY_PRED_POINTER ? (_data)[_i].payload.attached.chunk :      \
            (_data)[_i].payload.chunk;                                     \
        chunk_refs[chunk_pos].ordinal = _i;                                 \
        chunk_refs[chunk_pos].family = (_family);                            \
        chunk_pos++;                                                         \
    }
    ADD_CHUNK_REFS(input->primitive_candidates, input->primitive_count,
                   OSPREY_DECODE_FAMILY_PRIMITIVE);
    ADD_CHUNK_REFS(input->scalar_candidates, input->scalar_count,
                   OSPREY_DECODE_FAMILY_SCALAR);
    ADD_CHUNK_REFS(input->field_candidates, input->field_count,
                   OSPREY_DECODE_FAMILY_FIELD);
    ADD_CHUNK_REFS(input->pointer_candidates, input->pointer_count,
                   OSPREY_DECODE_FAMILY_POINTER);
#undef ADD_CHUNK_REFS
    if (chunk_pos != chunk_count) {
        g_free(chunk_refs);
        return false;
    }
    if (chunk_count > 1) {
        qsort(chunk_refs, chunk_count, sizeof(*chunk_refs),
              decode_chunk_sort_ref_compare);
    }
    for (uint32_t i = 0; i < chunk_count; i++) {
        input->chunk_candidates[i].ordinal = chunk_refs[i].ordinal;
        input->chunk_candidates[i].family = chunk_refs[i].family;
        if (i == 0 || decode_chunk_compare(&chunk_refs[i - 1].chunk,
                                           &chunk_refs[i].chunk) != 0) {
            OspreyDecodeChunkRange *range =
                &input->chunk_ranges[input->chunk_range_count++];
            range->chunk = chunk_refs[i].chunk;
            range->begin = i;
            range->count = 1;
        } else {
            input->chunk_ranges[input->chunk_range_count - 1].count++;
        }
    }
    g_free(chunk_refs);

    uint32_t field_count = input->field_count;
    input->field_by_base_count = field_count;
    input->field_by_base = decode_alloc(allocator, field_count,
                                         sizeof(*input->field_by_base));
    input->field_base_ranges = decode_alloc(allocator, field_count,
                                            sizeof(*input->field_base_ranges));
    if (field_count != 0 && (input->field_by_base == NULL ||
                             input->field_base_ranges == NULL)) return false;
    DecodeFieldSortRef *field_refs = decode_alloc(allocator, field_count,
                                                  sizeof(*field_refs));
    if (field_count != 0 && field_refs == NULL) return false;
    for (uint32_t i = 0; i < field_count; i++) {
        field_refs[i].base = input->field_candidates[i].payload.attached.base;
        field_refs[i].ordinal = i;
    }
    if (field_count > 1) {
        qsort(field_refs, field_count, sizeof(*field_refs),
              decode_field_sort_ref_compare);
    }
    for (uint32_t i = 0; i < field_count; i++) {
        input->field_by_base[i] = field_refs[i].ordinal;
        if (i == 0 || decode_address_compare(&field_refs[i - 1].base,
                                             &field_refs[i].base) != 0) {
            OspreyDecodeBaseRange *range =
                &input->field_base_ranges[input->field_base_range_count++];
            range->base = field_refs[i].base;
            range->begin = i;
            range->count = 1;
        } else {
            input->field_base_ranges[input->field_base_range_count - 1].count++;
        }
    }
    g_free(field_refs);

    uint32_t array_count = input->array_count;
    input->array_by_region_count = array_count;
    input->array_region_ranges = decode_alloc(
        allocator, array_count, sizeof(*input->array_region_ranges));
    input->array_by_region = decode_alloc(allocator, array_count,
                                          sizeof(*input->array_by_region));
    input->array_by_region_stride_count = array_count;
    input->array_by_region_stride = decode_alloc(
        allocator, array_count, sizeof(*input->array_by_region_stride));
    input->array_region_stride_ranges = decode_alloc(
        allocator, array_count, sizeof(*input->array_region_stride_ranges));
    if (array_count != 0 &&
        (input->array_region_ranges == NULL ||
         input->array_by_region == NULL ||
         input->array_by_region_stride == NULL ||
         input->array_region_stride_ranges == NULL)) {
        return false;
    }
    DecodeArraySortRef *array_refs = decode_alloc(allocator, array_count,
                                                  sizeof(*array_refs));
    if (array_count != 0 && array_refs == NULL) return false;
    for (uint32_t i = 0; i < array_count; i++) {
        array_refs[i].region = input->array_candidates[i].payload.segment.a1.region;
        array_refs[i].stride = (uint64_t)input->array_candidates[i]
            .payload.segment.size;
        array_refs[i].ordinal = i;
    }
    if (array_count > 1) {
        qsort(array_refs, array_count, sizeof(*array_refs),
              decode_array_region_sort_ref_compare);
    }
    for (uint32_t i = 0; i < array_count; i++) {
        input->array_by_region[i] = array_refs[i].ordinal;
        if (i == 0 ||
            decode_region_compare(&array_refs[i - 1].region,
                                  &array_refs[i].region) != 0) {
            OspreyDecodeRegionRange *range =
                &input->array_region_ranges[input->array_region_range_count++];
            range->region = array_refs[i].region;
            range->begin = i;
            range->count = 1;
        } else {
            input->array_region_ranges[
                input->array_region_range_count - 1].count++;
        }
    }

    if (array_count > 1) {
        qsort(array_refs, array_count, sizeof(*array_refs),
              decode_array_stride_sort_ref_compare);
    }
    for (uint32_t i = 0; i < array_count; i++) {
        input->array_by_region_stride[i] = array_refs[i].ordinal;
        if (i == 0 ||
            decode_region_compare(&array_refs[i - 1].region,
                                  &array_refs[i].region) != 0 ||
            array_refs[i - 1].stride != array_refs[i].stride) {
            OspreyDecodeRegionStrideRange *range =
                &input->array_region_stride_ranges[
                    input->array_region_stride_range_count++];
            range->region = array_refs[i].region;
            range->stride = array_refs[i].stride;
            range->begin = i;
            range->count = 1;
        } else {
            input->array_region_stride_ranges[
                input->array_region_stride_range_count - 1].count++;
        }
    }
    g_free(array_refs);
    return true;
}

OspreyStatus osprey_decode_input_build(const OspreyContext *ctx,
                                       OspreyDecodeInput **out)
{
    OspreyDecodeAllocator allocator;
    DecodeVarRef *var_refs = NULL;
    uint32_t *hard_counts = NULL;
    OspreyDecodeInput *input = NULL;
    uint32_t variable_count = 0;
    double threshold;

    if (out == NULL) return OSPREY_INVALID_MODEL;
    *out = NULL;
    memset(&allocator, 0, sizeof(allocator));
    if (!decode_validate_graph(ctx, &allocator, &var_refs, &hard_counts,
                               &variable_count)) {
        g_free(var_refs);
        g_free(hard_counts);
        return OSPREY_INVALID_MODEL;
    }
    input = decode_alloc(&allocator, 1, sizeof(*input));
    if (input == NULL) {
        g_free(var_refs);
        g_free(hard_counts);
        return OSPREY_INVALID_MODEL;
    }
    threshold = ctx->config.report_threshold;
    if (!decode_copy_extents(ctx->graph, input, &allocator) ||
        !decode_count_candidates(ctx->graph, input, variable_count, threshold)) {
        g_free(var_refs);
        g_free(hard_counts);
        osprey_decode_input_free(input);
        return OSPREY_INVALID_MODEL;
    }

#define ALLOC_FAMILY(_kind, _field, _count)                                \
    do {                                                                    \
        input->_field = decode_alloc(&allocator, (_count),                \
                                     sizeof(*input->_field));               \
        if ((_count) != 0 && input->_field == NULL) goto input_failure;     \
    } while (0)
    ALLOC_FAMILY(OSPREY_PRED_PRIMITIVE_VAR, primitive_candidates,
                 input->primitive_count);
    ALLOC_FAMILY(OSPREY_PRED_SCALAR, scalar_candidates, input->scalar_count);
    ALLOC_FAMILY(OSPREY_PRED_ARRAY, array_candidates, input->array_count);
    ALLOC_FAMILY(OSPREY_PRED_FIELD_OF, field_candidates, input->field_count);
    ALLOC_FAMILY(OSPREY_PRED_POINTER, pointer_candidates,
                 input->pointer_count);
#undef ALLOC_FAMILY

    if (!decode_fill_candidates(ctx->graph, input, variable_count, threshold) ||
        !decode_build_indexes(input, &allocator)) {
        goto input_failure;
    }
    g_free(var_refs);
    g_free(hard_counts);
    *out = input;
    return OSPREY_OK;

input_failure:
    g_free(var_refs);
    g_free(hard_counts);
    osprey_decode_input_free(input);
    return OSPREY_INVALID_MODEL;
}

void osprey_decode_input_free(OspreyDecodeInput *input)
{
    if (input == NULL) return;
    g_free(input->primitive_candidates);
    g_free(input->scalar_candidates);
    g_free(input->array_candidates);
    g_free(input->field_candidates);
    g_free(input->pointer_candidates);
    g_free(input->chunk_candidates);
    g_free(input->chunk_ranges);
    g_free(input->field_by_base);
    g_free(input->field_base_ranges);
    g_free(input->array_by_region);
    g_free(input->array_region_ranges);
    g_free(input->array_by_region_stride);
    g_free(input->array_region_stride_ranges);
    g_free(input->extents);
    g_free(input);
}

static bool decode_dump_key(FILE *out, const OspreyKey *key)
{
    if (out == NULL || key == NULL ||
        fprintf(out, "[key 0x%016" PRIx64, key->tag) < 0) return false;
    for (size_t i = 0; i < G_N_ELEMENTS(key->w); i++) {
        if (fprintf(out, " 0x%016" PRIx64, key->w[i]) < 0) return false;
    }
    return fputc(']', out) != EOF;
}

static const OspreyDecodeCandidate *decode_ref_candidate(
    const OspreyDecodeInput *input, const OspreyDecodeCandidateRef *ref)
{
    if (input == NULL || ref == NULL) return NULL;
    switch (ref->family) {
    case OSPREY_DECODE_FAMILY_PRIMITIVE:
        return ref->ordinal < input->primitive_count
            ? &input->primitive_candidates[ref->ordinal] : NULL;
    case OSPREY_DECODE_FAMILY_SCALAR:
        return ref->ordinal < input->scalar_count
            ? &input->scalar_candidates[ref->ordinal] : NULL;
    case OSPREY_DECODE_FAMILY_FIELD:
        return ref->ordinal < input->field_count
            ? &input->field_candidates[ref->ordinal] : NULL;
    case OSPREY_DECODE_FAMILY_POINTER:
        return ref->ordinal < input->pointer_count
            ? &input->pointer_candidates[ref->ordinal] : NULL;
    case OSPREY_DECODE_FAMILY_ARRAY:
        return ref->ordinal < input->array_count
            ? &input->array_candidates[ref->ordinal] : NULL;
    default:
        return NULL;
    }
}

static bool decode_dump_candidate(FILE *out, const char *label,
                                  const OspreyDecodeCandidate *candidate)
{
    if (fprintf(out, "[%s] [kind %u] ", label,
                (unsigned)candidate->predicate_kind) < 0 ||
        !decode_dump_key(out, &candidate->key) ||
        fprintf(out, " [posterior-bits 0x%016" PRIx64 "]"
                     " [support %" PRIu64 "] [source-rules 0x%016" PRIx64 "]\n",
                candidate->posterior_bits, candidate->direct_support,
                candidate->source_rule_bits) < 0) {
        return false;
    }
    return true;
}

static bool decode_dump_family(FILE *out, const char *label,
                               const OspreyDecodeCandidate *candidates,
                               uint32_t count)
{
    for (uint32_t i = 0; i < count; i++) {
        if (!decode_dump_candidate(out, label, &candidates[i])) return false;
    }
    return true;
}

bool osprey_decode_input_dump_file(const OspreyDecodeInput *input, FILE *out)
{
    if (input == NULL || out == NULL) return false;
    if (fprintf(out, "[discarded-hard-false %" PRIu64 "] "
                     "[discarded-threshold %" PRIu64 "]\n",
                input->discarded_hard_false,
                input->discarded_threshold) < 0 ||
        !decode_dump_family(out, "primitive", input->primitive_candidates,
                            input->primitive_count) ||
        !decode_dump_family(out, "scalar", input->scalar_candidates,
                            input->scalar_count) ||
        !decode_dump_family(out, "array", input->array_candidates,
                            input->array_count) ||
        !decode_dump_family(out, "field", input->field_candidates,
                            input->field_count) ||
        !decode_dump_family(out, "pointer", input->pointer_candidates,
                            input->pointer_count)) {
        return false;
    }
    for (uint32_t i = 0; i < input->extent_count; i++) {
        const OspreyRegionExtent *e = &input->extents[i];
        if (fprintf(out, "[extent] [region %u] [image 0x%016" PRIx64 "]"
                         " [site 0x%016" PRIx64 "] [lo %" PRId64 "]"
                         " [hi %" PRId64 "]\n", (unsigned)e->region.kind,
                    e->region.code_image_id, e->region.site_offset,
                    e->lo, e->hi) < 0) return false;
    }
    for (uint32_t i = 0; i < input->chunk_range_count; i++) {
        const OspreyDecodeChunkRange *range = &input->chunk_ranges[i];
        if (fprintf(out, "[chunk-range] ") < 0) return false;
        for (uint32_t j = 0; j < range->count; j++) {
            const OspreyDecodeCandidate *candidate = decode_ref_candidate(
                input, &input->chunk_candidates[range->begin + j]);
            if (candidate == NULL ||
                (j != 0 && fprintf(out, ",") < 0) ||
                !decode_dump_key(out, &candidate->key)) return false;
        }
        if (fputc('\n', out) == EOF) return false;
    }
    for (uint32_t i = 0; i < input->field_base_range_count; i++) {
        const OspreyDecodeBaseRange *range = &input->field_base_ranges[i];
        if (fprintf(out, "[base-range] [region %u] [image 0x%016" PRIx64 "]"
                         " [site 0x%016" PRIx64 "] [offset %" PRId64 "]"
                         " [members", (unsigned)range->base.region.kind,
                    range->base.region.code_image_id,
                    range->base.region.site_offset, range->base.offset) < 0) {
            return false;
        }
        for (uint32_t j = 0; j < range->count; j++) {
            uint32_t ordinal = input->field_by_base[range->begin + j];
            if (ordinal >= input->field_count ||
                (j != 0 && fputc(',', out) == EOF) ||
                !decode_dump_key(out, &input->field_candidates[ordinal].key)) {
                return false;
            }
        }
        if (fprintf(out, "]\n") < 0) return false;
    }
    for (uint32_t i = 0; i < input->array_region_range_count; i++) {
        const OspreyDecodeRegionRange *range = &input->array_region_ranges[i];
        if (fprintf(out, "[array-region-range] [region %u]"
                         " [image 0x%016" PRIx64 "]"
                         " [site 0x%016" PRIx64 "] [members",
                    (unsigned)range->region.kind,
                    range->region.code_image_id,
                    range->region.site_offset) < 0) return false;
        for (uint32_t j = 0; j < range->count; j++) {
            if (range->begin > input->array_by_region_count ||
                j > input->array_by_region_count - range->begin ||
                range->begin + j >= input->array_by_region_count) {
                return false;
            }
            uint32_t ordinal = input->array_by_region[range->begin + j];
            if (ordinal >= input->array_count ||
                (j != 0 && fputc(',', out) == EOF) ||
                !decode_dump_key(out, &input->array_candidates[ordinal].key)) {
                return false;
            }
        }
        if (fprintf(out, "]\n") < 0) return false;
    }
    for (uint32_t i = 0; i < input->array_region_stride_range_count; i++) {
        const OspreyDecodeRegionStrideRange *range =
            &input->array_region_stride_ranges[i];
        if (fprintf(out, "[array-range] [region %u] [image 0x%016" PRIx64 "]"
                         " [site 0x%016" PRIx64 "] [stride %" PRIu64 "]"
                         " [members", (unsigned)range->region.kind,
                    range->region.code_image_id, range->region.site_offset,
                    range->stride) < 0) return false;
        for (uint32_t j = 0; j < range->count; j++) {
            uint32_t ordinal = input->array_by_region_stride[range->begin + j];
            if (ordinal >= input->array_count ||
                (j != 0 && fputc(',', out) == EOF) ||
                !decode_dump_key(out, &input->array_candidates[ordinal].key)) {
                return false;
            }
        }
        if (fprintf(out, "]\n") < 0) return false;
    }
    return ferror(out) == 0;
}

/* ------------------------------------------------------------------ */
/* Stage 6.2 deterministic role selection                             */
/* ------------------------------------------------------------------ */

typedef struct DecodeFieldItem {
    uint32_t candidate_ordinal;
    int64_t start;
    int64_t end;
} DecodeFieldItem;

static bool decode_memory_overlap(const void *left, size_t left_count,
                                  size_t left_size, const void *right,
                                  size_t right_count, size_t right_size)
{
    size_t left_bytes;
    size_t right_bytes;
    uintptr_t left_start;
    uintptr_t right_start;

    if (left_count == 0 || right_count == 0 || left == NULL ||
        right == NULL) {
        return false;
    }
    if (!decode_size_mul(left_count, left_size, &left_bytes) ||
        !decode_size_mul(right_count, right_size, &right_bytes)) {
        return true;
    }
    left_start = (uintptr_t)left;
    right_start = (uintptr_t)right;
    if (left_bytes > UINTPTR_MAX - left_start ||
        right_bytes > UINTPTR_MAX - right_start) {
        return true;
    }
    return left_start < right_start + right_bytes &&
           right_start < left_start + left_bytes;
}

static uint32_t decode_input_chunk_count(const OspreyDecodeInput *input,
                                         bool *ok)
{
    uint32_t count = 0;

    if (ok == NULL || input == NULL) return 0;
    *ok = decode_u32_add(input->primitive_count, input->scalar_count,
                         &count) &&
          decode_u32_add(count, input->field_count, &count) &&
          decode_u32_add(count, input->pointer_count, &count);
    return *ok ? count : 0;
}

static const OspreyDecodeCandidate *decode_family_candidate(
    const OspreyDecodeInput *input, uint8_t family, uint32_t ordinal)
{
    if (input == NULL) return NULL;
    switch (family) {
    case OSPREY_DECODE_FAMILY_PRIMITIVE:
        return ordinal < input->primitive_count
            ? &input->primitive_candidates[ordinal] : NULL;
    case OSPREY_DECODE_FAMILY_SCALAR:
        return ordinal < input->scalar_count
            ? &input->scalar_candidates[ordinal] : NULL;
    case OSPREY_DECODE_FAMILY_FIELD:
        return ordinal < input->field_count
            ? &input->field_candidates[ordinal] : NULL;
    case OSPREY_DECODE_FAMILY_POINTER:
        return ordinal < input->pointer_count
            ? &input->pointer_candidates[ordinal] : NULL;
    default:
        return NULL;
    }
}

static bool decode_family_global_ordinal(const OspreyDecodeInput *input,
                                         uint8_t family, uint32_t ordinal,
                                         uint32_t *out)
{
    uint32_t base;

    if (input == NULL || out == NULL) return false;
    switch (family) {
    case OSPREY_DECODE_FAMILY_PRIMITIVE:
        base = 0;
        break;
    case OSPREY_DECODE_FAMILY_SCALAR:
        base = input->primitive_count;
        break;
    case OSPREY_DECODE_FAMILY_FIELD:
        if (!decode_u32_add(input->primitive_count, input->scalar_count,
                            &base)) return false;
        break;
    case OSPREY_DECODE_FAMILY_POINTER:
        if (!decode_u32_add(input->primitive_count, input->scalar_count,
                            &base) ||
            !decode_u32_add(base, input->field_count, &base)) return false;
        break;
    default:
        return false;
    }
    if (ordinal > UINT32_MAX - base) return false;
    *out = base + ordinal;
    return true;
}

static bool decode_input_family_valid(const OspreyDecodeCandidate *data,
                                      uint32_t count, uint8_t kind)
{
    if ((count != 0 && data == NULL) || (count == 0 && data != NULL)) {
        return false;
    }
    for (uint32_t i = 0; i < count; i++) {
        const OspreyDecodeCandidate *candidate = &data[i];
        OspreyKey expected;
        uint64_t posterior_bits;

        if (candidate->predicate_kind != kind ||
            !osprey_var_payload_valid(kind, &candidate->payload)) {
            return false;
        }
        expected = osprey_var_key(kind, &candidate->payload);
        if (decode_key_compare(&candidate->key, &expected) != 0 ||
            !isfinite(candidate->posterior) || candidate->posterior < 0.0 ||
            candidate->posterior > 1.0) {
            return false;
        }
        memcpy(&posterior_bits, &candidate->posterior,
               sizeof(posterior_bits));
        if (candidate->posterior_bits != posterior_bits) return false;
        if (i != 0 && decode_candidate_compare(&data[i - 1], candidate) >= 0) {
            return false;
        }
    }
    return true;
}

static bool decode_input_aliases_graph(const OspreyContext *ctx,
                                       const OspreyDecodeInput *input)
{
    const OspreyDecodeCandidate *families[5];
    uint32_t counts[5];

    if (ctx == NULL || input == NULL) return true;
    families[0] = input->primitive_candidates;
    families[1] = input->scalar_candidates;
    families[2] = input->array_candidates;
    families[3] = input->field_candidates;
    families[4] = input->pointer_candidates;
    counts[0] = input->primitive_count;
    counts[1] = input->scalar_count;
    counts[2] = input->array_count;
    counts[3] = input->field_count;
    counts[4] = input->pointer_count;

    for (size_t i = 0; i < G_N_ELEMENTS(families); i++) {
        for (size_t j = 0; j < i; j++) {
            if (decode_memory_overlap(families[i], counts[i],
                                      sizeof(*families[i]), families[j],
                                      counts[j], sizeof(*families[j]))) {
                return true;
            }
        }
    }

    const OspreyGraph *graph = ctx->graph;
    if (graph == NULL) return false;
    const void *storage[3] = { NULL, NULL, NULL };
    size_t storage_counts[3] = { 0, 0, 0 };
    size_t storage_sizes[3] = {
        sizeof(OspreyVar), sizeof(OspreyFactor *),
        sizeof(OspreyRegionExtent),
    };
    if (graph->vars != NULL) {
        storage[0] = graph->vars->data;
        storage_counts[0] = graph->vars->len;
    }
    if (graph->factors != NULL) {
        storage[1] = graph->factors->data;
        storage_counts[1] = graph->factors->len;
    }
    if (graph->extents != NULL) {
        storage[2] = graph->extents->data;
        storage_counts[2] = graph->extents->len;
    }
    for (size_t i = 0; i < G_N_ELEMENTS(families); i++) {
        for (size_t j = 0; j < G_N_ELEMENTS(storage); j++) {
            if (decode_memory_overlap(families[i], counts[i],
                                      sizeof(*families[i]), storage[j],
                                      storage_counts[j], storage_sizes[j])) {
                return true;
            }
        }
    }
    return false;
}

static bool decode_input_chunk_index_valid(const OspreyDecodeInput *input,
                                           OspreyDecodeAllocator *allocator)
{
    bool count_ok;
    uint32_t expected = decode_input_chunk_count(input, &count_ok);
    uint32_t next_begin = 0;
    uint8_t *seen;

    if (!count_ok || input == NULL ||
        input->chunk_candidate_count != expected ||
        (expected != 0 && (input->chunk_candidates == NULL ||
                           input->chunk_ranges == NULL)) ||
        (expected == 0 && (input->chunk_candidates != NULL ||
                           input->chunk_ranges != NULL)) ||
        input->chunk_range_count > expected || allocator == NULL) {
        return false;
    }
    seen = decode_alloc(allocator, expected, sizeof(*seen));
    if (expected != 0 && seen == NULL) return false;
    for (uint32_t i = 0; i < input->chunk_range_count; i++) {
        const OspreyDecodeChunkRange *range = &input->chunk_ranges[i];
        uint32_t end;
        const OspreyDecodeCandidate *previous = NULL;

        if (range->count == 0 || range->begin != next_begin ||
            !decode_u32_add(range->begin, range->count, &end) ||
            end > expected ||
            (i != 0 && decode_chunk_compare(
                &input->chunk_ranges[i - 1].chunk, &range->chunk) >= 0)) {
            g_free(seen);
            return false;
        }
        for (uint32_t j = 0; j < range->count; j++) {
            const OspreyDecodeCandidateRef *ref =
                &input->chunk_candidates[range->begin + j];
            const OspreyDecodeCandidate *candidate =
                decode_family_candidate(input, ref->family, ref->ordinal);
            OspreyChunk candidate_chunk;
            uint32_t global_ordinal;

            if (candidate == NULL ||
                !decode_family_global_ordinal(input, ref->family,
                                              ref->ordinal,
                                              &global_ordinal) ||
                global_ordinal >= expected || seen[global_ordinal]) {
                g_free(seen);
                return false;
            }
            candidate_chunk = candidate->predicate_kind ==
                                  OSPREY_PRED_FIELD_OF ||
                              candidate->predicate_kind ==
                                  OSPREY_PRED_POINTER
                ? candidate->payload.attached.chunk
                : candidate->payload.chunk;
            if (decode_chunk_compare(&candidate_chunk, &range->chunk) != 0 ||
                (previous != NULL &&
                 decode_candidate_compare(previous, candidate) >= 0)) {
                g_free(seen);
                return false;
            }
            previous = candidate;
            seen[global_ordinal] = 1;
        }
        next_begin = end;
    }
    if (next_begin != expected) {
        g_free(seen);
        return false;
    }
    for (uint32_t i = 0; i < expected; i++) {
        if (!seen[i]) {
            g_free(seen);
            return false;
        }
    }
    g_free(seen);
    return true;
}

static bool decode_input_field_index_valid(const OspreyDecodeInput *input,
                                           OspreyDecodeAllocator *allocator)
{
    uint32_t next_begin = 0;
    uint8_t *seen;

    if (input == NULL || allocator == NULL ||
        input->field_by_base_count != input->field_count ||
        (input->field_count != 0 && input->field_by_base == NULL) ||
        (input->field_count == 0 && input->field_by_base != NULL) ||
        (input->field_base_range_count != 0 &&
         input->field_base_ranges == NULL) ||
        (input->field_base_range_count == 0 &&
         input->field_base_ranges != NULL)) {
        return false;
    }
    seen = decode_alloc(allocator, input->field_count, sizeof(*seen));
    if (input->field_count != 0 && seen == NULL) return false;
    for (uint32_t i = 0; i < input->field_base_range_count; i++) {
        const OspreyDecodeBaseRange *range = &input->field_base_ranges[i];
        uint32_t end;
        const OspreyDecodeCandidate *previous = NULL;

        if (range->count == 0 || range->begin != next_begin ||
            !decode_u32_add(range->begin, range->count, &end) ||
            end > input->field_by_base_count ||
            (i != 0 && decode_address_compare(
                &input->field_base_ranges[i - 1].base, &range->base) >= 0)) {
            g_free(seen);
            return false;
        }
        for (uint32_t j = 0; j < range->count; j++) {
            uint32_t ordinal = input->field_by_base[range->begin + j];
            const OspreyDecodeCandidate *candidate;
            if (ordinal >= input->field_count || seen[ordinal]) {
                g_free(seen);
                return false;
            }
            candidate = &input->field_candidates[ordinal];
            if (decode_address_compare(&candidate->payload.attached.base,
                                       &range->base) != 0 ||
                (previous != NULL &&
                 decode_candidate_compare(previous, candidate) >= 0)) {
                g_free(seen);
                return false;
            }
            previous = candidate;
            seen[ordinal] = 1;
        }
        next_begin = end;
    }
    if (next_begin != input->field_by_base_count) {
        g_free(seen);
        return false;
    }
    for (uint32_t i = 0; i < input->field_count; i++) {
        if (!seen[i]) {
            g_free(seen);
            return false;
        }
    }
    g_free(seen);
    return true;
}

static bool decode_input_array_indexes_valid(const OspreyDecodeInput *input,
                                             OspreyDecodeAllocator *allocator)
{
    uint32_t next_region_begin = 0;
    uint32_t next_stride_begin = 0;
    uint8_t *region_seen;
    uint8_t *stride_seen;

    if (input == NULL || allocator == NULL ||
        input->array_by_region_count != input->array_count ||
        input->array_by_region_stride_count != input->array_count ||
        (input->array_count != 0 &&
         (input->array_by_region == NULL ||
          input->array_by_region_stride == NULL)) ||
        (input->array_count == 0 &&
         (input->array_by_region != NULL ||
          input->array_by_region_stride != NULL)) ||
        (input->array_region_range_count != 0 &&
         input->array_region_ranges == NULL) ||
        (input->array_region_range_count == 0 &&
         input->array_region_ranges != NULL) ||
        (input->array_region_stride_range_count != 0 &&
         input->array_region_stride_ranges == NULL) ||
        (input->array_region_stride_range_count == 0 &&
         input->array_region_stride_ranges != NULL)) {
        return false;
    }
    region_seen = decode_alloc(allocator, input->array_count,
                               sizeof(*region_seen));
    stride_seen = decode_alloc(allocator, input->array_count,
                               sizeof(*stride_seen));
    if (input->array_count != 0 &&
        (region_seen == NULL || stride_seen == NULL)) {
        g_free(region_seen);
        g_free(stride_seen);
        return false;
    }

    for (uint32_t i = 0; i < input->array_region_range_count; i++) {
        const OspreyDecodeRegionRange *range =
            &input->array_region_ranges[i];
        uint32_t end;
        const OspreyDecodeCandidate *previous = NULL;
        if (range->count == 0 || range->begin != next_region_begin ||
            !decode_u32_add(range->begin, range->count, &end) ||
            end > input->array_by_region_count ||
            (i != 0 && decode_region_compare(
                &input->array_region_ranges[i - 1].region,
                &range->region) >= 0)) {
            g_free(region_seen);
            g_free(stride_seen);
            return false;
        }
        for (uint32_t j = 0; j < range->count; j++) {
            uint32_t ordinal = input->array_by_region[range->begin + j];
            const OspreyDecodeCandidate *candidate;
            if (ordinal >= input->array_count || region_seen[ordinal]) {
                g_free(region_seen);
                g_free(stride_seen);
                return false;
            }
            candidate = &input->array_candidates[ordinal];
            if (decode_region_compare(&candidate->payload.segment.a1.region,
                                      &range->region) != 0 ||
                (previous != NULL &&
                 decode_candidate_compare(previous, candidate) >= 0)) {
                g_free(region_seen);
                g_free(stride_seen);
                return false;
            }
            previous = candidate;
            region_seen[ordinal] = 1;
        }
        next_region_begin = end;
    }
    if (next_region_begin != input->array_by_region_count) {
        g_free(region_seen);
        g_free(stride_seen);
        return false;
    }

    for (uint32_t i = 0; i < input->array_region_stride_range_count; i++) {
        const OspreyDecodeRegionStrideRange *range =
            &input->array_region_stride_ranges[i];
        uint32_t end;
        const OspreyDecodeCandidate *previous = NULL;
        if (range->count == 0 || range->begin != next_stride_begin ||
            !decode_u32_add(range->begin, range->count, &end) ||
            end > input->array_by_region_stride_count ||
            (i != 0 &&
             (decode_region_compare(
                  &input->array_region_stride_ranges[i - 1].region,
                  &range->region) > 0 ||
              (decode_region_compare(
                   &input->array_region_stride_ranges[i - 1].region,
                   &range->region) == 0 &&
               input->array_region_stride_ranges[i - 1].stride >=
                   range->stride)))) {
            g_free(region_seen);
            g_free(stride_seen);
            return false;
        }
        for (uint32_t j = 0; j < range->count; j++) {
            uint32_t ordinal =
                input->array_by_region_stride[range->begin + j];
            const OspreyDecodeCandidate *candidate;
            if (ordinal >= input->array_count || stride_seen[ordinal]) {
                g_free(region_seen);
                g_free(stride_seen);
                return false;
            }
            candidate = &input->array_candidates[ordinal];
            if (decode_region_compare(&candidate->payload.segment.a1.region,
                                      &range->region) != 0 ||
                (uint64_t)candidate->payload.segment.size != range->stride ||
                (previous != NULL &&
                 decode_candidate_compare(previous, candidate) >= 0)) {
                g_free(region_seen);
                g_free(stride_seen);
                return false;
            }
            previous = candidate;
            stride_seen[ordinal] = 1;
        }
        next_stride_begin = end;
    }
    if (next_stride_begin != input->array_by_region_stride_count) {
        g_free(region_seen);
        g_free(stride_seen);
        return false;
    }
    for (uint32_t i = 0; i < input->array_count; i++) {
        if (!region_seen[i] || !stride_seen[i]) {
            g_free(region_seen);
            g_free(stride_seen);
            return false;
        }
    }
    g_free(region_seen);
    g_free(stride_seen);
    return true;
}

static bool decode_input_extents_valid(const OspreyDecodeInput *input)
{
    if (input == NULL || (input->extent_count != 0 &&
                          input->extents == NULL) ||
        (input->extent_count == 0 && input->extents != NULL)) {
        return false;
    }
    for (uint32_t i = 0; i < input->extent_count; i++) {
        const OspreyRegionExtent *extent = &input->extents[i];
        if (!decode_region_valid(&extent->region) || extent->lo > extent->hi ||
            (i != 0 && decode_region_compare(
                &input->extents[i - 1].region, &extent->region) >= 0)) {
            return false;
        }
    }
    return true;
}

static bool decode_input_for_roles_valid(const OspreyContext *ctx,
                                         const OspreyDecodeInput *input,
                                         OspreyDecodeAllocator *allocator)
{
    bool count_ok = false;
    uint32_t projected_count;

    if (ctx == NULL || input == NULL || allocator == NULL ||
        decode_input_aliases_graph(ctx, input)) {
        return false;
    }
    projected_count = decode_input_chunk_count(input, &count_ok);
    if (!count_ok || !decode_u32_add(projected_count, input->array_count,
                                     &projected_count) ||
        projected_count > ctx->config.max_variables ||
        !decode_input_family_valid(input->primitive_candidates,
                                   input->primitive_count,
                                   OSPREY_PRED_PRIMITIVE_VAR) ||
        !decode_input_family_valid(input->scalar_candidates,
                                   input->scalar_count, OSPREY_PRED_SCALAR) ||
        !decode_input_family_valid(input->array_candidates,
                                   input->array_count, OSPREY_PRED_ARRAY) ||
        !decode_input_family_valid(input->field_candidates,
                                   input->field_count, OSPREY_PRED_FIELD_OF) ||
        !decode_input_family_valid(input->pointer_candidates,
                                   input->pointer_count, OSPREY_PRED_POINTER) ||
        !decode_input_extents_valid(input) ||
        !decode_input_chunk_index_valid(input, allocator) ||
        !decode_input_field_index_valid(input, allocator) ||
        !decode_input_array_indexes_valid(input, allocator)) {
        return false;
    }
    return true;
}

static const OspreyRegionExtent *decode_input_find_extent(
    const OspreyDecodeInput *input, const OspreyRegionId *region)
{
    if (input == NULL || region == NULL) return NULL;
    for (uint32_t i = 0; i < input->extent_count; i++) {
        if (decode_region_compare(&input->extents[i].region, region) == 0) {
            return &input->extents[i];
        }
    }
    return NULL;
}

static bool decode_chunk_end(const OspreyChunk *chunk, int64_t *end)
{
    if (chunk == NULL || end == NULL || chunk->size == 0 ||
        chunk->size > (uint64_t)INT64_MAX) return false;
    return osprey_check_add(chunk->address.offset,
                            (int64_t)chunk->size, end);
}

static bool decode_field_geometry_valid(const OspreyDecodeInput *input,
                                        const OspreyDecodeCandidate *candidate)
{
    const OspreyAddress *base;
    const OspreyChunk *chunk;
    const OspreyRegionExtent *extent;
    int64_t end;
    int64_t field_end;
    int64_t relative;

    if (input == NULL || candidate == NULL ||
        candidate->predicate_kind != OSPREY_PRED_FIELD_OF) return false;
    chunk = &candidate->payload.attached.chunk;
    base = &candidate->payload.attached.base;
    if (decode_region_compare(&chunk->address.region, &base->region) != 0 ||
        !decode_chunk_end(chunk, &end) ||
        !osprey_check_sub(chunk->address.offset, base->offset, &relative) ||
        relative < 0 ||
        !osprey_check_add(relative, (int64_t)chunk->size, &field_end)) {
        return false;
    }
    extent = decode_input_find_extent(input, &base->region);
    return extent != NULL && base->offset >= extent->lo &&
           base->offset < extent->hi && chunk->address.offset >= extent->lo &&
           end <= extent->hi;
}

static bool decode_pointer_target_valid(const OspreyDecodeInput *input,
                                        const OspreyDecodeCandidate *candidate)
{
    const OspreyAddress *target;
    const OspreyRegionExtent *extent;

    if (input == NULL || candidate == NULL ||
        candidate->predicate_kind != OSPREY_PRED_POINTER ||
        candidate->payload.attached.chunk.size != sizeof(target_ulong)) {
        return false;
    }
    target = &candidate->payload.attached.base;
    extent = decode_input_find_extent(input, &target->region);
    return extent != NULL && target->offset >= extent->lo &&
           target->offset < extent->hi;
}

static bool decode_candidate_better(const OspreyDecodeCandidate *candidate,
                                    const OspreyDecodeCandidate *current)
{
    if (candidate == NULL) return false;
    if (current == NULL) return true;
    return candidate->posterior > current->posterior ||
           (candidate->posterior == current->posterior &&
            decode_candidate_compare(candidate, current) < 0);
}

static bool decode_score_add_probability(OspreyDecodeScore *score,
                                         double probability,
                                         bool subtract)
{
    OspreyDecodeScore term;

    if (score == NULL || !isfinite(probability) || probability < 0.0 ||
        probability > 1.0) return false;
    memset(&term, 0, sizeof(term));
    if (probability == 0.0) {
        /* max(0, logit(0)) is zero for a displacement penalty. */
        if (subtract) return true;
        term.negative_infinite = 1;
    } else if (probability == 1.0) {
        term.infinity_balance = subtract ? -1 : 1;
    } else {
        term.finite = log(probability) - log1p(-probability);
        if (!isfinite(term.finite)) return false;
        if (subtract) term.finite = -term.finite;
    }
    return osprey_decode_score_add(score, &term);
}

typedef enum DecodeArrayGeometryResult {
    DECODE_ARRAY_GEOMETRY_INVALID = 0,
    DECODE_ARRAY_GEOMETRY_NON_DIVISIBLE = 1,
    DECODE_ARRAY_GEOMETRY_VALID = 2,
} DecodeArrayGeometryResult;

typedef struct DecodeArrayWork {
    OspreyDecodeArray array;
    uint32_t input_ordinal;
} DecodeArrayWork;

typedef struct DecodeArrayDisplacement {
    OspreyChunk chunk;
    OspreyKey role_key;
    double posterior;
    uint32_t decision_ordinal;
} DecodeArrayDisplacement;

typedef struct DecodeArrayOrderRef {
    OspreyRegionId region;
    int64_t lo;
    int64_t hi;
    uint64_t stride;
    OspreyKey key;
    uint32_t work_ordinal;
} DecodeArrayOrderRef;

static void decode_array_free_contents(OspreyDecodeArray *array)
{
    if (array == NULL) return;
    g_free(array->member_decision_ordinals);
    g_free(array->displacement_keys);
    array->member_decision_ordinals = NULL;
    array->displacement_keys = NULL;
    array->member_count = 0;
    array->displacement_count = 0;
}

static void decode_array_work_free(DecodeArrayWork *work, uint32_t count)
{
    if (work == NULL) return;
    for (uint32_t i = 0; i < count; i++) {
        decode_array_free_contents(&work[i].array);
    }
    g_free(work);
}

static DecodeArrayGeometryResult decode_array_geometry(
    const OspreyDecodeInput *input, const OspreyDecodeCandidate *candidate,
    OspreyDecodeArray *array)
{
    const OspreyAddress *lo;
    const OspreyAddress *hi;
    const OspreyRegionExtent *extent;
    int64_t span;
    uint64_t stride;
    uint64_t count;

    if (input == NULL || candidate == NULL || array == NULL ||
        candidate->predicate_kind != OSPREY_PRED_ARRAY) {
        return DECODE_ARRAY_GEOMETRY_INVALID;
    }
    lo = &candidate->payload.segment.a1;
    hi = &candidate->payload.segment.a2;
    if (!decode_region_valid(&lo->region) ||
        decode_region_compare(&lo->region, &hi->region) != 0 ||
        lo->offset >= hi->offset || candidate->payload.segment.size <= 0) {
        return DECODE_ARRAY_GEOMETRY_INVALID;
    }
    stride = (uint64_t)candidate->payload.segment.size;
    if (stride > (uint64_t)INT64_MAX ||
        !osprey_check_sub(hi->offset, lo->offset, &span) || span <= 0) {
        return DECODE_ARRAY_GEOMETRY_INVALID;
    }
    if (span < (int64_t)stride) return DECODE_ARRAY_GEOMETRY_INVALID;
    extent = decode_input_find_extent(input, &lo->region);
    if (extent == NULL || lo->offset < extent->lo ||
        hi->offset > extent->hi) {
        return DECODE_ARRAY_GEOMETRY_INVALID;
    }
    if (span % (int64_t)stride != 0) {
        return DECODE_ARRAY_GEOMETRY_NON_DIVISIBLE;
    }
    count = (uint64_t)(span / (int64_t)stride);
    if (count == 0) return DECODE_ARRAY_GEOMETRY_INVALID;

    memset(array, 0, sizeof(*array));
    array->key = candidate->key;
    array->region = lo->region;
    array->lo = lo->offset;
    array->hi = hi->offset;
    array->stride = stride;
    array->count = count;
    array->posterior = candidate->posterior;
    array->posterior_bits = candidate->posterior_bits;
    array->direct_support = candidate->direct_support;
    array->source_rule_bits = candidate->source_rule_bits;
    return DECODE_ARRAY_GEOMETRY_VALID;
}

static bool decode_array_member_info(const OspreyDecodeArray *array,
                                     const OspreyChunk *chunk,
                                     bool *intersects, bool *legal)
{
    int64_t chunk_end;
    int64_t member_delta;

    if (array == NULL || chunk == NULL || intersects == NULL ||
        legal == NULL) return false;
    *intersects = false;
    *legal = false;
    if (decode_region_compare(&chunk->address.region, &array->region) != 0) {
        return true;
    }
    if (!decode_chunk_end(chunk, &chunk_end)) return false;
    *intersects = chunk->address.offset < array->hi &&
                  array->lo < chunk_end;
    if (!*intersects) return true;
    if (!osprey_check_sub(chunk->address.offset, array->lo,
                          &member_delta)) return false;
    *legal = chunk->address.offset >= array->lo &&
             chunk_end <= array->hi && member_delta >= 0 &&
             member_delta % (int64_t)array->stride == 0 &&
             chunk->size <= array->stride;
    return true;
}

static int decode_array_displacement_compare(const void *ap,
                                             const void *bp)
{
    const DecodeArrayDisplacement *a = ap;
    const DecodeArrayDisplacement *b = bp;
    int c = decode_chunk_compare(&a->chunk, &b->chunk);
    if (c != 0) return c;
    c = decode_key_compare(&a->role_key, &b->role_key);
    return c != 0 ? c : decode_cmp_u64(a->decision_ordinal,
                                       b->decision_ordinal);
}

static bool decode_array_build_members(const OspreyDecodeInput *input,
                                       const OspreyDecodePlan *plan,
                                       OspreyDecodeArray *array,
                                       OspreyDecodeAllocator *allocator)
{
    uint32_t member_count = 0;
    uint32_t displacement_count = 0;
    bool hard_conflict = false;
    DecodeArrayDisplacement *displacements = NULL;

    if (input == NULL || plan == NULL || array == NULL || allocator == NULL) {
        return false;
    }
    for (uint32_t i = 0; i < plan->decision_count; i++) {
        const OspreyChunkDecision *decision = &plan->decisions[i];
        bool intersects;
        bool legal;
        if (!decode_array_member_info(array, &decision->chunk, &intersects,
                                      &legal)) return false;
        if (!intersects) continue;
        if (!legal) {
            hard_conflict = true;
            continue;
        }
        if (member_count == UINT32_MAX) return false;
        member_count++;
        if (decision->role_has_predicate &&
            (decision->provisional_role == OSPREY_STORAGE_SCALAR ||
             decision->provisional_role == OSPREY_STORAGE_FIELD)) {
            if (displacement_count == UINT32_MAX) return false;
            displacement_count++;
        }
    }
    if (hard_conflict) return false;

    array->member_decision_ordinals = decode_alloc(
        allocator, member_count, sizeof(*array->member_decision_ordinals));
    array->displacement_keys = decode_alloc(
        allocator, displacement_count, sizeof(*array->displacement_keys));
    if ((member_count != 0 && array->member_decision_ordinals == NULL) ||
        (displacement_count != 0 && array->displacement_keys == NULL)) {
        return false;
    }
    if (displacement_count != 0) {
        displacements = decode_alloc(allocator, displacement_count,
                                     sizeof(*displacements));
        if (displacements == NULL) return false;
    }

    uint32_t member_position = 0;
    uint32_t displacement_position = 0;
    for (uint32_t i = 0; i < plan->decision_count; i++) {
        const OspreyChunkDecision *decision = &plan->decisions[i];
        bool intersects;
        bool legal;
        if (!decode_array_member_info(array, &decision->chunk, &intersects,
                                      &legal)) {
            g_free(displacements);
            return false;
        }
        if (!intersects) continue;
        if (!legal || member_position == member_count) {
            g_free(displacements);
            return false;
        }
        array->member_decision_ordinals[member_position++] = i;
        if (decision->role_has_predicate &&
            (decision->provisional_role == OSPREY_STORAGE_SCALAR ||
             decision->provisional_role == OSPREY_STORAGE_FIELD)) {
            if (displacement_position == displacement_count) {
                g_free(displacements);
                return false;
            }
            displacements[displacement_position].chunk = decision->chunk;
            displacements[displacement_position].role_key = decision->role_key;
            displacements[displacement_position].posterior =
                decision->role_posterior;
            displacements[displacement_position].decision_ordinal = i;
            displacement_position++;
        }
    }
    if (member_position != member_count ||
        displacement_position != displacement_count) {
        g_free(displacements);
        return false;
    }
    if (displacement_count > 1) {
        qsort(displacements, displacement_count, sizeof(*displacements),
              decode_array_displacement_compare);
    }
    OspreyDecodeScore score = { 0 };
    if (!decode_score_add_probability(&score, array->posterior, false)) {
        g_free(displacements);
        return false;
    }
    for (uint32_t i = 0; i < displacement_count; i++) {
        array->displacement_keys[i] = displacements[i].role_key;
        if (displacements[i].posterior > 0.5 &&
            !decode_score_add_probability(&score,
                                          displacements[i].posterior, true)) {
            g_free(displacements);
            return false;
        }
    }
    array->member_count = member_count;
    array->displacement_count = displacement_count;
    array->adjusted_score = score;
    g_free(displacements);
    return true;
}

static bool decode_array_work_build(const OspreyContext *ctx,
                                    const OspreyDecodeInput *input,
                                    const OspreyDecodePlan *plan,
                                    OspreyDecodeAllocator *allocator,
                                    DecodeArrayWork **work_out,
                                    uint32_t *work_count_out,
                                    uint64_t *discarded_layout)
{
    DecodeArrayWork *work = NULL;
    uint32_t work_count = 0;

    if (ctx == NULL || input == NULL || plan == NULL || allocator == NULL ||
        work_out == NULL || work_count_out == NULL ||
        discarded_layout == NULL) return false;
    *work_out = NULL;
    *work_count_out = 0;
    *discarded_layout = 0;
    work = decode_alloc(allocator, input->array_count, sizeof(*work));
    if (input->array_count != 0 && work == NULL) return false;

    for (uint32_t i = 0; i < input->array_count; i++) {
        OspreyDecodeArray array;
        DecodeArrayGeometryResult geometry = decode_array_geometry(
            input, &input->array_candidates[i], &array);
        if (geometry == DECODE_ARRAY_GEOMETRY_INVALID) {
            decode_array_work_free(work, work_count);
            return false;
        }
        if (geometry == DECODE_ARRAY_GEOMETRY_NON_DIVISIBLE) {
            if (*discarded_layout == UINT64_MAX) {
                decode_array_work_free(work, work_count);
                return false;
            }
            (*discarded_layout)++;
            continue;
        }
        if (!decode_array_build_members(input, plan, &array, allocator)) {
            /* A false result is either a hard overlap conflict or an
             * allocation/arithmetic failure.  Recheck the conflict without
             * allocating so the caller can count it as layout loss. */
            bool hard_conflict = false;
            for (uint32_t d = 0; d < plan->decision_count; d++) {
                bool intersects;
                bool legal;
                if (!decode_array_member_info(&array, &plan->decisions[d].chunk,
                                              &intersects, &legal)) {
                    decode_array_free_contents(&array);
                    decode_array_work_free(work, work_count);
                    return false;
                }
                if (intersects && !legal) hard_conflict = true;
            }
            decode_array_free_contents(&array);
            if (!hard_conflict) {
                decode_array_work_free(work, work_count);
                return false;
            }
            if (*discarded_layout == UINT64_MAX) {
                decode_array_work_free(work, work_count);
                return false;
            }
            (*discarded_layout)++;
            continue;
        }
        if (work_count == UINT32_MAX) {
            decode_array_free_contents(&array);
            decode_array_work_free(work, work_count);
            return false;
        }
        work[work_count].array = array;
        work[work_count].input_ordinal = i;
        work_count++;
    }
    *work_out = work;
    *work_count_out = work_count;
    return true;
}

static int decode_array_order_compare(const void *ap, const void *bp,
                                      void *opaque)
{
    const DecodeArrayOrderRef *a = ap;
    const DecodeArrayOrderRef *b = bp;
    (void)opaque;
    int c = decode_region_compare(&a->region, &b->region);
    if (c != 0) return c;
    c = decode_cmp_i64(a->hi, b->hi);
    if (c != 0) return c;
    c = decode_cmp_i64(a->lo, b->lo);
    if (c != 0) return c;
    c = decode_cmp_u64(a->stride, b->stride);
    return c != 0 ? c : decode_key_compare(&a->key, &b->key);
}

static int decode_array_order_compare_qsort(const void *ap, const void *bp)
{
    return decode_array_order_compare(ap, bp, NULL);
}

static bool decode_path_bit_array(const uint64_t *bits, uint32_t ordinal)
{
    return (bits[ordinal / 64u] & (UINT64_C(1) << (ordinal % 64u))) != 0;
}

static bool decode_array_schedule_score(
    const uint64_t *bits, const DecodeArrayOrderRef *items, uint32_t count,
    const uint32_t *key_positions, const DecodeArrayWork *work,
    OspreyDecodeScore *out)
{
    OspreyDecodeScore score = { 0 };
    if (bits == NULL || items == NULL || key_positions == NULL ||
        work == NULL || out == NULL) return false;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t position = key_positions[i];
        if (position >= count) return false;
        if (decode_path_bit_array(bits, position) &&
            !osprey_decode_score_add(&score,
                &work[items[position].work_ordinal].array.adjusted_score)) {
            return false;
        }
    }
    *out = score;
    return true;
}

static bool decode_array_schedule_key_less(
    const uint64_t *left, const uint64_t *right,
    const DecodeArrayOrderRef *items, uint32_t count,
    const uint32_t *key_positions, const DecodeArrayWork *work)
{
    uint32_t left_position = 0;
    uint32_t right_position = 0;

    if (left == NULL || right == NULL || items == NULL || key_positions == NULL ||
        work == NULL) return false;
    for (;;) {
        while (left_position < count &&
               !decode_path_bit_array(left, key_positions[left_position])) {
            left_position++;
        }
        while (right_position < count &&
               !decode_path_bit_array(right, key_positions[right_position])) {
            right_position++;
        }
        if (left_position == count || right_position == count) {
            return left_position == count && right_position != count;
        }
        uint32_t left_work = items[key_positions[left_position]].work_ordinal;
        uint32_t right_work = items[key_positions[right_position]].work_ordinal;
        int c = decode_cmp_u64(work[left_work].input_ordinal,
                               work[right_work].input_ordinal);
        if (c != 0) return c < 0;
        left_position++;
        right_position++;
    }
}

static bool decode_array_order_intersects(const DecodeArrayOrderRef *left,
                                          const DecodeArrayOrderRef *right)
{
    return left->lo < right->hi && right->lo < left->hi;
}

static bool decode_array_state_score(
    const uint8_t *state, const DecodeArrayOrderRef *items, uint32_t count,
    const uint32_t *key_positions, const DecodeArrayWork *work,
    OspreyDecodeScore *out)
{
    OspreyDecodeScore score = { 0 };

    if (state == NULL || items == NULL || key_positions == NULL ||
        work == NULL || out == NULL) return false;
    for (uint32_t key_position = 0; key_position < count; key_position++) {
        uint32_t item = key_positions[key_position];
        if (item >= count) return false;
        if (state[item] == 1 &&
            !osprey_decode_score_add(
                &score,
                &work[items[item].work_ordinal].array.adjusted_score)) {
            return false;
        }
    }
    *out = score;
    return true;
}

/* Return the maximum schedule score under canonical-key prefix constraints.
 * `extra` is one temporarily forced item; all previously forced items are
 * marked in state.  Recomputing every path in complete P08-key order keeps
 * the binary64 result independent of interval-DP traversal order. */
static bool decode_array_constrained_score(
    const DecodeArrayWork *work, const DecodeArrayOrderRef *items,
    uint32_t count, const uint32_t *predecessors,
    const uint32_t *key_positions, const uint8_t *state,
    const uint8_t *blocked, uint32_t extra, uint64_t *path_bits,
    uint64_t *scratch_bits, size_t nwords, size_t row_bytes,
    size_t path_word_count, OspreyDecodeScore *out)
{
    if (work == NULL || items == NULL || predecessors == NULL ||
        key_positions == NULL || state == NULL || blocked == NULL ||
        path_bits == NULL || scratch_bits == NULL || out == NULL ||
        extra >= count || state[extra] != 0 || blocked[extra]) {
        return false;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (state[i] == 1 && i != extra &&
            decode_array_order_intersects(&items[i], &items[extra])) {
            return false;
        }
    }

    memset(path_bits, 0, path_word_count * sizeof(*path_bits));
    for (uint32_t i = 0; i < count; i++) {
        size_t source_row = predecessors[i] == UINT32_MAX
            ? 0 : (size_t)predecessors[i] + 1u;
        size_t exclude_row = i;
        size_t destination_row = (size_t)i + 1u;
        bool extra_conflict = i != extra &&
            decode_array_order_intersects(&items[i], &items[extra]);
        bool can_include = state[i] != 2 && !blocked[i] && !extra_conflict;
        uint64_t *destination = path_bits + destination_row * nwords;
        const uint64_t *exclude = path_bits + exclude_row * nwords;

        memcpy(destination, exclude, row_bytes);
        if (!can_include) continue;
        memcpy(scratch_bits, path_bits + source_row * nwords, row_bytes);
        scratch_bits[i / 64u] |= UINT64_C(1) << (i % 64u);
        OspreyDecodeScore include_score;
        OspreyDecodeScore exclude_score;
        if (!decode_array_schedule_score(scratch_bits, items, count,
                                         key_positions, work,
                                         &include_score) ||
            !decode_array_schedule_score(exclude, items, count,
                                         key_positions, work,
                                         &exclude_score)) {
            return false;
        }
        int comparison = osprey_decode_score_compare(&include_score,
                                                      &exclude_score);
        bool choose_include = state[i] == 1 || i == extra ||
            comparison > 0 ||
            (comparison == 0 && decode_array_schedule_key_less(
                scratch_bits, exclude, items, count, key_positions, work));
        if (choose_include) memcpy(destination, scratch_bits, row_bytes);
    }
    return decode_array_schedule_score(
        path_bits + (size_t)count * nwords, items, count, key_positions, work,
        out);
}

static bool decode_force_array_item(const DecodeArrayOrderRef *items,
                                    uint32_t count, uint8_t *state,
                                    uint8_t *blocked, uint32_t item)
{
    if (items == NULL || state == NULL || blocked == NULL || item >= count ||
        state[item] != 0 || blocked[item]) return false;
    for (uint32_t i = 0; i < count; i++) {
        if (state[i] == 1 && i != item &&
            decode_array_order_intersects(&items[i], &items[item])) {
            return false;
        }
    }
    state[item] = 1;
    for (uint32_t i = 0; i < count; i++) {
        if (i != item &&
            decode_array_order_intersects(&items[i], &items[item])) {
            blocked[i] = 1;
        }
    }
    return true;
}

static bool decode_array_schedule_region(
    const DecodeArrayWork *work, const DecodeArrayOrderRef *items,
    uint32_t count, uint8_t *selected, OspreyDecodeAllocator *allocator)
{
    uint32_t *predecessors = NULL;
    uint32_t *key_positions = NULL;
    uint64_t *path_bits = NULL;
    uint64_t *scratch_bits = NULL;
    OspreyDecodeScore *scores = NULL;
    uint8_t *state = NULL;
    uint8_t *blocked = NULL;
    size_t padded_count;
    size_t nwords;
    size_t rows;
    size_t path_word_count;
    size_t row_bytes;
    bool ok = false;

    if (work == NULL || items == NULL || selected == NULL || allocator == NULL) {
        return false;
    }
    if (count == 0) return true;
    predecessors = decode_alloc(allocator, count, sizeof(*predecessors));
    key_positions = decode_alloc(allocator, count, sizeof(*key_positions));
    if (predecessors == NULL || key_positions == NULL) goto done;
    for (uint32_t i = 0; i < count; i++) {
        key_positions[i] = i;
        uint32_t j = i;
        while (j != 0) {
            uint32_t previous = key_positions[j - 1];
            uint32_t current = key_positions[j];
            if (work[items[previous].work_ordinal].input_ordinal <=
                work[items[current].work_ordinal].input_ordinal) break;
            key_positions[j] = previous;
            key_positions[j - 1] = current;
            j--;
        }
    }
    for (uint32_t i = 0; i < count; i++) {
        uint32_t low = 0;
        uint32_t high = i;
        while (low < high) {
            uint32_t middle = low + (high - low) / 2;
            if (items[middle].hi <= items[i].lo) low = middle + 1;
            else high = middle;
        }
        predecessors[i] = low == 0 ? UINT32_MAX : low - 1;
    }
    if (!decode_size_add((size_t)count, 63u, &padded_count) ||
        !decode_size_add((size_t)count, 1u, &rows)) goto done;
    nwords = padded_count / 64u;
    if (nwords == 0 || !decode_size_mul(rows, nwords, &path_word_count) ||
        !decode_size_mul(nwords, sizeof(*scratch_bits), &row_bytes)) goto done;
    path_bits = decode_alloc(allocator, path_word_count, sizeof(*path_bits));
    scratch_bits = decode_alloc(allocator, nwords, sizeof(*scratch_bits));
    scores = decode_alloc(allocator, rows, sizeof(*scores));
    if (path_bits == NULL || scratch_bits == NULL || scores == NULL) goto done;

    for (uint32_t i = 0; i < count; i++) {
        size_t source_row = predecessors[i] == UINT32_MAX
            ? 0 : (size_t)predecessors[i] + 1u;
        size_t exclude_row = i;
        size_t destination_row = (size_t)i + 1u;
        memcpy(path_bits + destination_row * nwords,
               path_bits + exclude_row * nwords, row_bytes);
        memcpy(scratch_bits, path_bits + source_row * nwords, row_bytes);
        scratch_bits[i / 64u] |= UINT64_C(1) << (i % 64u);
        OspreyDecodeScore include_score;
        if (!decode_array_schedule_score(scratch_bits, items, count,
                                          key_positions, work,
                                          &include_score)) goto done;
        OspreyDecodeScore exclude_score = scores[i];
        int comparison = osprey_decode_score_compare(&include_score,
                                                      &exclude_score);
        bool choose_include = comparison > 0 ||
            (comparison == 0 && decode_array_schedule_key_less(
                scratch_bits, path_bits + exclude_row * nwords, items, count,
                key_positions, work));
        if (choose_include) {
            memcpy(path_bits + destination_row * nwords, scratch_bits, row_bytes);
            scores[destination_row] = include_score;
        } else {
            scores[destination_row] = exclude_score;
        }
    }

    OspreyDecodeScore empty = { 0 };
    OspreyDecodeScore target_score = scores[count];
    if (osprey_decode_score_compare(&target_score, &empty) > 0) {
        state = decode_alloc(allocator, count, sizeof(*state));
        blocked = decode_alloc(allocator, count, sizeof(*blocked));
        if (state == NULL || blocked == NULL) goto done;

        /* Local DP tie choices are not globally compositional: adding a
         * later key can reverse the order of two equal-score prefix vectors.
         * Recover the globally smallest complete P08-key vector by forcing
         * the first feasible key at each canonical position. */
        uint32_t cursor = 0;
        while (cursor < count) {
            OspreyDecodeScore selected_score;
            if (!decode_array_state_score(state, items, count, key_positions,
                                          work, &selected_score)) goto done;
            if (osprey_decode_score_compare(&selected_score,
                                            &target_score) == 0) {
                break;
            }
            bool found = false;
            for (uint32_t key_position = cursor; key_position < count;
                 key_position++) {
                uint32_t item = key_positions[key_position];
                for (uint32_t prior = cursor; prior < key_position; prior++) {
                    uint32_t prior_item = key_positions[prior];
                    if (state[prior_item] == 0) state[prior_item] = 2;
                }
                if (item >= count || state[item] != 0 || blocked[item]) {
                    continue;
                }
                OspreyDecodeScore candidate_score;
                if (!decode_array_constrained_score(
                        work, items, count, predecessors, key_positions,
                        state, blocked, item, path_bits, scratch_bits, nwords,
                        row_bytes, path_word_count, &candidate_score)) {
                    goto done;
                }
                if (osprey_decode_score_compare(&candidate_score,
                                                &target_score) == 0) {
                    if (!decode_force_array_item(items, count, state, blocked,
                                                 item)) {
                        goto done;
                    }
                    cursor = key_position + 1;
                    found = true;
                    break;
                }
            }
            if (!found) goto done;
        }
        OspreyDecodeScore recovered_score;
        if (!decode_array_state_score(state, items, count, key_positions,
                                      work, &recovered_score) ||
            osprey_decode_score_compare(&recovered_score,
                                        &target_score) != 0) {
            goto done;
        }
        for (uint32_t i = 0; i < count; i++) {
            if (state[i] == 1) {
                uint32_t work_ordinal = items[i].work_ordinal;
                selected[work_ordinal] = 1;
            }
        }
    }
    ok = true;

done:
    g_free(predecessors);
    g_free(key_positions);
    g_free(path_bits);
    g_free(scratch_bits);
    g_free(scores);
    g_free(state);
    g_free(blocked);
    return ok;
}

static bool decode_array_work_schedule(
    const DecodeArrayWork *work, uint32_t work_count, uint8_t *selected,
    OspreyDecodeAllocator *allocator)
{
    DecodeArrayOrderRef *order = NULL;
    uint32_t order_count = work_count;

    if ((work_count != 0 && work == NULL) ||
        (work_count != 0 && selected == NULL) || allocator == NULL) {
        return false;
    }
    order = decode_alloc(allocator, order_count, sizeof(*order));
    if (order_count != 0 && order == NULL) return false;
    for (uint32_t i = 0; i < order_count; i++) {
        order[i].region = work[i].array.region;
        order[i].lo = work[i].array.lo;
        order[i].hi = work[i].array.hi;
        order[i].stride = work[i].array.stride;
        order[i].key = work[i].array.key;
        order[i].work_ordinal = i;
    }
    if (order_count > 1) {
        qsort(order, order_count, sizeof(*order),
              decode_array_order_compare_qsort);
    }
    uint32_t begin = 0;
    while (begin < order_count) {
        uint32_t end = begin + 1;
        while (end < order_count &&
               decode_region_compare(&order[begin].region,
                                     &order[end].region) == 0) {
            end++;
        }
        if (!decode_array_schedule_region(work, order + begin, end - begin,
                                          selected, allocator)) {
            g_free(order);
            return false;
        }
        begin = end;
    }
    g_free(order);
    return true;
}

static int decode_field_item_compare(const void *ap, const void *bp)
{
    const DecodeFieldItem *a = ap;
    const DecodeFieldItem *b = bp;
    int c = decode_cmp_i64(a->end, b->end);
    if (c != 0) return c;
    c = decode_cmp_i64(a->start, b->start);
    return c != 0 ? c :
        decode_cmp_u64(a->candidate_ordinal, b->candidate_ordinal);
}

static bool decode_path_bit(const uint64_t *bits, size_t ordinal)
{
    return (bits[ordinal / 64u] & (UINT64_C(1) << (ordinal % 64u))) != 0;
}

static double decode_field_path_score(
    const uint64_t *bits, const DecodeFieldItem *items, uint32_t item_count,
    const uint32_t *key_items, const OspreyDecodeInput *input)
{
    double score = 0.0;

    for (uint32_t key_position = 0; key_position < item_count;
         key_position++) {
        uint32_t item_ordinal = key_items[key_position];
        if (decode_path_bit(bits, item_ordinal)) {
            uint32_t candidate_ordinal = items[item_ordinal].candidate_ordinal;
            score += input->field_candidates[candidate_ordinal].posterior;
        }
    }
    return score == 0.0 ? 0.0 : score;
}

static bool decode_field_path_lex_less(
    const uint64_t *left, const uint64_t *right,
    const DecodeFieldItem *items, uint32_t item_count,
    const uint32_t *key_items)
{
    uint32_t left_position = 0;
    uint32_t right_position = 0;

    for (;;) {
        while (left_position < item_count &&
               !decode_path_bit(left, key_items[left_position])) {
            left_position++;
        }
        while (right_position < item_count &&
               !decode_path_bit(right, key_items[right_position])) {
            right_position++;
        }
        if (left_position == item_count || right_position == item_count) {
            return left_position == item_count && right_position != item_count;
        }
        uint32_t left_ordinal =
            items[key_items[left_position]].candidate_ordinal;
        uint32_t right_ordinal =
            items[key_items[right_position]].candidate_ordinal;
        int c = decode_cmp_u64(left_ordinal, right_ordinal);
        if (c != 0) return c < 0;
        left_position++;
        right_position++;
    }
}

static bool decode_field_intervals_overlap(const DecodeFieldItem *left,
                                             const DecodeFieldItem *right)
{
    return left->start < right->end && right->start < left->end;
}

static double decode_field_state_score(
    const uint8_t *state, const DecodeFieldItem *items, uint32_t item_count,
    const uint32_t *key_items, const OspreyDecodeInput *input)
{
    double score = 0.0;

    for (uint32_t key_position = 0; key_position < item_count;
         key_position++) {
        uint32_t item_ordinal = key_items[key_position];
        if (state[item_ordinal] == 1) {
            uint32_t candidate_ordinal = items[item_ordinal].candidate_ordinal;
            score += input->field_candidates[candidate_ordinal].posterior;
        }
    }
    return score == 0.0 ? 0.0 : score;
}

/* Return the maximum interval-schedule score under key-prefix constraints.
 * `extra` is one temporarily forced item; all other forced items are marked
 * in state.  Path scores are recomputed in complete P09-key order, matching
 * the package's binary64 score contract. */
static double decode_constrained_field_score(
    const OspreyDecodeInput *input, const DecodeFieldItem *items,
    uint32_t item_count, const uint32_t *predecessors, const uint8_t *state,
    const uint8_t *blocked, uint32_t extra, uint64_t *path_bits,
    uint64_t *scratch_bits, size_t nwords, size_t row_bytes,
    size_t path_word_count, const uint32_t *key_items)
{
    if (extra != UINT32_MAX) {
        if (extra >= item_count || state[extra] != 0 || blocked[extra]) {
            return -INFINITY;
        }
        for (uint32_t i = 0; i < item_count; i++) {
            if (state[i] == 1 && i != extra &&
                decode_field_intervals_overlap(&items[i], &items[extra])) {
                return -INFINITY;
            }
        }
    }
    memset(path_bits, 0, path_word_count * sizeof(*path_bits));
    for (uint32_t i = 0; i < item_count; i++) {
        size_t source_row = predecessors[i] == UINT32_MAX
            ? 0 : (size_t)predecessors[i] + 1u;
        size_t exclude_row = (size_t)i;
        size_t destination_row = (size_t)i + 1u;
        bool extra_conflict = extra != UINT32_MAX && i != extra &&
            decode_field_intervals_overlap(&items[i], &items[extra]);
        bool can_include = state[i] != 2 && !blocked[i] &&
                           !extra_conflict;
        uint64_t *destination = path_bits + destination_row * nwords;
        const uint64_t *exclude = path_bits + exclude_row * nwords;

        memcpy(destination, exclude, row_bytes);
        if (!can_include) continue;
        memcpy(scratch_bits, path_bits + source_row * nwords, row_bytes);
        scratch_bits[i / 64u] |= UINT64_C(1) << (i % 64u);
        double exclude_score = decode_field_path_score(
            exclude, items, item_count, key_items, input);
        double include_score = decode_field_path_score(
            scratch_bits, items, item_count, key_items, input);
        bool choose_include = state[i] == 1 || i == extra ||
            include_score > exclude_score ||
            (include_score == exclude_score &&
             decode_field_path_lex_less(
                 scratch_bits, exclude, items, item_count, key_items));
        if (choose_include) memcpy(destination, scratch_bits, row_bytes);
    }
    return decode_field_path_score(
        path_bits + (size_t)item_count * nwords, items, item_count,
        key_items, input);
}

static bool decode_force_field_item(const DecodeFieldItem *items,
                                    uint32_t item_count, uint8_t *state,
                                    uint8_t *blocked, uint32_t item)
{
    if (item >= item_count || state[item] != 0 || blocked[item]) return false;
    for (uint32_t i = 0; i < item_count; i++) {
        if (state[i] == 1 && i != item &&
            decode_field_intervals_overlap(&items[i], &items[item])) {
            return false;
        }
    }
    state[item] = 1;
    for (uint32_t i = 0; i < item_count; i++) {
        if (i != item && decode_field_intervals_overlap(&items[i], &items[item])) {
            blocked[i] = 1;
        }
    }
    return true;
}

static bool decode_schedule_field_base(
    const OspreyDecodeInput *input, const OspreyDecodeBaseRange *range,
    const uint8_t *field_winner, const uint32_t *field_decision,
    uint8_t *scheduled, OspreyDecodeAllocator *allocator)
{
    DecodeFieldItem *items = NULL;
    uint32_t *predecessors = NULL;
    double *scores = NULL;
    uint64_t *path_bits = NULL;
    uint64_t *scratch_bits = NULL;
    uint8_t *state = NULL;
    uint8_t *blocked = NULL;
    uint32_t *key_items = NULL;
    uint32_t item_count = 0;
    size_t nwords;
    size_t rows;
    size_t padded_count;
    size_t path_word_count;
    size_t row_bytes;
    bool ok = false;

    if (input == NULL || range == NULL || field_winner == NULL ||
        field_decision == NULL || scheduled == NULL || allocator == NULL) {
        return false;
    }
    items = decode_alloc(allocator, range->count, sizeof(*items));
    if (range->count != 0 && items == NULL) goto done;
    for (uint32_t i = 0; i < range->count; i++) {
        uint32_t candidate_ordinal = input->field_by_base[range->begin + i];
        const OspreyDecodeCandidate *candidate;
        int64_t end;

        if (candidate_ordinal >= input->field_count ||
            !field_winner[candidate_ordinal] ||
            field_decision[candidate_ordinal] == UINT32_MAX) {
            continue;
        }
        candidate = &input->field_candidates[candidate_ordinal];
        if (!decode_chunk_end(&candidate->payload.attached.chunk, &end) ||
            item_count == UINT32_MAX) {
            goto done;
        }
        items[item_count].candidate_ordinal = candidate_ordinal;
        items[item_count].start =
            candidate->payload.attached.chunk.address.offset;
        items[item_count].end = end;
        item_count++;
    }
    if (item_count == 0) {
        ok = true;
        goto done;
    }
    if (item_count > 1) {
        qsort(items, item_count, sizeof(*items), decode_field_item_compare);
    }
    key_items = decode_alloc(allocator, item_count, sizeof(*key_items));
    if (key_items == NULL) goto done;
    for (uint32_t i = 0; i < item_count; i++) key_items[i] = i;
    for (uint32_t i = 1; i < item_count; i++) {
        uint32_t item = key_items[i];
        uint32_t j = i;
        while (j != 0 && items[item].candidate_ordinal <
                             items[key_items[j - 1]].candidate_ordinal) {
            key_items[j] = key_items[j - 1];
            j--;
        }
        key_items[j] = item;
    }

    predecessors = decode_alloc(allocator, item_count, sizeof(*predecessors));
    if (predecessors == NULL) goto done;
    for (uint32_t i = 0; i < item_count; i++) {
        uint32_t low = 0;
        uint32_t high = i;
        while (low < high) {
            uint32_t middle = low + (high - low) / 2;
            if (items[middle].end <= items[i].start) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        predecessors[i] = low == 0 ? UINT32_MAX : low - 1;
    }

    if (!decode_size_add((size_t)item_count, 63u, &padded_count) ||
        !decode_size_add((size_t)item_count, 1u, &rows)) goto done;
    nwords = padded_count / 64u;
    if (nwords == 0 ||
        !decode_size_mul(rows, nwords, &path_word_count)) {
        goto done;
    }
    if (!decode_size_mul(nwords, sizeof(*scratch_bits), &row_bytes)) {
        goto done;
    }
    path_bits = decode_alloc(allocator, path_word_count,
                             sizeof(*path_bits));
    scratch_bits = decode_alloc(allocator, nwords, sizeof(*scratch_bits));
    scores = decode_alloc(allocator, rows, sizeof(*scores));
    if (path_bits == NULL || scratch_bits == NULL || scores == NULL) goto done;

    scores[0] = 0.0;
    for (uint32_t i = 0; i < item_count; i++) {
        size_t source_row = predecessors[i] == UINT32_MAX
            ? 0 : (size_t)predecessors[i] + 1u;
        size_t exclude_row = (size_t)i;
        size_t destination_row = (size_t)i + 1u;
        double exclude_score;
        double include_score;
        bool choose_include;

        memcpy(path_bits + destination_row * nwords,
               path_bits + exclude_row * nwords, row_bytes);
        exclude_score = scores[i];
        memcpy(scratch_bits, path_bits + source_row * nwords, row_bytes);
        scratch_bits[i / 64u] |= UINT64_C(1) << (i % 64u);
        include_score = decode_field_path_score(
            scratch_bits, items, item_count, key_items, input);
        choose_include = include_score > exclude_score ||
            (include_score == exclude_score &&
             decode_field_path_lex_less(
                 scratch_bits, path_bits + exclude_row * nwords, items,
                 item_count, key_items));
        if (choose_include) {
            memcpy(path_bits + destination_row * nwords, scratch_bits,
                   row_bytes);
            scores[destination_row] = include_score;
        } else {
            scores[destination_row] = exclude_score;
        }
    }
    /* The interval recurrence determines the maximum score.  Its local
     * lexicographic choices are not sufficient when a zero-score path is a
     * prefix of a later path, so recover the globally smallest canonical
     * selected-key sequence with constrained score queries. */
    state = decode_alloc(allocator, item_count, sizeof(*state));
    blocked = decode_alloc(allocator, item_count, sizeof(*blocked));
    if (state == NULL || blocked == NULL) goto done;

    double target_score = scores[item_count];
    uint32_t cursor = 0;
    while (cursor < item_count && target_score != 0.0) {
        bool selected = false;
        if (decode_field_state_score(state, items, item_count, key_items,
                                     input) == target_score) {
            break;
        }
        for (uint32_t key_position = cursor;
             key_position < item_count; key_position++) {
            uint32_t item_ordinal = key_items[key_position];
            for (uint32_t prior = cursor; prior < key_position; prior++) {
                uint32_t prior_item = key_items[prior];
                if (state[prior_item] == 0) state[prior_item] = 2;
            }
            if (state[item_ordinal] != 0 || blocked[item_ordinal]) continue;
            double candidate_score = decode_constrained_field_score(
                input, items, item_count, predecessors, state, blocked,
                item_ordinal, path_bits, scratch_bits, nwords, row_bytes,
                path_word_count, key_items);
            if (candidate_score == target_score) {
                if (!decode_force_field_item(items, item_count, state, blocked,
                                             item_ordinal)) {
                    goto done;
                }
                cursor = key_position + 1;
                selected = true;
                break;
            }
        }
        if (!selected) goto done;
    }
    for (uint32_t i = 0; i < item_count; i++) {
        if (state != NULL && state[i] == 1) {
            scheduled[items[i].candidate_ordinal] = 1;
        }
    }
    ok = true;

done:
    g_free(items);
    g_free(predecessors);
    g_free(scores);
    g_free(path_bits);
    g_free(scratch_bits);
    g_free(state);
    g_free(blocked);
    g_free(key_items);
    return ok;
}

static void decode_plan_arrays_free(OspreyDecodeArray *arrays,
                                    uint32_t count);

static bool decode_plan_key_used(const OspreyDecodePlan *plan,
                                 const OspreyKey *key)
{
    if (plan == NULL || key == NULL) return false;
    for (uint32_t i = 0; i < plan->decision_count; i++) {
        const OspreyChunkDecision *decision = &plan->decisions[i];
        if ((decision->role_has_predicate &&
             decode_key_compare(&decision->role_key, key) == 0) ||
            (decision->has_pointer_target &&
             decode_key_compare(&decision->pointer_key, key) == 0)) {
            return true;
        }
    }
    return false;
}

static bool decode_plan_add_role_losses(
    const OspreyDecodeInput *input, OspreyDecodePlan *plan,
    OspreyDecodeAllocator *allocator)
{
    bool count_ok;
    uint32_t capacity = decode_input_chunk_count(input, &count_ok);
    uint32_t count = 0;
    const OspreyDecodeCandidate *families[4];
    uint32_t family_counts[4];

    if (!count_ok || input == NULL || plan == NULL || allocator == NULL) {
        return false;
    }
    plan->role_loss_keys = decode_alloc(allocator, capacity,
                                        sizeof(*plan->role_loss_keys));
    if (capacity != 0 && plan->role_loss_keys == NULL) return false;
    families[0] = input->primitive_candidates;
    families[1] = input->scalar_candidates;
    families[2] = input->field_candidates;
    families[3] = input->pointer_candidates;
    family_counts[0] = input->primitive_count;
    family_counts[1] = input->scalar_count;
    family_counts[2] = input->field_count;
    family_counts[3] = input->pointer_count;
    for (size_t family = 0; family < G_N_ELEMENTS(families); family++) {
        for (uint32_t i = 0; i < family_counts[family]; i++) {
            if (decode_plan_key_used(plan, &families[family][i].key)) continue;
            if (count == capacity) return false;
            plan->role_loss_keys[count++] = families[family][i].key;
        }
    }
    /* Families are in predicate-kind order and each Stage 6.1 family is in
     * signed canonical payload order.  Filtering selected keys preserves that
     * order; raw OspreyKey word order would put negative offsets last. */
    for (uint32_t i = 1; i < count; i++) {
        if (decode_key_compare(&plan->role_loss_keys[i - 1],
                               &plan->role_loss_keys[i]) == 0) {
            return false;
        }
    }
    plan->role_loss_count = count;
    return true;
}

void osprey_decode_plan_free(OspreyDecodePlan *plan)
{
    if (plan == NULL) return;
    if (plan->field_groups != NULL) {
        for (uint32_t i = 0; i < plan->field_group_count; i++) {
            g_free(plan->field_groups[i].decision_ordinals);
        }
    }
    decode_plan_arrays_free(plan->arrays, plan->array_count);
    g_free(plan->decisions);
    g_free(plan->chunk_index);
    g_free(plan->field_groups);
    g_free(plan->role_loss_keys);
    g_free(plan);
}

OspreyStatus osprey_decode_roles(const OspreyContext *ctx,
                                 const OspreyDecodeInput *input,
                                 OspreyDecodePlan **out)
{
    OspreyDecodeAllocator allocator;
    OspreyDecodePlan *plan = NULL;
    uint8_t *field_winner = NULL;
    uint8_t *scheduled = NULL;
    uint32_t *field_decision = NULL;

    if (out == NULL) return OSPREY_INVALID_MODEL;
    *out = NULL;
    memset(&allocator, 0, sizeof(allocator));
    if (!decode_input_for_roles_valid(ctx, input, &allocator)) {
        return OSPREY_INVALID_MODEL;
    }

    plan = decode_alloc(&allocator, 1, sizeof(*plan));
    if (plan == NULL) goto failure;
    plan->decision_count = input->chunk_range_count;
    plan->chunk_index_count = input->chunk_range_count;
    plan->decisions = decode_alloc(&allocator, plan->decision_count,
                                   sizeof(*plan->decisions));
    plan->chunk_index = decode_alloc(&allocator, plan->chunk_index_count,
                                     sizeof(*plan->chunk_index));
    if ((plan->decision_count != 0 && plan->decisions == NULL) ||
        (plan->chunk_index_count != 0 && plan->chunk_index == NULL)) {
        goto failure;
    }
    for (uint32_t i = 0; i < plan->decision_count; i++) {
        plan->decisions[i].chunk = input->chunk_ranges[i].chunk;
        plan->chunk_index[i].chunk = input->chunk_ranges[i].chunk;
        plan->chunk_index[i].decision_ordinal = i;
    }

    if (input->field_count != 0) {
        field_winner = decode_alloc(&allocator, input->field_count,
                                    sizeof(*field_winner));
        scheduled = decode_alloc(&allocator, input->field_count,
                                 sizeof(*scheduled));
        field_decision = decode_alloc(&allocator, input->field_count,
                                      sizeof(*field_decision));
        if (field_winner == NULL || scheduled == NULL ||
            field_decision == NULL) goto failure;
        for (uint32_t i = 0; i < input->field_count; i++) {
            field_decision[i] = UINT32_MAX;
        }
    }

    for (uint32_t i = 0; i < input->chunk_range_count; i++) {
        const OspreyDecodeChunkRange *range = &input->chunk_ranges[i];
        const OspreyDecodeCandidate *field = NULL;
        uint32_t field_ordinal = UINT32_MAX;
        for (uint32_t j = 0; j < range->count; j++) {
            const OspreyDecodeCandidateRef *ref =
                &input->chunk_candidates[range->begin + j];
            const OspreyDecodeCandidate *candidate =
                decode_family_candidate(input, ref->family, ref->ordinal);
            if (ref->family != OSPREY_DECODE_FAMILY_FIELD) continue;
            if (candidate == NULL || ref->ordinal >= input->field_count) {
                goto failure;
            }
            field_decision[ref->ordinal] = i;
            if (decode_candidate_better(candidate, field)) {
                field = candidate;
                field_ordinal = ref->ordinal;
            }
        }
        if (field != NULL) {
            if (!decode_field_geometry_valid(input, field)) goto failure;
            field_winner[field_ordinal] = 1;
        }
    }

    for (uint32_t i = 0; i < input->field_base_range_count; i++) {
        if (!decode_schedule_field_base(
                input, &input->field_base_ranges[i], field_winner,
                field_decision, scheduled, &allocator)) {
            goto failure;
        }
    }

    for (uint32_t i = 0; i < input->chunk_range_count; i++) {
        const OspreyDecodeChunkRange *range = &input->chunk_ranges[i];
        OspreyChunkDecision *decision = &plan->decisions[i];
        const OspreyDecodeCandidate *primitive = NULL;
        const OspreyDecodeCandidate *scalar = NULL;
        const OspreyDecodeCandidate *field = NULL;
        const OspreyDecodeCandidate *pointer = NULL;
        for (uint32_t j = 0; j < range->count; j++) {
            const OspreyDecodeCandidateRef *ref =
                &input->chunk_candidates[range->begin + j];
            const OspreyDecodeCandidate *candidate =
                decode_family_candidate(input, ref->family, ref->ordinal);
            if (candidate == NULL) goto failure;
            switch (ref->family) {
            case OSPREY_DECODE_FAMILY_PRIMITIVE:
                if (primitive != NULL) goto failure;
                primitive = candidate;
                break;
            case OSPREY_DECODE_FAMILY_SCALAR:
                if (scalar != NULL) goto failure;
                scalar = candidate;
                break;
            case OSPREY_DECODE_FAMILY_FIELD:
                if (ref->ordinal >= input->field_count ||
                    !scheduled[ref->ordinal]) break;
                if (field != NULL) goto failure;
                field = candidate;
                break;
            case OSPREY_DECODE_FAMILY_POINTER:
                if (decode_candidate_better(candidate, pointer)) {
                    pointer = candidate;
                }
                break;
            default:
                goto failure;
            }
        }

        memset(decision, 0, sizeof(*decision));
        decision->chunk = range->chunk;
        const OspreyDecodeCandidate *role = NULL;
        if (field != NULL) role = field;
        if (scalar != NULL && decode_candidate_better(scalar, role)) {
            role = scalar;
        }
        if (role != NULL) {
            decision->provisional_role = role->predicate_kind ==
                OSPREY_PRED_SCALAR ? OSPREY_STORAGE_SCALAR :
                OSPREY_STORAGE_FIELD;
            decision->role_has_predicate = 1;
            decision->role_posterior = role->posterior;
            decision->role_support = role->direct_support;
            decision->role_source_rule_bits = role->source_rule_bits;
            decision->role_key = role->key;
            if (decision->provisional_role == OSPREY_STORAGE_FIELD) {
                decision->owner_base = role->payload.attached.base;
            }
        } else if (primitive != NULL) {
            decision->provisional_role = OSPREY_STORAGE_PRIMITIVE;
            decision->role_has_predicate = 1;
            decision->role_posterior = primitive->posterior;
            decision->role_support = primitive->direct_support;
            decision->role_source_rule_bits = primitive->source_rule_bits;
            decision->role_key = primitive->key;
        } else {
            decision->provisional_role = OSPREY_STORAGE_PRIMITIVE;
            decision->role_has_predicate = 0;
            decision->role_posterior = 0.0;
        }
        decision->final_role = decision->provisional_role;
        if (pointer != NULL) {
            if (!decode_pointer_target_valid(input, pointer)) goto failure;
            decision->has_pointer_target = 1;
            decision->pointer_target = pointer->payload.attached.base;
            decision->pointer_posterior = pointer->posterior;
            decision->pointer_support = pointer->direct_support;
            decision->pointer_source_rule_bits = pointer->source_rule_bits;
            decision->pointer_key = pointer->key;
        }
    }

    plan->field_groups = decode_alloc(
        &allocator, input->field_base_range_count,
        sizeof(*plan->field_groups));
    if (input->field_base_range_count != 0 && plan->field_groups == NULL) {
        goto failure;
    }
    for (uint32_t i = 0; i < input->field_base_range_count; i++) {
        const OspreyDecodeBaseRange *range = &input->field_base_ranges[i];
        uint32_t survivors = 0;
        for (uint32_t j = 0; j < range->count; j++) {
            uint32_t field_ordinal = input->field_by_base[range->begin + j];
            uint32_t decision_ordinal = field_decision[field_ordinal];
            if (scheduled[field_ordinal] &&
                decision_ordinal < plan->decision_count &&
                plan->decisions[decision_ordinal].provisional_role ==
                    OSPREY_STORAGE_FIELD) {
                survivors++;
            }
        }
        if (survivors == 0) continue;
        OspreyDecodeFieldGroup *group =
            &plan->field_groups[plan->field_group_count++];
        group->base = range->base;
        group->field_count = survivors;
        group->decision_ordinals = decode_alloc(
            &allocator, survivors, sizeof(*group->decision_ordinals));
        if (group->decision_ordinals == NULL) goto failure;
        uint32_t position = 0;
        for (uint32_t j = 0; j < range->count; j++) {
            uint32_t field_ordinal = input->field_by_base[range->begin + j];
            uint32_t decision_ordinal = field_decision[field_ordinal];
            if (scheduled[field_ordinal] &&
                decision_ordinal < plan->decision_count &&
                plan->decisions[decision_ordinal].provisional_role ==
                    OSPREY_STORAGE_FIELD) {
                group->decision_ordinals[position++] = decision_ordinal;
            }
        }
        if (position != survivors) goto failure;
    }

    if (!decode_plan_add_role_losses(input, plan, &allocator)) goto failure;
    g_free(field_winner);
    g_free(scheduled);
    g_free(field_decision);
    *out = plan;
    return OSPREY_OK;

failure:
    g_free(field_winner);
    g_free(scheduled);
    g_free(field_decision);
    osprey_decode_plan_free(plan);
    return OSPREY_INVALID_MODEL;
}

static const OspreyChunk *decode_candidate_chunk(
    const OspreyDecodeCandidate *candidate)
{
    if (candidate == NULL) return NULL;
    return candidate->predicate_kind == OSPREY_PRED_FIELD_OF ||
           candidate->predicate_kind == OSPREY_PRED_POINTER
        ? &candidate->payload.attached.chunk : &candidate->payload.chunk;
}

static bool decode_plan_candidate_retained(
    const OspreyDecodeInput *input, const OspreyChunkDecision *decisions,
    uint32_t decision_count, const OspreyDecodeCandidate *candidate)
{
    const OspreyChunk *chunk = decode_candidate_chunk(candidate);

    if (input == NULL || decisions == NULL || candidate == NULL ||
        chunk == NULL) return false;
    for (uint32_t i = 0; i < decision_count; i++) {
        const OspreyChunkDecision *decision = &decisions[i];
        if (decode_chunk_compare(&decision->chunk, chunk) != 0) continue;
        switch (candidate->predicate_kind) {
        case OSPREY_PRED_PRIMITIVE_VAR:
            return decision->final_role == OSPREY_STORAGE_PRIMITIVE &&
                   decision->role_has_predicate &&
                   decode_key_compare(&decision->role_key,
                                      &candidate->key) == 0;
        case OSPREY_PRED_SCALAR:
            return decision->final_role == OSPREY_STORAGE_SCALAR &&
                   decision->role_has_predicate &&
                   decode_key_compare(&decision->role_key,
                                      &candidate->key) == 0;
        case OSPREY_PRED_FIELD_OF:
            return decision->final_role == OSPREY_STORAGE_FIELD &&
                   decision->role_has_predicate &&
                   decode_key_compare(&decision->role_key,
                                      &candidate->key) == 0;
        case OSPREY_PRED_POINTER:
            return decision->has_pointer_target &&
                   decode_key_compare(&decision->pointer_key,
                                      &candidate->key) == 0;
        default:
            return false;
        }
    }
    return false;
}

static bool decode_plan_for_array_selection_valid(
    const OspreyContext *ctx, const OspreyDecodeInput *input,
    const OspreyDecodePlan *plan, OspreyDecodeAllocator *allocator)
{
    uint8_t *field_seen = NULL;
    uint32_t expected_role_keys;
    bool count_ok;

    if (ctx == NULL || input == NULL || plan == NULL || allocator == NULL ||
        !decode_input_for_roles_valid(ctx, input, allocator) ||
        plan->decision_count != input->chunk_range_count ||
        plan->chunk_index_count != plan->decision_count ||
        (plan->decision_count != 0 &&
         (plan->decisions == NULL || plan->chunk_index == NULL)) ||
        (plan->field_group_count != 0 && plan->field_groups == NULL) ||
        plan->field_group_count > input->field_base_range_count ||
        plan->array_count != 0 || plan->arrays != NULL ||
        plan->discarded_layout != 0) {
        return false;
    }
    expected_role_keys = 0;
    count_ok = decode_u32_add(input->primitive_count, input->scalar_count,
                              &expected_role_keys) &&
               decode_u32_add(expected_role_keys, input->field_count,
                              &expected_role_keys) &&
               decode_u32_add(expected_role_keys, input->pointer_count,
                              &expected_role_keys);
    if (!count_ok || plan->role_loss_count > expected_role_keys ||
        (plan->role_loss_count != 0 && plan->role_loss_keys == NULL)) {
        return false;
    }
    for (uint32_t i = 0; i < plan->decision_count; i++) {
        const OspreyChunkDecision *decision = &plan->decisions[i];
        const OspreyDecodePlanChunkIndex *index = &plan->chunk_index[i];
        if (decode_chunk_compare(&decision->chunk,
                                 &input->chunk_ranges[i].chunk) != 0 ||
            index->decision_ordinal != i ||
            decode_chunk_compare(&index->chunk, &decision->chunk) != 0 ||
            decision->provisional_role < OSPREY_STORAGE_PRIMITIVE ||
            decision->provisional_role > OSPREY_STORAGE_FIELD ||
            decision->final_role != decision->provisional_role ||
            decision->has_pointer_target > 1 ||
            decision->role_has_predicate > 1 ||
            decision->has_array_owner != 0) {
            return false;
        }
        if (decision->provisional_role == OSPREY_STORAGE_FIELD &&
            (!decision->role_has_predicate ||
             !decode_field_geometry_valid(input, &(OspreyDecodeCandidate){
                 .key = decision->role_key,
                 .payload = { .attached = {
                     .chunk = decision->chunk,
                     .base = decision->owner_base }},
                 .predicate_kind = OSPREY_PRED_FIELD_OF,
                 .posterior = decision->role_posterior }))) {
            return false;
        }
        if (decision->has_pointer_target) {
            const OspreyRegionExtent *extent = decode_input_find_extent(
                input, &decision->pointer_target.region);
            if (decision->chunk.size != sizeof(target_ulong) ||
                extent == NULL || decision->pointer_target.offset < extent->lo ||
                decision->pointer_target.offset >= extent->hi) {
                return false;
            }
        }
    }
    field_seen = decode_alloc(allocator, plan->decision_count,
                              sizeof(*field_seen));
    if (plan->decision_count != 0 && field_seen == NULL) return false;
    for (uint32_t i = 0; i < plan->field_group_count; i++) {
        const OspreyDecodeFieldGroup *group = &plan->field_groups[i];
        if (group->field_count == 0 ||
            group->field_count > plan->decision_count ||
            group->decision_ordinals == NULL ||
            (i != 0 && decode_address_compare(
                &plan->field_groups[i - 1].base, &group->base) >= 0)) {
            g_free(field_seen);
            return false;
        }
        for (uint32_t j = 0; j < group->field_count; j++) {
            uint32_t ordinal = group->decision_ordinals[j];
            if (ordinal >= plan->decision_count || field_seen[ordinal] ||
                plan->decisions[ordinal].final_role != OSPREY_STORAGE_FIELD ||
                decode_address_compare(&plan->decisions[ordinal].owner_base,
                                       &group->base) != 0) {
                g_free(field_seen);
                return false;
            }
            field_seen[ordinal] = 1;
        }
    }
    for (uint32_t i = 0; i < plan->decision_count; i++) {
        if (plan->decisions[i].final_role == OSPREY_STORAGE_FIELD &&
            !field_seen[i]) {
            g_free(field_seen);
            return false;
        }
    }
    g_free(field_seen);
    for (uint32_t i = 0; i < plan->role_loss_count; i++) {
        for (uint32_t j = 0; j < i; j++) {
            if (decode_key_compare(&plan->role_loss_keys[i],
                                   &plan->role_loss_keys[j]) == 0) {
                return false;
            }
        }
    }
    return true;
}

static void decode_field_groups_free(OspreyDecodeFieldGroup *groups,
                                     uint32_t count)
{
    if (groups == NULL) return;
    for (uint32_t i = 0; i < count; i++) g_free(groups[i].decision_ordinals);
    g_free(groups);
}

static void decode_plan_arrays_free(OspreyDecodeArray *arrays, uint32_t count)
{
    if (arrays == NULL) return;
    for (uint32_t i = 0; i < count; i++) {
        decode_array_free_contents(&arrays[i]);
    }
    g_free(arrays);
}

static bool decode_rebuild_field_groups(
    const OspreyDecodePlan *plan, const OspreyChunkDecision *decisions,
    OspreyDecodeFieldGroup **groups_out, uint32_t *count_out,
    OspreyDecodeAllocator *allocator)
{
    OspreyDecodeFieldGroup *groups = NULL;
    uint32_t group_count = 0;

    if (plan == NULL || (plan->decision_count != 0 && decisions == NULL) ||
        groups_out == NULL || count_out == NULL || allocator == NULL) {
        return false;
    }
    *groups_out = NULL;
    *count_out = 0;
    groups = decode_alloc(allocator, plan->field_group_count,
                          sizeof(*groups));
    if (plan->field_group_count != 0 && groups == NULL) return false;
    for (uint32_t i = 0; i < plan->field_group_count; i++) {
        const OspreyDecodeFieldGroup *old = &plan->field_groups[i];
        uint32_t survivors = 0;
        for (uint32_t j = 0; j < old->field_count; j++) {
            uint32_t ordinal = old->decision_ordinals[j];
            if (ordinal >= plan->decision_count) {
                decode_field_groups_free(groups, group_count);
                return false;
            }
            if (decisions[ordinal].final_role == OSPREY_STORAGE_FIELD) {
                if (survivors == UINT32_MAX) {
                    decode_field_groups_free(groups, group_count);
                    return false;
                }
                survivors++;
            }
        }
        if (survivors == 0) continue;
        if (group_count == UINT32_MAX) {
            decode_field_groups_free(groups, group_count);
            return false;
        }
        OspreyDecodeFieldGroup *group = &groups[group_count++];
        group->base = old->base;
        group->field_count = survivors;
        group->decision_ordinals = decode_alloc(
            allocator, survivors, sizeof(*group->decision_ordinals));
        if (group->decision_ordinals == NULL) {
            decode_field_groups_free(groups, group_count);
            return false;
        }
        uint32_t position = 0;
        for (uint32_t j = 0; j < old->field_count; j++) {
            uint32_t ordinal = old->decision_ordinals[j];
            if (decisions[ordinal].final_role != OSPREY_STORAGE_FIELD) continue;
            uint32_t at = position++;
            while (at != 0) {
                uint32_t previous = group->decision_ordinals[at - 1];
                int c = decode_chunk_compare(&decisions[previous].chunk,
                                             &decisions[ordinal].chunk);
                if (c < 0 || (c == 0 && previous < ordinal)) break;
                group->decision_ordinals[at] = previous;
                at--;
            }
            group->decision_ordinals[at] = ordinal;
        }
        if (position != survivors) {
            decode_field_groups_free(groups, group_count);
            return false;
        }
    }
    *groups_out = groups;
    *count_out = group_count;
    return true;
}

static bool decode_build_final_role_losses(
    const OspreyDecodeInput *input, const OspreyChunkDecision *decisions,
    uint32_t decision_count, OspreyKey **keys_out, uint32_t *count_out,
    OspreyDecodeAllocator *allocator)
{
    const OspreyDecodeCandidate *families[4];
    uint32_t counts[4];
    bool count_ok;
    uint32_t capacity;
    uint32_t count = 0;

    if (input == NULL || (decision_count != 0 && decisions == NULL) ||
        keys_out == NULL || count_out == NULL || allocator == NULL) {
        return false;
    }
    capacity = decode_input_chunk_count(input, &count_ok);
    if (!count_ok) return false;
    *keys_out = decode_alloc(allocator, capacity, sizeof(**keys_out));
    if (capacity != 0 && *keys_out == NULL) return false;
    families[0] = input->primitive_candidates;
    families[1] = input->scalar_candidates;
    families[2] = input->field_candidates;
    families[3] = input->pointer_candidates;
    counts[0] = input->primitive_count;
    counts[1] = input->scalar_count;
    counts[2] = input->field_count;
    counts[3] = input->pointer_count;
    for (size_t family = 0; family < G_N_ELEMENTS(families); family++) {
        for (uint32_t i = 0; i < counts[family]; i++) {
            if (decode_plan_candidate_retained(input, decisions,
                                               decision_count,
                                               &families[family][i])) {
                continue;
            }
            if (count == capacity) {
                g_free(*keys_out);
                *keys_out = NULL;
                return false;
            }
            (*keys_out)[count++] = families[family][i].key;
        }
    }
    for (uint32_t i = 0; i < count; i++) {
        for (uint32_t j = 0; j < i; j++) {
            if (decode_key_compare(&(*keys_out)[i], &(*keys_out)[j]) == 0) {
                g_free(*keys_out);
                *keys_out = NULL;
                return false;
            }
        }
    }
    *count_out = count;
    return true;
}

static bool decode_copy_selected_arrays(
    const DecodeArrayWork *work, uint32_t work_count, const uint8_t *selected,
    uint32_t selected_count, OspreyDecodeArray **arrays_out,
    OspreyDecodeAllocator *allocator)
{
    OspreyDecodeArray *arrays;
    uint32_t position = 0;

    if (arrays_out == NULL || allocator == NULL ||
        (work_count != 0 && (work == NULL || selected == NULL))) return false;
    *arrays_out = NULL;
    arrays = decode_alloc(allocator, selected_count, sizeof(*arrays));
    if (selected_count != 0 && arrays == NULL) return false;
    for (uint32_t i = 0; i < work_count; i++) {
        if (!selected[i]) continue;
        if (position == selected_count) {
            decode_plan_arrays_free(arrays, position);
            return false;
        }
        arrays[position] = work[i].array;
        arrays[position].member_decision_ordinals = NULL;
        arrays[position].displacement_keys = NULL;
        if (work[i].array.member_count != 0) {
            arrays[position].member_decision_ordinals = decode_alloc(
                allocator, work[i].array.member_count,
                sizeof(*arrays[position].member_decision_ordinals));
            if (arrays[position].member_decision_ordinals == NULL) {
                decode_plan_arrays_free(arrays, position + 1);
                return false;
            }
            memcpy(arrays[position].member_decision_ordinals,
                   work[i].array.member_decision_ordinals,
                   (size_t)work[i].array.member_count *
                   sizeof(*arrays[position].member_decision_ordinals));
        }
        if (work[i].array.displacement_count != 0) {
            arrays[position].displacement_keys = decode_alloc(
                allocator, work[i].array.displacement_count,
                sizeof(*arrays[position].displacement_keys));
            if (arrays[position].displacement_keys == NULL) {
                decode_plan_arrays_free(arrays, position + 1);
                return false;
            }
            memcpy(arrays[position].displacement_keys,
                   work[i].array.displacement_keys,
                   (size_t)work[i].array.displacement_count *
                   sizeof(*arrays[position].displacement_keys));
        }
        position++;
    }
    if (position != selected_count) {
        decode_plan_arrays_free(arrays, position);
        return false;
    }
    *arrays_out = arrays;
    return true;
}

OspreyStatus osprey_decode_select_arrays(const OspreyContext *ctx,
                                         const OspreyDecodeInput *input,
                                         OspreyDecodePlan *plan)
{
    OspreyDecodeAllocator allocator;
    DecodeArrayWork *work = NULL;
    uint32_t work_count = 0;
    uint8_t *selected = NULL;
    uint64_t layout_discarded = 0;
    uint32_t selected_count = 0;
    OspreyChunkDecision *new_decisions = NULL;
    OspreyDecodeFieldGroup *new_groups = NULL;
    OspreyDecodeArray *new_arrays = NULL;
    OspreyKey *new_role_losses = NULL;
    uint32_t new_group_count = 0;
    uint32_t new_role_loss_count = 0;
    uint8_t *owned_members = NULL;

    memset(&allocator, 0, sizeof(allocator));
    if (!decode_plan_for_array_selection_valid(ctx, input, plan,
                                               &allocator)) {
        return OSPREY_INVALID_MODEL;
    }
    if (!decode_array_work_build(ctx, input, plan, &allocator, &work,
                                 &work_count, &layout_discarded)) {
        decode_array_work_free(work, work_count);
        return OSPREY_INVALID_MODEL;
    }
    selected = decode_alloc(&allocator, work_count, sizeof(*selected));
    if (work_count != 0 && selected == NULL) goto failure;
    if (!decode_array_work_schedule(work, work_count, selected, &allocator)) {
        goto failure;
    }
    for (uint32_t i = 0; i < work_count; i++) {
        if (!selected[i]) continue;
        if (selected_count == UINT32_MAX) goto failure;
        selected_count++;
    }
    if (work_count > UINT64_MAX - layout_discarded) goto failure;
    layout_discarded += work_count;
    if (selected_count > layout_discarded) goto failure;
    layout_discarded -= selected_count;

    if (selected_count == 0) {
        plan->discarded_layout = layout_discarded;
        decode_array_work_free(work, work_count);
        g_free(selected);
        return OSPREY_OK;
    }

    new_decisions = decode_alloc(&allocator, plan->decision_count,
                                 sizeof(*new_decisions));
    if (plan->decision_count != 0 && new_decisions == NULL) goto failure;
    if (plan->decision_count != 0) {
        memcpy(new_decisions, plan->decisions,
               (size_t)plan->decision_count * sizeof(*new_decisions));
    }
    owned_members = decode_alloc(&allocator, plan->decision_count,
                                 sizeof(*owned_members));
    if (plan->decision_count != 0 && owned_members == NULL) goto failure;
    for (uint32_t i = 0; i < work_count; i++) {
        if (!selected[i]) continue;
        const OspreyDecodeArray *array = &work[i].array;
        for (uint32_t j = 0; j < array->member_count; j++) {
            uint32_t ordinal = array->member_decision_ordinals[j];
            if (ordinal >= plan->decision_count || owned_members[ordinal]) {
                goto failure;
            }
            owned_members[ordinal] = 1;
            new_decisions[ordinal].final_role =
                OSPREY_STORAGE_ARRAY_ELEMENT;
            new_decisions[ordinal].has_array_owner = 1;
            new_decisions[ordinal].array_owner = (OspreyAddress){
                .region = array->region, .offset = array->lo };
            new_decisions[ordinal].array_posterior = array->posterior;
            new_decisions[ordinal].array_support = array->direct_support;
            new_decisions[ordinal].array_source_rule_bits =
                array->source_rule_bits;
            new_decisions[ordinal].array_key = array->key;
        }
    }
    if (!decode_copy_selected_arrays(work, work_count, selected,
                                     selected_count, &new_arrays,
                                     &allocator) ||
        !decode_rebuild_field_groups(plan, new_decisions, &new_groups,
                                     &new_group_count, &allocator) ||
        !decode_build_final_role_losses(input, new_decisions,
                                        plan->decision_count,
                                        &new_role_losses,
                                        &new_role_loss_count, &allocator)) {
        goto failure;
    }

    OspreyDecodeArray *old_arrays = plan->arrays;
    OspreyChunkDecision *old_decisions = plan->decisions;
    OspreyDecodeFieldGroup *old_groups = plan->field_groups;
    OspreyKey *old_role_losses = plan->role_loss_keys;
    uint32_t old_array_count = plan->array_count;
    uint32_t old_group_count = plan->field_group_count;
    plan->arrays = new_arrays;
    plan->array_count = selected_count;
    plan->discarded_layout = layout_discarded;
    plan->decisions = new_decisions;
    plan->field_groups = new_groups;
    plan->field_group_count = new_group_count;
    plan->role_loss_keys = new_role_losses;
    plan->role_loss_count = new_role_loss_count;
    new_arrays = NULL;
    new_decisions = NULL;
    new_groups = NULL;
    new_role_losses = NULL;
    decode_plan_arrays_free(old_arrays, old_array_count);
    g_free(old_decisions);
    decode_field_groups_free(old_groups, old_group_count);
    g_free(old_role_losses);
    decode_array_work_free(work, work_count);
    g_free(selected);
    g_free(owned_members);
    return OSPREY_OK;

failure:
    decode_plan_arrays_free(new_arrays, selected_count);
    g_free(new_decisions);
    decode_field_groups_free(new_groups, new_group_count);
    g_free(new_role_losses);
    decode_array_work_free(work, work_count);
    g_free(selected);
    g_free(owned_members);
    return OSPREY_INVALID_MODEL;
}

static const char *decode_storage_role_name(uint8_t role)
{
    switch (role) {
    case OSPREY_STORAGE_PRIMITIVE: return "primitive";
    case OSPREY_STORAGE_SCALAR: return "scalar";
    case OSPREY_STORAGE_FIELD: return "field";
    case OSPREY_STORAGE_ARRAY_ELEMENT: return "array-element";
    default: return "invalid";
    }
}

static bool decode_dump_region_id(FILE *out, const char *prefix,
                                  const OspreyRegionId *region)
{
    return fprintf(out, "[%s-region %u] [%s-image 0x%016" PRIx64 "]"
                   " [%s-site 0x%016" PRIx64 "]", prefix,
                   (unsigned)region->kind, prefix, region->code_image_id,
                   prefix, region->site_offset) >= 0;
}

static bool decode_dump_address(FILE *out, const char *prefix,
                                const OspreyAddress *address)
{
    return decode_dump_region_id(out, prefix, &address->region) &&
           fprintf(out, " [%s-offset %" PRId64 "]", prefix,
                   address->offset) >= 0;
}

static bool decode_dump_chunk(FILE *out, const char *prefix,
                              const OspreyChunk *chunk)
{
    return decode_dump_address(out, prefix, &chunk->address) &&
           fprintf(out, " [%s-size %" PRIu64 "]", prefix, chunk->size) >= 0;
}

static bool decode_dump_decision(FILE *out, uint32_t ordinal,
                                 const OspreyChunkDecision *decision)
{
    uint64_t role_bits;
    uint64_t pointer_bits;
    uint64_t array_bits;

    if (out == NULL || decision == NULL) return false;
    memcpy(&role_bits, &decision->role_posterior, sizeof(role_bits));
    memcpy(&pointer_bits, &decision->pointer_posterior,
           sizeof(pointer_bits));
    memcpy(&array_bits, &decision->array_posterior, sizeof(array_bits));
    if (fprintf(out, "[decision %u] ", ordinal) < 0 ||
        !decode_dump_chunk(out, "chunk", &decision->chunk) ||
        fprintf(out, " [provisional %s] [final %s]"
                     " [role-source %u] [role-posterior-bits 0x%016" PRIx64 "]"
                     " [role-support %" PRIu64 "]"
                     " [role-rules 0x%016" PRIx64 "]",
                decode_storage_role_name(decision->provisional_role),
                decode_storage_role_name(decision->final_role),
                (unsigned)decision->role_has_predicate, role_bits,
                decision->role_support, decision->role_source_rule_bits) < 0) {
        return false;
    }
    if (decision->role_has_predicate) {
        if (fprintf(out, " [role-key ") < 0 ||
            !decode_dump_key(out, &decision->role_key) ||
            fputc(']', out) == EOF) return false;
    } else if (fprintf(out, " [role-key implicit]") < 0) {
        return false;
    }
    if (decision->provisional_role == OSPREY_STORAGE_FIELD &&
        !decode_dump_address(out, "owner", &decision->owner_base)) {
        return false;
    }
    if (decision->has_array_owner) {
        if (decision->final_role != OSPREY_STORAGE_ARRAY_ELEMENT ||
            !decode_dump_address(out, "array-owner", &decision->array_owner) ||
            fprintf(out, " [array-posterior-bits 0x%016" PRIx64 "]"
                         " [array-support %" PRIu64 "]"
                         " [array-rules 0x%016" PRIx64 "] [array-key ",
                    array_bits, decision->array_support,
                    decision->array_source_rule_bits) < 0 ||
            !decode_dump_key(out, &decision->array_key) ||
            fputc(']', out) == EOF) {
            return false;
        }
    }
    if (fprintf(out, " [pointer %u] [pointer-posterior-bits 0x%016" PRIx64 "]"
                     " [pointer-support %" PRIu64 "]"
                     " [pointer-rules 0x%016" PRIx64 "]",
                (unsigned)decision->has_pointer_target, pointer_bits,
                decision->pointer_support,
                decision->pointer_source_rule_bits) < 0) {
        return false;
    }
    if (decision->has_pointer_target &&
        (!decode_dump_address(out, "target", &decision->pointer_target) ||
         fprintf(out, " [pointer-key ") < 0 ||
         !decode_dump_key(out, &decision->pointer_key) ||
         fputc(']', out) == EOF)) {
        return false;
    }
    return fputc('\n', out) != EOF;
}

static bool decode_dump_array(FILE *out, uint32_t ordinal,
                              const OspreyDecodeArray *array,
                              const OspreyDecodePlan *plan)
{
    uint64_t score_bits;

    if (out == NULL || array == NULL || plan == NULL ||
        (array->member_count != 0 && array->member_decision_ordinals == NULL) ||
        (array->displacement_count != 0 && array->displacement_keys == NULL) ||
        array->adjusted_score.negative_infinite > 1 ||
        !isfinite(array->adjusted_score.finite)) {
        return false;
    }
    memcpy(&score_bits, &array->adjusted_score.finite, sizeof(score_bits));
    if (fprintf(out, "[array %u] ", ordinal) < 0 ||
        !decode_dump_region_id(out, "region", &array->region) ||
        fprintf(out, " [key ") < 0 || !decode_dump_key(out, &array->key) ||
        fputc(']', out) == EOF ||
        fprintf(out, " [lo %" PRId64 "] [hi %" PRId64 "]"
                     " [stride %" PRIu64 "] [count %" PRIu64 "]"
                     " [posterior-bits 0x%016" PRIx64 "]"
                     " [support %" PRIu64 "]"
                     " [source-rules 0x%016" PRIx64 "]"
                     " [score-infinity %" PRId64 "]"
                     " [score-negative-infinite %u]"
                     " [score-finite-bits 0x%016" PRIx64 "] [members",
                array->lo, array->hi, array->stride, array->count,
                array->posterior_bits, array->direct_support,
                array->source_rule_bits, array->adjusted_score.infinity_balance,
                (unsigned)array->adjusted_score.negative_infinite,
                score_bits) < 0) {
        return false;
    }
    for (uint32_t i = 0; i < array->member_count; i++) {
        uint32_t decision_ordinal = array->member_decision_ordinals[i];
        OspreyKey chunk_key;
        if (decision_ordinal >= plan->decision_count ||
            (i != 0 && fputc(',', out) == EOF)) return false;
        chunk_key = osprey_chunk_key(&plan->decisions[decision_ordinal].chunk);
        if (!decode_dump_key(out, &chunk_key)) return false;
    }
    if (fprintf(out, "] [displacement") < 0) return false;
    for (uint32_t i = 0; i < array->displacement_count; i++) {
        if (fprintf(out, "%s", i == 0 ? " " : ",") < 0 ||
            !decode_dump_key(out, &array->displacement_keys[i])) {
            return false;
        }
    }
    return fprintf(out, "]\n") >= 0;
}

bool osprey_decode_plan_dump_file(const OspreyDecodePlan *plan, FILE *out)
{
    if (plan == NULL || out == NULL ||
        (plan->decision_count != 0 && plan->decisions == NULL) ||
        (plan->field_group_count != 0 && plan->field_groups == NULL) ||
        (plan->array_count != 0 && plan->arrays == NULL) ||
        (plan->role_loss_count != 0 && plan->role_loss_keys == NULL)) {
        return false;
    }
    if (fprintf(out, "[decisions %u] [field-groups %u] [arrays %u]"
                     " [discarded-layout %" PRIu64 "] [role-loss %u]\n",
                plan->decision_count, plan->field_group_count,
                plan->array_count, plan->discarded_layout,
                plan->role_loss_count) < 0) return false;
    for (uint32_t i = 0; i < plan->decision_count; i++) {
        if (!decode_dump_decision(out, i, &plan->decisions[i])) return false;
    }
    for (uint32_t i = 0; i < plan->field_group_count; i++) {
        const OspreyDecodeFieldGroup *group = &plan->field_groups[i];
        if (group->field_count == 0 || group->decision_ordinals == NULL ||
            (fprintf(out, "[field-group] ") < 0) ||
            !decode_dump_address(out, "base", &group->base) ||
            fprintf(out, " [members") < 0) return false;
        for (uint32_t j = 0; j < group->field_count; j++) {
            uint32_t ordinal = group->decision_ordinals[j];
            if (ordinal >= plan->decision_count ||
                (j != 0 && fputc(',', out) == EOF) ||
                fprintf(out, "%u", ordinal) < 0) return false;
        }
        if (fprintf(out, "]\n") < 0) return false;
    }
    for (uint32_t i = 0; i < plan->array_count; i++) {
        if (!decode_dump_array(out, i, &plan->arrays[i], plan)) return false;
    }
    if (fprintf(out, "[role-loss-keys") < 0) return false;
    for (uint32_t i = 0; i < plan->role_loss_count; i++) {
        if (fprintf(out, "%s", i == 0 ? " " : ",") < 0 ||
            !decode_dump_key(out, &plan->role_loss_keys[i])) return false;
    }
    return fprintf(out, "]\n") >= 0 && ferror(out) == 0;
}

/* ------------------------------------------------------------------ */
/* Model helpers                                                       */
/* ------------------------------------------------------------------ */

static void bucket_free(gpointer p) {
    g_array_free((GArray *)p, TRUE);
}

static OspreyModel *model_new(void) {
    OspreyModel *m = g_new0(OspreyModel, 1);
    m->objects = g_array_new(FALSE, FALSE, sizeof(OspreyDecodedObject));
    m->by_chunk = g_hash_table_new_full(osprey_key_hash, osprey_key_equal,
                                        osprey_key_free, NULL);
    m->type_names = g_array_new(FALSE, FALSE, sizeof(char *));
    m->raw_spans = g_array_new(FALSE, FALSE, sizeof(OspRawSpan));
    m->fields_by_base = g_hash_table_new_full(osprey_key_hash,
                                              osprey_key_equal,
                                              osprey_key_free, bucket_free);
    m->ptr_by_chunk = g_hash_table_new_full(osprey_key_hash,
                                            osprey_key_equal,
                                            osprey_key_free, NULL);
    return m;
}

static uint32_t model_add_type_name(OspreyModel *m, const char *name) {
    for (guint i = 0; i < m->type_names->len; i++) {
        if (strcmp(g_array_index(m->type_names, char *, i), name) == 0) {
            return i;
        }
    }
    char *copy = g_strdup(name);
    g_array_append_val(m->type_names, copy);
    return m->type_names->len - 1;
}

/* Insert or merge; returns the object index. */
static uint32_t model_upsert(OspreyModel *m, const OspreyDecodedObject *o) {
    OspreyKey k = osprey_chunk_key(&o->chunk);
    gpointer existing = g_hash_table_lookup(m->by_chunk, &k);
    if (existing != NULL) {
        uint32_t idx = (uint32_t)(uintptr_t)existing - 1;
        OspreyDecodedObject *cur = &g_array_index(m->objects,
                                                  OspreyDecodedObject, idx);
        /* keep the higher-posterior interpretation */
        if (o->posterior > cur->posterior) {
            *cur = *o;
        }
        return idx;
    }
    g_array_append_val(m->objects, *o);
    uint32_t idx = m->objects->len - 1;
    g_hash_table_insert(m->by_chunk, osprey_key_new(&k),
                        GSIZE_TO_POINTER((gsize)idx + 1));
    return idx;
}

/* ------------------------------------------------------------------ */
/* Region name parts for deterministic naming                          */
/* ------------------------------------------------------------------ */

static const char *region_tag(const OspreyRegionId *r) {
    switch (r->kind) {
    case OSPREY_REGION_HEAP_SITE: return "H";
    case OSPREY_REGION_STACK_FUNCTION: return "S";
    default: return "G";
    }
}

static void region_name(const OspreyRegionId *r, char *buf, size_t n) {
    snprintf(buf, n, "%s_%llx", region_tag(r),
             (unsigned long long)r->site_offset);
}

/* ------------------------------------------------------------------ */
/* Decode: build the OspreyModel from posterior predicates            */
/* ------------------------------------------------------------------ */

static OspreyStatus decode_graph(OspreyContext *ctx) {
    OspreyGraph *g = ctx->graph;
    OspreyModel *m = ctx->staged_model;
    if (m == NULL) return OSPREY_INVALID_MODEL;
    double thresh = ctx->config.report_threshold;
    /* Pass 1: primitives (with their chunk key) and pointers. */
    GHashTable *prim_by_chunk = g_hash_table_new_full(osprey_key_hash,
                                                     osprey_key_equal,
                                                     osprey_key_free, NULL);
    for (guint i = 0; i < g->vars->len; i++) {
        OspreyVar *v = &g_array_index(g->vars, OspreyVar, i);
        if (v->hard_false) continue;
        if (v->belief < thresh) continue;
        switch (v->kind) {
        case OSPREY_PRED_PRIMITIVE_VAR:
            {
                OspreyKey ck = osprey_chunk_key(&v->payload.chunk);
                g_hash_table_insert(prim_by_chunk, osprey_key_new(&ck),
                                    GSIZE_TO_POINTER(v->id + 1));
            }
            break;
        case OSPREY_PRED_POINTER:
            /* pointer: at most one target base per chunk (§10.5),
             * selected by max posterior; independent of the
             * scalar/field exclusivity on the same chunk (§10.2). */
            {
                OspreyKey ck = osprey_chunk_key(&v->payload.attached.chunk);
                gpointer cur = g_hash_table_lookup(m->ptr_by_chunk,
                                                   &ck);
                OspreyDecodedObject o;
                memset(&o, 0, sizeof(o));
                o.chunk = v->payload.attached.chunk;
                o.kind = OSPREY_DECODED_POINTER;
                o.posterior = v->belief;
                o.parent_region = v->payload.attached.base.region;
                o.parent_offset = v->payload.attached.base.offset;
                if (cur != NULL) {
                    uint32_t cidx = (uint32_t)(uintptr_t)cur - 1;
                    OspreyDecodedObject *cur_o = &g_array_index(
                        m->objects, OspreyDecodedObject, cidx);
                    if (o.posterior <= cur_o->posterior) break;
                    *cur_o = o;
                } else {
                    uint32_t idx = m->objects->len;
                    g_array_append_val(m->objects, o);
                    g_hash_table_insert(m->ptr_by_chunk,
                                        osprey_key_new(&ck),
                                        GSIZE_TO_POINTER((gsize)idx + 1));
                }
            }
            break;
        case OSPREY_PRED_SCALAR:
            {
                OspreyDecodedObject o;
                memset(&o, 0, sizeof(o));
                o.chunk = v->payload.chunk;
                o.kind = OSPREY_DECODED_SCALAR;
                o.posterior = v->belief;
                model_upsert(m, &o);
            }
            break;
        case OSPREY_PRED_FIELD_OF:
            {
                OspreyDecodedObject o;
                memset(&o, 0, sizeof(o));
                o.chunk = v->payload.attached.chunk;
                o.kind = OSPREY_DECODED_FIELD;
                o.posterior = v->belief;
                o.parent_region = v->payload.attached.base.region;
                o.parent_offset = v->payload.attached.base.offset;
                uint32_t idx = model_upsert(m, &o);
                /* index fields by base for the consumer */
                OspreyKey bk = osprey_base_key(&o.parent_region,
                                               o.parent_offset);
                GArray *list = g_hash_table_lookup(m->fields_by_base,
                                                   &bk);
                if (list == NULL) {
                    list = g_array_new(FALSE, FALSE, sizeof(uint32_t));
                    g_hash_table_insert(m->fields_by_base,
                                        osprey_key_new(&bk), list);
                }
                g_array_append_val(list, idx);
            }
            break;
        case OSPREY_PRED_ARRAY_START:
            {
                OspreyDecodedObject o;
                memset(&o, 0, sizeof(o));
                o.chunk.address = v->payload.addr;
                o.chunk.size = 0; /* start marker */
                o.kind = OSPREY_DECODED_ARRAY_START;
                o.posterior = v->belief;
                model_upsert(m, &o);
            }
            break;
        default:
            break;
        }
    }
    g_hash_table_destroy(prim_by_chunk);

    /* Pass 2: arrays by weighted interval scheduling per (region,
     * stride).  Interval weight = logit(P(Array)); avoid spans covered
     * by scalar/field primitives. */
    {
        GArray *arrs = g_array_new(FALSE, FALSE, sizeof(uint32_t));
        for (guint i = 0; i < g->vars->len; i++) {
            OspreyVar *v = &g_array_index(g->vars, OspreyVar, i);
            if (v->kind != OSPREY_PRED_ARRAY) continue;
            if (v->hard_false || v->belief < thresh) continue;
            uint32_t id = v->id;
            g_array_append_val(arrs, id);
        }
        /* group by (region, stride) and run interval scheduling per
         * group; bounded by OSPREY_DECODE_MAX_ARRAYS_PER_SIDE. */
        if (arrs->len > OSPREY_DECODE_MAX_ARRAYS_PER_SIDE) {
            g_array_free(arrs, TRUE);
            return OSPREY_LIMIT_EXCEEDED;
        }
        {
            for (guint i = 0; i < arrs->len; i++) {
                uint32_t id = g_array_index(arrs, uint32_t, i);
                OspreyVar *v = &g_array_index(g->vars, OspreyVar, id);
                OspreyDecodedObject o;
                memset(&o, 0, sizeof(o));
                o.chunk.address.region = v->payload.segment.a1.region;
                o.chunk.address.offset = v->payload.segment.a1.offset;
                o.chunk.size = (uint64_t)v->payload.segment.size;
                o.kind = OSPREY_DECODED_ARRAY;
                o.posterior = v->belief;
                o.parent_region = v->payload.segment.a2.region;
                o.parent_offset = v->payload.segment.a2.offset;
                model_upsert(m, &o);
            }
        }
        g_array_free(arrs, TRUE);
    }

    /* Pass 3: struct bases — field groups per base with non-overlap. */
    {
        GHashTable *bases = g_hash_table_new_full(osprey_key_hash,
                                                  osprey_key_equal,
                                                  osprey_key_free,
                                                  bucket_free);
        for (guint i = 0; i < g->vars->len; i++) {
            OspreyVar *v = &g_array_index(g->vars, OspreyVar, i);
            if (v->kind != OSPREY_PRED_FIELD_OF) continue;
            if (v->hard_false || v->belief < thresh) continue;
            OspreyKey bk = osprey_base_key(&v->payload.attached.base.region,
                                           v->payload.attached.base.offset);
            GArray *fields = g_hash_table_lookup(bases, &bk);
            if (fields == NULL) {
                fields = g_array_new(FALSE, FALSE, sizeof(uint32_t));
                g_hash_table_insert(bases, osprey_key_new(&bk), fields);
            }
            uint32_t id = v->id;
            g_array_append_val(fields, id);
        }
        GHashTableIter bit;
        gpointer rk, arr_ptr;
        g_hash_table_iter_init(&bit, bases);
        while (g_hash_table_iter_next(&bit, &rk, &arr_ptr)) {
            GArray *fields = (GArray *)arr_ptr;
            if (fields->len > OSPREY_DECODE_MAX_FIELDS_PER_BASE) {
                g_hash_table_destroy(bases);
                return OSPREY_LIMIT_EXCEEDED;
            }
            /* sort by offset */
            uint32_t *ids = (uint32_t *)fields->data;
            for (guint i = 1; i < fields->len; i++) {
                uint32_t id = ids[i];
                guint j = i;
                while (j > 0 &&
                       g_array_index(g->vars, OspreyVar, ids[j - 1])
                           .payload.attached.chunk.address.offset >
                       g_array_index(g->vars, OspreyVar, id)
                           .payload.attached.chunk.address.offset) {
                    ids[j] = ids[j - 1];
                    j--;
                }
                ids[j] = id;
            }
            /* greedy non-overlapping selection by start offset; fields
             * already entered the model in pass 1, so this only decides
             * which fields stay (drop the overlapping ones). */
            int64_t last_end = INT64_MIN;
            for (guint i = 0; i < fields->len; i++) {
                OspreyVar *v = &g_array_index(g->vars, OspreyVar, ids[i]);
                int64_t off = v->payload.attached.chunk.address.offset;
                int64_t end = off + (int64_t)v->payload.attached.chunk.size;
                if (off < last_end) {
                    /* overlap: drop the lower-posterior field */
                    OspreyKey ck = osprey_chunk_key(
                        &v->payload.attached.chunk);
                    gpointer cur = g_hash_table_lookup(m->by_chunk,
                                                       &ck);
                    if (cur != NULL) {
                        uint32_t idx = (uint32_t)(uintptr_t)cur - 1;
                        OspreyDecodedObject *cur_o = &g_array_index(
                            m->objects, OspreyDecodedObject, idx);
                        cur_o->posterior = 0.0; /* discarded */
                    }
                    continue;
                }
                last_end = end;
            }
        }
        /* emit one STRUCT base object per surviving base */
        g_hash_table_iter_init(&bit, bases);
        while (g_hash_table_iter_next(&bit, &rk, &arr_ptr)) {
            GArray *fields = (GArray *)arr_ptr;
            if (fields->len == 0) continue;
            OspreyVar *f0 = &g_array_index(g->vars, OspreyVar,
                                           g_array_index(fields, uint32_t, 0));
            OspreyDecodedObject o;
            memset(&o, 0, sizeof(o));
            o.chunk.address.region = f0->payload.attached.base.region;
            o.chunk.address.offset = f0->payload.attached.base.offset;
            o.chunk.size = 0;
            o.kind = OSPREY_DECODED_STRUCT;
            o.posterior = f0->belief;
            o.parent_region = f0->payload.attached.base.region;
            o.parent_offset = f0->payload.attached.base.offset;
            model_upsert(m, &o);
        }
        g_hash_table_destroy(bases);
    }

    /* Pass 4: raw spans from merged region instances. */
    for (guint j = 0; j < m->objects->len; j++) {
        OspreyDecodedObject *o = &g_array_index(m->objects,
                                                OspreyDecodedObject, j);
        if (o->kind == OSPREY_DECODED_SCALAR ||
            o->kind == OSPREY_DECODED_POINTER ||
            o->kind == OSPREY_DECODED_FIELD) {
            continue;
        }
        if (o->kind == OSPREY_DECODED_ARRAY_START &&
            o->chunk.size != 0) {
            continue;
        }
        for (guint i = 0; i < ctx->region_instances->len; i++) {
            const OspreyRegionInstance *ri = &g_array_index(
                ctx->region_instances, OspreyRegionInstance, i);
            if (ri->region.kind != o->chunk.address.region.kind) continue;
            if (ri->region.code_image_id !=
                o->chunk.address.region.code_image_id)
                continue;
            if (ri->region.site_offset !=
                o->chunk.address.region.site_offset)
                continue;
            OspRawSpan sp;
            memset(&sp, 0, sizeof(sp));
            if (ri->region.kind == OSPREY_REGION_STACK_FUNCTION) {
                /* Downward stack: the observed window is
                 * [raw_min, raw_max] == [min_sp, entry_sp].  The object
                 * span runs from the object's raw start up to the frame
                 * top; never treat the stack as
                 * [entry_sp, entry_sp+extent). */
                sp.raw_start = ri->raw_base +
                               (uint64_t)o->chunk.address.offset;
                sp.raw_end = ri->raw_max;
            } else {
                sp.raw_start = ri->raw_base +
                               (uint64_t)o->chunk.address.offset;
                /* raw_max is already the exclusive end of the runtime
                 * region.  Adding the full span to an interior object
                 * start would extend the lookup beyond the allocation or
                 * global window. */
                sp.raw_end = ri->raw_max;
            }
            sp.obj_idx = j;
            sp.is_chunk = 0;
            g_array_append_val(m->raw_spans, sp);
            break;
        }
    }
    /* chunk-exact spans for scalars/fields/pointers */
    for (guint j = 0; j < m->objects->len; j++) {
        OspreyDecodedObject *o = &g_array_index(m->objects,
                                                OspreyDecodedObject, j);
        if (o->kind != OSPREY_DECODED_SCALAR &&
            o->kind != OSPREY_DECODED_POINTER &&
            o->kind != OSPREY_DECODED_FIELD) {
            continue;
        }
        if (o->chunk.size == 0) continue;
        for (guint i = 0; i < ctx->region_instances->len; i++) {
            const OspreyRegionInstance *ri = &g_array_index(
                ctx->region_instances, OspreyRegionInstance, i);
            if (ri->region.kind != o->chunk.address.region.kind) continue;
            if (ri->region.code_image_id !=
                o->chunk.address.region.code_image_id)
                continue;
            if (ri->region.site_offset !=
                o->chunk.address.region.site_offset)
                continue;
            OspRawSpan sp;
            memset(&sp, 0, sizeof(sp));
            sp.raw_start = ri->raw_base +
                           (uint64_t)o->chunk.address.offset;
            sp.raw_end = sp.raw_start + o->chunk.size;
            sp.obj_idx = j;
            sp.is_chunk = 1;
            g_array_append_val(m->raw_spans, sp);
            break;
        }
    }
    /* sort by raw_start */
    for (guint i = 1; i < m->raw_spans->len; i++) {
        OspRawSpan v = g_array_index(m->raw_spans, OspRawSpan, i);
        guint j2 = i;
        while (j2 > 0 &&
               g_array_index(m->raw_spans, OspRawSpan, j2 - 1).raw_start >
                   v.raw_start) {
            g_array_index(m->raw_spans, OspRawSpan, j2) =
                g_array_index(m->raw_spans, OspRawSpan, j2 - 1);
            j2--;
        }
        g_array_index(m->raw_spans, OspRawSpan, j2) = v;
    }

    /* Pass 5: type names + type_id ordinals. */
    for (guint i = 0; i < m->objects->len; i++) {
        OspreyDecodedObject *o = &g_array_index(m->objects,
                                                OspreyDecodedObject, i);
        char name[128];
        switch (o->kind) {
        case OSPREY_DECODED_STRUCT:
        case OSPREY_DECODED_FIELD:
            {
                char rn[64];
                region_name(&o->parent_region, rn, sizeof(rn));
                snprintf(name, sizeof(name), "struct_%s_off%llx", rn,
                         (unsigned long long)o->parent_offset);
            }
            break;
        case OSPREY_DECODED_ARRAY:
            {
                char rn[64];
                region_name(&o->chunk.address.region, rn, sizeof(rn));
                snprintf(name, sizeof(name), "array_%s_%llx", rn,
                         (unsigned long long)o->chunk.address.offset);
            }
            break;
        case OSPREY_DECODED_POINTER:
            {
                char rn[64];
                region_name(&o->parent_region, rn, sizeof(rn));
                snprintf(name, sizeof(name), "ptr_%s_%llx", rn,
                         (unsigned long long)o->parent_offset);
            }
            break;
        case OSPREY_DECODED_SCALAR:
        default:
            snprintf(name, sizeof(name), "prim_%llx",
                     (unsigned long long)o->chunk.size);
            break;
        }
        o->type_id = model_add_type_name(m, name);
    }

    return OSPREY_OK;
}

/* Free a decoded model and everything it owns (Stage 0/1 ownership). */
void osprey_model_free(OspreyModel *m) {
    if (m == NULL) return;
    if (m->by_chunk != NULL) g_hash_table_destroy(m->by_chunk);
    if (m->fields_by_base != NULL) g_hash_table_destroy(m->fields_by_base);
    if (m->ptr_by_chunk != NULL) g_hash_table_destroy(m->ptr_by_chunk);
    if (m->objects != NULL) g_array_free(m->objects, TRUE);
    if (m->raw_spans != NULL) g_array_free(m->raw_spans, TRUE);
    if (m->type_names != NULL) {
        for (guint i = 0; i < m->type_names->len; i++) {
            g_free(g_array_index(m->type_names, char *, i));
        }
        g_array_free(m->type_names, TRUE);
    }
    g_free(m);
}

OspreyStatus osprey_decode(OspreyContext *ctx) {
    if (ctx == NULL || !ctx->config.enabled) return OSPREY_DISABLED;
    if (ctx->graph == NULL) return OSPREY_INCOMPLETE_FACTS;
    /* Stage 0: build the new model off to the side.  The committed
     * model is untouched until the whole transaction is OSPREY_OK and
     * osprey_tx_install() swaps it in. */
    OspreyModel *m = model_new();
    OspreyModel *prev_staged = ctx->staged_model;
    ctx->staged_model = m;
    OspreyStatus st = decode_graph(ctx);
    if (st != OSPREY_OK) {
        ctx->staged_model = prev_staged;
        osprey_model_free(m);
        return st;
    }
    log_msg("[osprey] [decode] [objects %u] [types %u] "
            "[raw-spans %u]\n",
            m->objects->len, m->type_names->len, m->raw_spans->len);
    return OSPREY_OK;
}

/* ------------------------------------------------------------------ */
/* Consumer lookups (parent side)                                      */
/* ------------------------------------------------------------------ */

const OspreyDecodedObject *osprey_lookup_chunk(const OspreyModel *model,
                                                const OspreyChunk *chunk) {
    if (model == NULL || chunk == NULL) return NULL;
    OspreyKey k = osprey_chunk_key(chunk);
    const OspreyDecodedObject *ptr_o = NULL;
    gpointer pcur = g_hash_table_lookup(model->ptr_by_chunk,
                                        &k);
    if (pcur != NULL) {
        uint32_t pidx = (uint32_t)(uintptr_t)pcur - 1;
        ptr_o = &g_array_index(model->objects, OspreyDecodedObject, pidx);
    }
    gpointer cur = g_hash_table_lookup(model->by_chunk, &k);
    const OspreyDecodedObject *o = NULL;
    if (cur != NULL) {
        uint32_t idx = (uint32_t)(uintptr_t)cur - 1;
        o = &g_array_index(model->objects, OspreyDecodedObject, idx);
        if (o->posterior <= 0.0) o = NULL; /* discarded (overlap) */
    }
    /* the pointer interpretation wins when it beats the scalar/field */
    if (ptr_o != NULL && (o == NULL || ptr_o->posterior > o->posterior)) {
        return ptr_o;
    }
    return o;
}

/* Raw address -> decoded object: most specific span wins (exact chunk
 * beats region-span; smallest covering span first). */
const OspreyDecodedObject *osprey_lookup_raw(const OspreyModel *model,
                                              uint64_t raw) {
    if (model == NULL || model->raw_spans->len == 0) return NULL;
    const OspRawSpan *best = NULL;
    uint64_t best_span = UINT64_MAX;
    double best_plus = -1.0;
    uint8_t best_ptr = 0;
    for (guint i = 0; i < model->raw_spans->len; i++) {
        const OspRawSpan *sp = &g_array_index(model->raw_spans,
                                              OspRawSpan, i);
        if (raw < sp->raw_start) continue;
        if (sp->raw_end > 0 && raw >= sp->raw_end) continue;
        uint64_t span = sp->raw_end - sp->raw_start;
        const OspreyDecodedObject *cand = &g_array_index(
            model->objects, OspreyDecodedObject, sp->obj_idx);
        uint8_t is_ptr = cand->kind == OSPREY_DECODED_POINTER;
        if (span < best_span ||
            (span == best_span && (cand->posterior > best_plus ||
             (cand->posterior == best_plus && is_ptr && !best_ptr)))) {
            best_span = span;
            best_plus = cand->posterior;
            best_ptr = is_ptr;
            best = sp;
        }
    }
    if (best == NULL) return NULL;
    const OspreyDecodedObject *o = &g_array_index(
        model->objects, OspreyDecodedObject, best->obj_idx);
    if (o->posterior <= 0.0) return NULL;
    return o;
}

bool osprey_raw_extent(const OspreyModel *model,
                       const OspreyDecodedObject *obj, uint64_t *raw_out,
                       uint64_t *extent_out) {
    if (model == NULL || obj == NULL || raw_out == NULL ||
        extent_out == NULL) {
        return false;
    }
    for (guint i = 0; i < model->raw_spans->len; i++) {
        const OspRawSpan *sp = &g_array_index(model->raw_spans,
                                              OspRawSpan, i);
        if (sp->obj_idx >= model->objects->len) continue;
        const OspreyDecodedObject *o = &g_array_index(
            model->objects, OspreyDecodedObject, sp->obj_idx);
        if (o != obj) continue;
        *raw_out = sp->raw_start;
        *extent_out = sp->raw_end - sp->raw_start;
        return true;
    }
    return false;
}
