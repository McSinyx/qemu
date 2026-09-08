#include "stage6_decode_reference.h"

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* This implementation intentionally does not call any decoder production
 * helper.  The only production semantic operation used for ordering is the
 * accepted signed payload comparator. */

typedef struct RefCandidate {
    OspreyReferenceCandidate value;
} RefCandidate;

typedef struct RefDecision {
    OspreyChunk chunk;
    int32_t primitive;
    int32_t scalar;
    int32_t field_winner;
    int32_t pointer;
    int32_t role;
    uint8_t role_kind;
    uint8_t final_kind;
    uint8_t has_array;
    uint8_t reserved;
    OspreyAddress owner;
    OspreyAddress array_owner;
    double array_posterior;
    uint64_t array_support;
    uint64_t array_rules;
} RefDecision;

typedef struct RefGroup {
    OspreyAddress base;
    uint32_t *decisions;
    uint32_t count;
} RefGroup;

typedef struct RefScore {
    int64_t infinity_balance;
    double finite;
    bool negative_infinite;
} RefScore;

typedef struct RefArray {
    uint32_t candidate;
    OspreyRegionId region;
    int64_t lo;
    int64_t hi;
    uint64_t stride;
    uint64_t count;
    RefScore score;
    uint32_t *members;
    uint32_t member_count;
    bool selected;
} RefArray;

typedef struct RefTypeRef {
    uint8_t kind;
    uint8_t target_is_void;
    uint8_t target_aggregate_kind;
    uint8_t reserved;
    uint64_t primitive_size;
    OspreyAddress target;
} RefTypeRef;

typedef struct RefFieldWork {
    OspreyDecodedField field;
    RefTypeRef value_ref;
} RefFieldWork;

typedef struct RefTypeWork {
    uint8_t kind;
    uint8_t target_is_void;
    uint8_t target_aggregate_kind;
    uint8_t reserved;
    uint64_t size;
    uint64_t element_count;
    uint64_t element_size;
    OspreyAddress canonical_base;
    RefTypeRef element_ref;
    RefTypeRef target_ref;
    uint32_t element_work_id;
    uint32_t field_begin;
    uint32_t field_count;
    double posterior;
    uint64_t direct_support;
    uint64_t source_rule_bits;
    uint32_t old_index;
} RefTypeWork;

typedef struct RefListState {
    double score;
    uint32_t *items;
    uint32_t count;
} RefListState;

typedef struct RefArrayState {
    RefScore score;
    uint32_t *items;
    uint32_t count;
} RefArrayState;

static int ref_u64(uint64_t a, uint64_t b)
{
    return a < b ? -1 : (a != b);
}

static int ref_i64(int64_t a, int64_t b)
{
    return a < b ? -1 : (a != b);
}

static bool ref_add_i64(int64_t a, int64_t b, int64_t *out)
{
    if (out == NULL) return false;
    if ((b > 0 && a > INT64_MAX - b) ||
        (b < 0 && a < INT64_MIN - b)) return false;
    *out = a + b;
    return true;
}

static bool ref_sub_i64(int64_t a, int64_t b, int64_t *out)
{
    if (out == NULL) return false;
    if ((b > 0 && a < INT64_MIN + b) ||
        (b < 0 && a > INT64_MAX + b)) return false;
    *out = a - b;
    return true;
}

static bool ref_region_valid(const OspreyRegionId *region)
{
    return region != NULL && region->kind >= OSPREY_REGION_GLOBAL &&
           region->kind <= OSPREY_REGION_STACK_FUNCTION;
}

static int ref_region_compare(const OspreyRegionId *a,
                              const OspreyRegionId *b)
{
    int c;
    if (a == NULL || b == NULL) return a == b ? 0 : (a == NULL ? -1 : 1);
    c = ref_u64((uint64_t)a->kind, (uint64_t)b->kind);
    if (c != 0) return c;
    c = ref_u64(a->code_image_id, b->code_image_id);
    if (c != 0) return c;
    return ref_u64(a->site_offset, b->site_offset);
}

static int ref_address_compare(const OspreyAddress *a,
                               const OspreyAddress *b)
{
    int c = ref_region_compare(&a->region, &b->region);
    return c != 0 ? c : ref_i64(a->offset, b->offset);
}

static int ref_chunk_compare(const OspreyChunk *a, const OspreyChunk *b)
{
    int c = ref_address_compare(&a->address, &b->address);
    return c != 0 ? c : ref_u64(a->size, b->size);
}

static bool ref_region_equal(const OspreyRegionId *a,
                             const OspreyRegionId *b)
{
    return ref_region_compare(a, b) == 0;
}

static bool ref_address_equal(const OspreyAddress *a,
                              const OspreyAddress *b)
{
    return ref_address_compare(a, b) == 0;
}

static bool ref_chunk_equal(const OspreyChunk *a, const OspreyChunk *b)
{
    return ref_chunk_compare(a, b) == 0;
}

static bool ref_chunk_end(const OspreyChunk *chunk, int64_t *end)
{
    if (chunk == NULL || end == NULL || chunk->size == 0 ||
        chunk->size > (uint64_t)INT64_MAX) return false;
    return ref_add_i64(chunk->address.offset, (int64_t)chunk->size, end);
}

static const OspreyRegionExtent *ref_extent(
    const OspreyRegionExtent *extents, uint32_t count,
    const OspreyRegionId *region)
{
    if (extents == NULL || region == NULL) return NULL;
    for (uint32_t i = 0; i < count; i++) {
        if (ref_region_equal(&extents[i].region, region)) return &extents[i];
    }
    return NULL;
}

static bool ref_payload_valid(uint8_t kind, const OspreyVarPayload *payload)
{
    int64_t end;
    int64_t relative;
    if (payload == NULL) return false;
    switch (kind) {
    case OSPREY_PRED_PRIMITIVE_VAR:
    case OSPREY_PRED_SCALAR:
        return ref_region_valid(&payload->chunk.address.region) &&
               ref_chunk_end(&payload->chunk, &end);
    case OSPREY_PRED_PRIMITIVE_ACCESS:
        return ref_region_valid(&payload->prim_access.chunk.address.region) &&
               ref_chunk_end(&payload->prim_access.chunk, &end);
    case OSPREY_PRED_UNFOLDABLE_HEAP:
    case OSPREY_PRED_FOLDABLE_HEAP:
        return ref_region_valid(&payload->heap_fold.region) &&
               payload->heap_fold.region.kind == OSPREY_REGION_HEAP_SITE &&
               payload->heap_fold.size <= (uint64_t)INT64_MAX;
    case OSPREY_PRED_HOMO_SEGMENT: {
        int64_t other_end;
        if (!ref_region_valid(&payload->segment.a1.region) ||
            !ref_region_valid(&payload->segment.a2.region) ||
            payload->segment.size <= 0 ||
            ref_address_compare(&payload->segment.a1,
                                &payload->segment.a2) > 0) return false;
        return ref_add_i64(payload->segment.a1.offset,
                           payload->segment.size, &end) &&
               ref_add_i64(payload->segment.a2.offset,
                           payload->segment.size, &other_end);
    }
    case OSPREY_PRED_ARRAY: {
        int64_t span;
        if (!ref_region_valid(&payload->segment.a1.region) ||
            !ref_region_equal(&payload->segment.a1.region,
                              &payload->segment.a2.region) ||
            payload->segment.a1.offset >= payload->segment.a2.offset ||
            payload->segment.size <= 0 ||
            payload->segment.size > INT64_MAX ||
            !ref_sub_i64(payload->segment.a2.offset,
                         payload->segment.a1.offset, &span)) return false;
        return span >= payload->segment.size;
    }
    case OSPREY_PRED_ARRAY_START:
        return ref_region_valid(&payload->addr.region);
    case OSPREY_PRED_FIELD_OF:
        if (!ref_region_valid(&payload->attached.chunk.address.region) ||
            !ref_region_valid(&payload->attached.base.region) ||
            !ref_region_equal(&payload->attached.chunk.address.region,
                              &payload->attached.base.region) ||
            !ref_chunk_end(&payload->attached.chunk, &end) ||
            !ref_add_i64(payload->attached.base.offset,
                         (int64_t)payload->attached.chunk.size, &end) ||
            !ref_sub_i64(payload->attached.chunk.address.offset,
                         payload->attached.base.offset, &relative)) {
            return false;
        }
        return relative >= 0;
    case OSPREY_PRED_POINTER:
        return ref_region_valid(&payload->attached.chunk.address.region) &&
               ref_region_valid(&payload->attached.base.region) &&
               ref_chunk_end(&payload->attached.chunk, &end);
    default:
        return false;
    }
}

static const OspreyChunk *ref_candidate_chunk(const RefCandidate *candidate)
{
    if (candidate == NULL) return NULL;
    if (candidate->value.kind == OSPREY_PRED_FIELD_OF ||
        candidate->value.kind == OSPREY_PRED_POINTER) {
        return &candidate->value.payload.attached.chunk;
    }
    return &candidate->value.payload.chunk;
}

static int ref_candidate_compare_values(const OspreyReferenceCandidate *a,
                                        const OspreyReferenceCandidate *b)
{
    int c = ref_u64(a->kind, b->kind);
    if (c != 0) return c;
    return osprey_var_payload_compare(a->kind, &a->payload, &b->payload);
}

static int ref_candidate_compare(const void *ap, const void *bp)
{
    const RefCandidate *a = ap;
    const RefCandidate *b = bp;
    return ref_candidate_compare_values(&a->value, &b->value);
}

static int ref_group_compare(const void *ap, const void *bp)
{
    const RefGroup *a = ap;
    const RefGroup *b = bp;
    return ref_address_compare(&a->base, &b->base);
}

static bool ref_is_projected(uint8_t kind)
{
    return kind == OSPREY_PRED_PRIMITIVE_VAR ||
           kind == OSPREY_PRED_SCALAR || kind == OSPREY_PRED_ARRAY ||
           kind == OSPREY_PRED_FIELD_OF || kind == OSPREY_PRED_POINTER;
}

static bool ref_belief_valid(const OspreyReferenceCandidate *candidate)
{
    return candidate != NULL && candidate->belief_valid == 1 &&
           isfinite(candidate->posterior) && candidate->posterior >= 0.0 &&
           candidate->posterior <= 1.0;
}

static bool ref_candidate_better(const RefCandidate *candidate,
                                 const RefCandidate *current)
{
    if (candidate == NULL) return false;
    if (current == NULL) return true;
    return candidate->value.posterior > current->value.posterior ||
           (candidate->value.posterior == current->value.posterior &&
            ref_candidate_compare(candidate, current) < 0);
}

static bool ref_field_geometry_valid(const OspreyReferenceCase *input,
                                     const RefCandidate *candidate)
{
    const OspreyChunk *chunk = &candidate->value.payload.attached.chunk;
    const OspreyAddress *base = &candidate->value.payload.attached.base;
    const OspreyRegionExtent *extent;
    int64_t end;
    int64_t relative;
    int64_t field_end;

    if (candidate->value.kind != OSPREY_PRED_FIELD_OF ||
        !ref_region_equal(&chunk->address.region, &base->region) ||
        !ref_chunk_end(chunk, &end) ||
        !ref_sub_i64(chunk->address.offset, base->offset, &relative) ||
        relative < 0 || !ref_add_i64(relative, (int64_t)chunk->size,
                                      &field_end)) return false;
    extent = ref_extent(input->extents, input->extent_count, &base->region);
    return extent != NULL && base->offset >= extent->lo &&
           base->offset < extent->hi && chunk->address.offset >= extent->lo &&
           end <= extent->hi;
}

static bool ref_pointer_geometry_valid(const OspreyReferenceCase *input,
                                       const RefCandidate *candidate)
{
    const OspreyChunk *chunk = &candidate->value.payload.attached.chunk;
    const OspreyAddress *target = &candidate->value.payload.attached.base;
    const OspreyRegionExtent *extent;
    if (candidate->value.kind != OSPREY_PRED_POINTER ||
        chunk->size != sizeof(target_ulong)) return false;
    extent = ref_extent(input->extents, input->extent_count,
                        &target->region);
    return extent != NULL && target->offset >= extent->lo &&
           target->offset < extent->hi;
}

static void ref_score_zero(RefScore *score)
{
    memset(score, 0, sizeof(*score));
}

static bool ref_score_add(RefScore *score, const RefScore *term)
{
    int64_t balance;
    double finite;
    if (score == NULL || term == NULL || term->negative_infinite) {
        if (score != NULL && term != NULL && term->negative_infinite) {
            score->negative_infinite = true;
            score->finite = 0.0;
            score->infinity_balance = 0;
            return true;
        }
        return false;
    }
    if (score->negative_infinite) return true;
    if ((term->infinity_balance > 0 &&
         score->infinity_balance > INT64_MAX - term->infinity_balance) ||
        (term->infinity_balance < 0 &&
         score->infinity_balance < INT64_MIN - term->infinity_balance)) {
        return false;
    }
    balance = score->infinity_balance + term->infinity_balance;
    finite = score->finite + term->finite;
    if (!isfinite(finite)) return false;
    score->infinity_balance = balance;
    score->finite = finite == 0.0 ? 0.0 : finite;
    return true;
}

static bool ref_score_probability(RefScore *score, double probability,
                                  bool subtract)
{
    RefScore term;
    ref_score_zero(&term);
    if (!isfinite(probability) || probability < 0.0 || probability > 1.0) {
        return false;
    }
    if (probability == 0.0) {
        if (subtract) return true;
        term.negative_infinite = true;
    } else if (probability == 1.0) {
        term.infinity_balance = subtract ? -1 : 1;
    } else {
        term.finite = log(probability) - log1p(-probability);
        if (!isfinite(term.finite)) return false;
        if (subtract) term.finite = -term.finite;
    }
    return ref_score_add(score, &term);
}

static int ref_score_compare(const RefScore *a, const RefScore *b)
{
    if (a->negative_infinite != b->negative_infinite) {
        return a->negative_infinite ? -1 : 1;
    }
    if (a->negative_infinite) return 0;
    if (a->infinity_balance != b->infinity_balance) {
        return a->infinity_balance < b->infinity_balance ? -1 : 1;
    }
    return a->finite < b->finite ? -1 : (a->finite != b->finite);
}

static void ref_free_groups(RefGroup *groups, uint32_t count)
{
    if (groups == NULL) return;
    for (uint32_t i = 0; i < count; i++) free(groups[i].decisions);
    free(groups);
}

static bool ref_append_u32(uint32_t **values, uint32_t *count,
                           uint32_t value)
{
    uint32_t *grown;
    if (values == NULL || count == NULL || *count == UINT32_MAX) return false;
    grown = realloc(*values, ((size_t)*count + 1u) * sizeof(**values));
    if (grown == NULL) return false;
    grown[*count] = value;
    *values = grown;
    (*count)++;
    return true;
}

static int ref_decision_compare(const void *ap, const void *bp)
{
    const RefDecision *a = ap;
    const RefDecision *b = bp;
    return ref_chunk_compare(&a->chunk, &b->chunk);
}

static int ref_find_decision(const RefDecision *decisions, uint32_t count,
                             const OspreyChunk *chunk)
{
    for (uint32_t i = 0; i < count; i++) {
        if (ref_chunk_equal(&decisions[i].chunk, chunk)) return (int)i;
    }
    return -1;
}

static bool ref_list_copy(uint32_t **out, const uint32_t *items,
                          uint32_t count)
{
    *out = NULL;
    if (count == 0) return true;
    *out = malloc((size_t)count * sizeof(**out));
    if (*out == NULL) return false;
    memcpy(*out, items, (size_t)count * sizeof(**out));
    return true;
}

static int ref_candidate_index_compare(uint32_t a, uint32_t b,
                                       const RefCandidate *candidates)
{
    return ref_candidate_compare(&candidates[a], &candidates[b]);
}

static void ref_sort_candidate_indexes(uint32_t *items, uint32_t count,
                                       const RefCandidate *candidates)
{
    for (uint32_t i = 1; i < count; i++) {
        uint32_t value = items[i];
        uint32_t j = i;
        while (j != 0 && ref_candidate_index_compare(
                   value, items[j - 1], candidates) < 0) {
            items[j] = items[j - 1];
            j--;
        }
        items[j] = value;
    }
}

static bool ref_list_lex_less(const uint32_t *left, uint32_t left_count,
                              const uint32_t *right, uint32_t right_count,
                              const RefCandidate *candidates)
{
    uint32_t n = left_count < right_count ? left_count : right_count;
    for (uint32_t i = 0; i < n; i++) {
        int c = ref_candidate_index_compare(left[i], right[i], candidates);
        if (c != 0) return c < 0;
    }
    return left_count < right_count;
}

static double ref_field_list_score(const uint32_t *items, uint32_t count,
                                   const RefCandidate *candidates)
{
    double score = 0.0;
    uint32_t *ordered = NULL;
    if (count != 0) {
        ordered = malloc((size_t)count * sizeof(*ordered));
        if (ordered == NULL) return NAN;
        memcpy(ordered, items, (size_t)count * sizeof(*ordered));
        ref_sort_candidate_indexes(ordered, count, candidates);
    }
    for (uint32_t i = 0; i < count; i++) {
        score += candidates[ordered[i]].value.posterior;
    }
    free(ordered);
    return score == 0.0 ? 0.0 : score;
}

static bool ref_state_better(const RefListState *left,
                             const RefListState *right,
                             const RefCandidate *candidates)
{
    if (left->score != right->score) return left->score > right->score;
    return ref_list_lex_less(left->items, left->count, right->items,
                             right->count, candidates);
}

static bool ref_schedule_group(const RefCandidate *candidates,
                               const uint32_t *field_candidates,
                               uint32_t count, uint8_t *scheduled,
                               uint32_t *selected_count)
{
    RefListState *states = NULL;
    uint32_t *predecessors = NULL;
    uint32_t *ordered = NULL;
    uint32_t *working = NULL;
    bool ok = false;

    if (selected_count != NULL) *selected_count = 0;
    if (count == 0) return true;
    states = calloc((size_t)count + 1u, sizeof(*states));
    predecessors = malloc((size_t)count * sizeof(*predecessors));
    ordered = malloc((size_t)count * sizeof(*ordered));
    if (states == NULL || predecessors == NULL || ordered == NULL) goto done;
    memcpy(ordered, field_candidates, (size_t)count * sizeof(*ordered));
    /* The production recurrence orders fields by end, start, then canonical
     * input ordinal.  The candidates already use canonical P09 order. */
    for (uint32_t i = 1; i < count; i++) {
        uint32_t value = ordered[i];
        int64_t value_end;
        int64_t value_start = candidates[value].value.payload.attached.chunk
            .address.offset;
        if (!ref_chunk_end(&candidates[value].value.payload.attached.chunk,
                           &value_end)) goto done;
        uint32_t j = i;
        while (j != 0) {
            uint32_t previous = ordered[j - 1];
            int64_t previous_end;
            if (!ref_chunk_end(&candidates[previous].value.payload.attached.chunk,
                               &previous_end)) goto done;
            int64_t previous_start = candidates[previous].value.payload
                .attached.chunk.address.offset;
            if (previous_end < value_end ||
                (previous_end == value_end &&
                 (previous_start < value_start ||
                  (previous_start == value_start &&
                   ref_candidate_index_compare(previous, value, candidates) <= 0)))) {
                break;
            }
            ordered[j] = previous;
            j--;
        }
        ordered[j] = value;
    }
    for (uint32_t i = 0; i < count; i++) {
        int64_t start = candidates[ordered[i]].value.payload.attached.chunk
            .address.offset;
        uint32_t low = 0;
        uint32_t high = i;
        while (low < high) {
            uint32_t middle = low + (high - low) / 2u;
            int64_t end;
            if (!ref_chunk_end(&candidates[ordered[middle]].value.payload.attached.chunk,
                               &end)) goto done;
            if (end <= start) low = middle + 1u;
            else high = middle;
        }
        predecessors[i] = low == 0 ? UINT32_MAX : low - 1u;
    }
    for (uint32_t i = 0; i < count; i++) {
        RefListState include = { 0 };
        RefListState exclude = states[i];
        uint32_t item = ordered[i];
        uint32_t base_count = predecessors[i] == UINT32_MAX ? 0 :
            states[predecessors[i] + 1u].count;
        include.count = base_count + 1u;
        include.items = malloc((size_t)include.count * sizeof(*include.items));
        if (include.items == NULL) goto done;
        if (base_count != 0) memcpy(include.items,
                                     states[predecessors[i] + 1u].items,
                                     (size_t)base_count * sizeof(*include.items));
        include.items[base_count] = item;
        include.score = ref_field_list_score(include.items, include.count,
                                             candidates);
        if (!isfinite(include.score)) {
            free(include.items);
            goto done;
        }
        if (ref_state_better(&include, &exclude, candidates)) {
            free(states[i + 1u].items);
            states[i + 1u] = include;
        } else {
            free(include.items);
            if (!ref_list_copy(&states[i + 1u].items, exclude.items,
                               exclude.count)) goto done;
            states[i + 1u].count = exclude.count;
            states[i + 1u].score = exclude.score;
        }
    }
    working = states[count].items;
    states[count].items = NULL;
    ref_sort_candidate_indexes(working, states[count].count, candidates);
    for (uint32_t i = 0; i < states[count].count; i++) {
        scheduled[working[i]] = 1;
    }
    if (selected_count != NULL) *selected_count = states[count].count;
    ok = true;

done:
    if (states != NULL) {
        for (uint32_t i = 0; i <= count; i++) free(states[i].items);
    }
    free(states);
    free(predecessors);
    free(ordered);
    free(working);
    return ok;
}

static bool ref_array_schedule_score(const RefArray *arrays,
                                     const uint32_t *items, uint32_t count,
                                     const RefCandidate *candidates,
                                     RefScore *score)
{
    uint32_t *ordered = NULL;
    bool ok = false;
    ref_score_zero(score);
    if (count != 0) {
        ordered = malloc((size_t)count * sizeof(*ordered));
        if (ordered == NULL) return false;
        memcpy(ordered, items, (size_t)count * sizeof(*ordered));
        for (uint32_t i = 1; i < count; i++) {
            uint32_t v = ordered[i];
            uint32_t j = i;
            while (j != 0 && ref_candidate_index_compare(
                       arrays[v].candidate, arrays[ordered[j - 1]].candidate,
                       candidates) < 0) {
                ordered[j] = ordered[j - 1];
                j--;
            }
            ordered[j] = v;
        }
    }
    for (uint32_t i = 0; i < count; i++) {
        if (!ref_score_add(score, &arrays[ordered[i]].score)) goto done;
    }
    ok = true;
done:
    free(ordered);
    return ok;
}

static int ref_array_interval_compare(uint32_t left, uint32_t right,
                                      const RefArray *arrays,
                                      const RefCandidate *candidates)
{
    int c = ref_i64(arrays[left].hi, arrays[right].hi);
    if (c != 0) return c;
    c = ref_i64(arrays[left].lo, arrays[right].lo);
    if (c != 0) return c;
    return ref_candidate_index_compare(arrays[left].candidate,
                                       arrays[right].candidate, candidates);
}

static bool ref_array_state_copy(RefArrayState *out,
                                 const RefArrayState *source)
{
    if (out == NULL || source == NULL) return false;
    memset(out, 0, sizeof(*out));
    out->score = source->score;
    out->count = source->count;
    return ref_list_copy(&out->items, source->items, source->count);
}

static bool ref_array_state_include(RefArrayState *out,
                                    const RefArrayState *base,
                                    uint32_t array_index,
                                    const RefArray *arrays,
                                    const RefCandidate *candidates)
{
    if (out == NULL || base == NULL || arrays == NULL || candidates == NULL ||
        base->count == UINT32_MAX) return false;
    memset(out, 0, sizeof(*out));
    out->count = base->count + 1u;
    out->items = malloc((size_t)out->count * sizeof(*out->items));
    if (out->items == NULL) return false;
    if (base->count != 0) {
        memcpy(out->items, base->items,
               (size_t)base->count * sizeof(*out->items));
    }
    out->items[base->count] = array_index;
    if (!ref_array_schedule_score(arrays, out->items, out->count,
                                  candidates, &out->score)) {
        free(out->items);
        memset(out, 0, sizeof(*out));
        return false;
    }
    return true;
}

static bool ref_array_intersects(const RefArray *left,
                                 const RefArray *right)
{
    return left->lo < right->hi && right->lo < left->hi;
}

/* Maximum score with canonical-key prefix decisions fixed in `state`.
 * Values are 0=undecided, 1=selected, 2=excluded; `extra` temporarily forces
 * one undecided item.  Scores are always recomputed in P08-key order. */
static bool ref_array_constrained_best(
    const RefArray *arrays, uint32_t start, uint32_t count,
    const RefCandidate *candidates, const uint8_t *state, uint32_t extra,
    RefScore *out)
{
    uint32_t *ordered = NULL;
    uint32_t *predecessors = NULL;
    uint32_t *fixed = NULL;
    uint32_t fixed_count = 0;
    RefArrayState *states = NULL;
    bool ok = false;

    if (arrays == NULL || candidates == NULL || state == NULL || out == NULL ||
        (extra != UINT32_MAX && extra >= count)) return false;
    ordered = malloc((size_t)count * sizeof(*ordered));
    predecessors = malloc((size_t)count * sizeof(*predecessors));
    fixed = malloc((size_t)count * sizeof(*fixed));
    states = calloc((size_t)count + 1u, sizeof(*states));
    if (ordered == NULL || predecessors == NULL || fixed == NULL ||
        states == NULL) goto done;
    for (uint32_t i = 0; i < count; i++) {
        ordered[i] = start + i;
        if (state[i] == 1 || i == extra) fixed[fixed_count++] = start + i;
    }
    for (uint32_t i = 0; i < fixed_count; i++) {
        for (uint32_t j = 0; j < i; j++) {
            if (ref_array_intersects(&arrays[fixed[i]], &arrays[fixed[j]])) {
                goto done;
            }
        }
    }
    if (!ref_list_copy(&states[0].items, fixed, fixed_count)) goto done;
    states[0].count = fixed_count;
    if (!ref_array_schedule_score(arrays, states[0].items, states[0].count,
                                  candidates, &states[0].score)) goto done;
    for (uint32_t i = 1; i < count; i++) {
        uint32_t value = ordered[i];
        uint32_t j = i;
        while (j != 0 && ref_array_interval_compare(
                   value, ordered[j - 1], arrays, candidates) < 0) {
            ordered[j] = ordered[j - 1];
            j--;
        }
        ordered[j] = value;
    }
    for (uint32_t i = 0; i < count; i++) {
        uint32_t low = 0;
        uint32_t high = i;
        while (low < high) {
            uint32_t middle = low + (high - low) / 2u;
            if (arrays[ordered[middle]].hi <= arrays[ordered[i]].lo) {
                low = middle + 1u;
            } else {
                high = middle;
            }
        }
        predecessors[i] = low == 0 ? UINT32_MAX : low - 1u;
    }
    for (uint32_t i = 0; i < count; i++) {
        uint32_t local = ordered[i] - start;
        bool forced = state[local] == 1 || local == extra;
        bool blocked = state[local] == 2;
        for (uint32_t j = 0; !forced && !blocked && j < fixed_count; j++) {
            if (ref_array_intersects(&arrays[ordered[i]], &arrays[fixed[j]])) {
                blocked = true;
            }
        }
        if (forced || blocked) {
            if (!ref_array_state_copy(&states[i + 1u], &states[i])) goto done;
            continue;
        }
        const RefArrayState *base = predecessors[i] == UINT32_MAX ?
            &states[0] : &states[predecessors[i] + 1u];
        RefArrayState include;
        if (!ref_array_state_include(&include, base, ordered[i], arrays,
                                     candidates)) goto done;
        if (ref_score_compare(&include.score, &states[i].score) > 0) {
            states[i + 1u] = include;
        } else {
            free(include.items);
            if (!ref_array_state_copy(&states[i + 1u], &states[i])) goto done;
        }
    }
    *out = states[count].score;
    ok = true;

done:
    if (states != NULL) {
        for (uint32_t i = 0; i <= count; i++) free(states[i].items);
    }
    free(states);
    free(fixed);
    free(predecessors);
    free(ordered);
    return ok;
}

/* Independent weighted-interval recurrence plus global canonical tie
 * reconstruction.  Local DP tie choices are not lexicographically
 * compositional when later zero-score arrays can be added. */
static bool ref_schedule_array_region(RefArray *arrays, uint32_t start,
                                      uint32_t count,
                                      const RefCandidate *candidates)
{
    uint8_t *state = NULL;
    uint32_t *key_order = NULL;
    RefScore target;
    RefScore empty;
    bool ok = false;

    if (count == 0) return true;
    state = calloc(count, sizeof(*state));
    key_order = malloc((size_t)count * sizeof(*key_order));
    if (state == NULL || key_order == NULL) goto done;
    for (uint32_t i = 0; i < count; i++) key_order[i] = i;
    for (uint32_t i = 1; i < count; i++) {
        uint32_t value = key_order[i];
        uint32_t j = i;
        while (j != 0 && ref_candidate_index_compare(
                   arrays[start + value].candidate,
                   arrays[start + key_order[j - 1]].candidate,
                   candidates) < 0) {
            key_order[j] = key_order[j - 1];
            j--;
        }
        key_order[j] = value;
    }
    ref_score_zero(&empty);
    if (!ref_array_constrained_best(arrays, start, count, candidates, state,
                                    UINT32_MAX, &target)) goto done;
    if (ref_score_compare(&target, &empty) <= 0) {
        ok = true;
        goto done;
    }
    for (uint32_t key = 0; key < count; key++) {
        RefScore selected_score;
        uint32_t item = key_order[key];
        if (!ref_array_schedule_score(arrays, NULL, 0, candidates,
                                      &selected_score)) goto done;
        uint32_t selected_items[64];
        uint32_t selected_count = 0;
        if (count > G_N_ELEMENTS(selected_items)) goto done;
        for (uint32_t i = 0; i < count; i++) {
            if (state[i] == 1) selected_items[selected_count++] = start + i;
        }
        if (!ref_array_schedule_score(arrays, selected_items, selected_count,
                                      candidates, &selected_score)) goto done;
        if (ref_score_compare(&selected_score, &target) == 0) break;
        if (state[item] != 0) continue;
        RefScore candidate_score;
        if (!ref_array_constrained_best(arrays, start, count, candidates,
                                        state, item, &candidate_score)) {
            state[item] = 2;
            continue;
        }
        if (ref_score_compare(&candidate_score, &target) == 0) {
            state[item] = 1;
            for (uint32_t i = 0; i < count; i++) {
                if (state[i] == 0 && ref_array_intersects(
                        &arrays[start + item], &arrays[start + i])) {
                    state[i] = 2;
                }
            }
        } else {
            state[item] = 2;
        }
    }
    for (uint32_t i = 0; i < count; i++) {
        arrays[start + i].selected = state[i] == 1;
    }
    RefScore recovered;
    uint32_t selected_items[64];
    uint32_t selected_count = 0;
    if (count > G_N_ELEMENTS(selected_items)) goto done;
    for (uint32_t i = 0; i < count; i++) {
        if (state[i] == 1) selected_items[selected_count++] = start + i;
    }
    if (!ref_array_schedule_score(arrays, selected_items, selected_count,
                                  candidates, &recovered) ||
        ref_score_compare(&recovered, &target) != 0) goto done;
    ok = true;

done:
    free(key_order);
    free(state);
    return ok;
}

static bool ref_find_aggregate(const RefGroup *groups, uint32_t group_count,
                               const RefArray *arrays, uint32_t array_count,
                               const OspreyAddress *base,
                               uint8_t *kind_out)
{
    bool found = false;
    uint8_t kind = 0;
    for (uint32_t i = 0; i < group_count; i++) {
        if (!ref_address_equal(&groups[i].base, base)) continue;
        if (found) return false;
        found = true;
        kind = OSPREY_TYPE_STRUCT;
    }
    for (uint32_t i = 0; i < array_count; i++) {
        OspreyAddress array_base = {
            .region = arrays[i].region, .offset = arrays[i].lo,
        };
        if (!arrays[i].selected || !ref_address_equal(&array_base, base)) continue;
        if (found) return false;
        found = true;
        kind = OSPREY_TYPE_ARRAY;
    }
    if (kind_out != NULL) *kind_out = kind;
    return true;
}

static bool ref_type_ref_equal(const RefTypeRef *a, const RefTypeRef *b)
{
    if (a->kind != b->kind) return false;
    if (a->kind == OSPREY_TYPE_PRIMITIVE) {
        return a->primitive_size == b->primitive_size;
    }
    return a->target_is_void == b->target_is_void &&
           a->target_aggregate_kind == b->target_aggregate_kind &&
           ref_address_equal(&a->target, &b->target);
}

static int ref_type_ref_compare(const RefTypeRef *a, const RefTypeRef *b)
{
    int c = ref_u64(a->kind, b->kind);
    if (c != 0) return c;
    if (a->kind == OSPREY_TYPE_PRIMITIVE) {
        return ref_u64(a->primitive_size, b->primitive_size);
    }
    c = ref_u64(a->target_is_void, b->target_is_void);
    if (c != 0) return c;
    c = ref_u64(a->target_aggregate_kind, b->target_aggregate_kind);
    return c != 0 ? c : ref_address_compare(&a->target, &b->target);
}

static int ref_type_work_compare(const RefTypeWork *a, const RefTypeWork *b,
                                 const RefFieldWork *fields)
{
    int c = ref_u64(a->kind, b->kind);
    if (c != 0) return c;
    if (a->kind == OSPREY_TYPE_PRIMITIVE) return ref_u64(a->size, b->size);
    if (a->kind == OSPREY_TYPE_POINTER) {
        c = ref_u64(a->target_is_void, b->target_is_void);
        if (c != 0) return c;
        c = ref_u64(a->target_aggregate_kind, b->target_aggregate_kind);
        return c != 0 ? c : ref_address_compare(&a->canonical_base,
                                                 &b->canonical_base);
    }
    c = ref_address_compare(&a->canonical_base, &b->canonical_base);
    if (c != 0) return c;
    if (a->kind == OSPREY_TYPE_ARRAY) {
        c = ref_u64(a->size, b->size);
        if (c != 0) return c;
        c = ref_u64(a->element_count, b->element_count);
        if (c != 0) return c;
        c = ref_u64(a->element_size, b->element_size);
        return c != 0 ? c : ref_type_ref_compare(&a->element_ref,
                                                  &b->element_ref);
    }
    c = ref_u64(a->field_count, b->field_count);
    if (c != 0) return c;
    for (uint32_t i = 0; i < a->field_count; i++) {
        const RefFieldWork *af = &fields[a->field_begin + i];
        const RefFieldWork *bf = &fields[b->field_begin + i];
        c = ref_u64(af->field.relative_offset, bf->field.relative_offset);
        if (c != 0) return c;
        c = ref_chunk_compare(&af->field.chunk, &bf->field.chunk);
        if (c != 0) return c;
        c = ref_type_ref_compare(&af->value_ref, &bf->value_ref);
        if (c != 0) return c;
    }
    return 0;
}

static bool ref_type_append(RefTypeWork **works, uint32_t *count,
                            uint32_t *capacity, const RefTypeWork *value,
                            uint32_t *id_out)
{
    RefTypeWork *grown;
    if (*count == *capacity) {
        uint32_t next = *capacity == 0 ? 16u : *capacity * 2u;
        if (next < *capacity) return false;
        grown = realloc(*works, (size_t)next * sizeof(**works));
        if (grown == NULL) return false;
        *works = grown;
        *capacity = next;
    }
    (*works)[*count] = *value;
    (*works)[*count].old_index = *count;
    if (id_out != NULL) *id_out = *count;
    (*count)++;
    return true;
}

static bool ref_type_add_primitive(RefTypeWork **works, uint32_t *count,
                                   uint32_t *capacity, uint64_t size,
                                   uint32_t *id_out)
{
    if (size == 0) return false;
    for (uint32_t i = 0; i < *count; i++) {
        if ((*works)[i].kind == OSPREY_TYPE_PRIMITIVE &&
            (*works)[i].size == size) {
            if (id_out != NULL) *id_out = i;
            return true;
        }
    }
    RefTypeWork value;
    memset(&value, 0, sizeof(value));
    value.kind = OSPREY_TYPE_PRIMITIVE;
    value.size = size;
    return ref_type_append(works, count, capacity, &value, id_out);
}

static bool ref_type_add_pointer(RefTypeWork **works, uint32_t *count,
                                 uint32_t *capacity, const RefTypeRef *ref,
                                 uint32_t *id_out)
{
    for (uint32_t i = 0; i < *count; i++) {
        if ((*works)[i].kind != OSPREY_TYPE_POINTER) continue;
        if ((*works)[i].target_is_void == ref->target_is_void &&
            (*works)[i].target_aggregate_kind == ref->target_aggregate_kind &&
            ref_address_equal(&(*works)[i].canonical_base, &ref->target)) {
            if (id_out != NULL) *id_out = i;
            return true;
        }
    }
    RefTypeWork value;
    memset(&value, 0, sizeof(value));
    value.kind = OSPREY_TYPE_POINTER;
    value.size = sizeof(target_ulong);
    value.target_is_void = ref->target_is_void;
    value.target_aggregate_kind = ref->target_aggregate_kind;
    value.canonical_base = ref->target;
    value.target_ref = *ref;
    return ref_type_append(works, count, capacity, &value, id_out);
}

static bool ref_pointer_type_ref(const RefDecision *decision,
                                 const RefCandidate *candidates,
                                 const RefGroup *groups, uint32_t group_count,
                                 const RefArray *arrays, uint32_t array_count,
                                 RefTypeRef *out)
{
    uint8_t aggregate_kind = 0;
    memset(out, 0, sizeof(*out));
    if (decision->pointer < 0) {
        out->kind = OSPREY_TYPE_PRIMITIVE;
        out->primitive_size = decision->chunk.size;
        return decision->chunk.size != 0;
    }
    if (!ref_find_aggregate(groups, group_count, arrays, array_count,
                             &candidates[decision->pointer].value.payload
                                  .attached.base,
                             &aggregate_kind)) return false;
    out->kind = OSPREY_TYPE_POINTER;
    out->target_is_void = aggregate_kind == 0;
    out->target_aggregate_kind = aggregate_kind;
    out->target = candidates[decision->pointer].value.payload.attached.base;
    return true;
}

static bool ref_build_model(const OspreyReferenceCase *input,
                            const RefCandidate *candidates,
                            uint32_t candidate_count,
                            RefDecision *decisions, uint32_t decision_count,
                            RefGroup *groups, uint32_t group_count,
                            RefArray *arrays, uint32_t array_count,
                            OspreyReferenceResult *result)
{
    RefTypeWork *works = NULL;
    RefFieldWork *fields = NULL;
    uint32_t work_count = 0;
    uint32_t work_capacity = 0;
    uint32_t field_count = 0;
    uint32_t field_capacity = 0;
    uint32_t *object_work = NULL;
    uint32_t *remap = NULL;
    uint32_t max_aggregate = 0;
    bool ok = false;

    (void)input;
    if (group_count > UINT32_MAX - array_count) return false;
    max_aggregate = group_count + array_count;
    object_work = decision_count == 0 ? NULL :
        malloc((size_t)decision_count * sizeof(*object_work));
    result->objects = decision_count == 0 ? NULL :
        calloc(decision_count, sizeof(*result->objects));
    if ((decision_count != 0 && (object_work == NULL || result->objects == NULL))) {
        goto done;
    }
    result->object_count = decision_count;

    for (uint32_t i = 0; i < decision_count; i++) {
        RefDecision *decision = &decisions[i];
        RefTypeRef ref;
        uint32_t type_id;
        if (!ref_pointer_type_ref(decision, candidates, groups, group_count,
                                  arrays, array_count, &ref)) goto done;
        if (ref.kind == OSPREY_TYPE_PRIMITIVE) {
            if (!ref_type_add_primitive(&works, &work_count, &work_capacity,
                                        ref.primitive_size, &type_id)) goto done;
        } else if (!ref_type_add_pointer(&works, &work_count, &work_capacity,
                                         &ref, &type_id)) goto done;
        object_work[i] = type_id;
        OspreyDecodedObject *object = &result->objects[i];
        object->chunk = decision->chunk;
        object->storage_role = decision->final_kind;
        if (decision->final_kind == OSPREY_STORAGE_FIELD) {
            object->owner_base = decision->owner;
        } else if (decision->final_kind == OSPREY_STORAGE_ARRAY_ELEMENT) {
            object->owner_base = decision->array_owner;
        }
        if (decision->final_kind == OSPREY_STORAGE_ARRAY_ELEMENT) {
            object->storage_posterior = decision->array_posterior;
            object->storage_support = decision->array_support;
            object->storage_source_rule_bits = decision->array_rules;
        } else if (decision->role >= 0) {
            object->storage_posterior = candidates[decision->role].value.posterior;
            object->storage_support = candidates[decision->role].value.direct_support;
            object->storage_source_rule_bits =
                candidates[decision->role].value.source_rule_bits;
        }
        if (decision->pointer >= 0) {
            object->has_pointer_target = 1;
            object->pointer_target = candidates[decision->pointer].value.payload
                .attached.base;
            object->pointer_posterior = candidates[decision->pointer].value.posterior;
            object->pointer_support = candidates[decision->pointer].value.direct_support;
            object->pointer_source_rule_bits = candidates[decision->pointer].value
                .source_rule_bits;
        }
    }

    for (uint32_t i = 0; i < group_count; i++) {
        RefTypeWork value;
        memset(&value, 0, sizeof(value));
        value.kind = OSPREY_TYPE_STRUCT;
        value.canonical_base = groups[i].base;
        value.field_begin = field_count;
        value.field_count = groups[i].count;
        value.posterior = 1.0;
        value.direct_support = UINT64_MAX;
        for (uint32_t j = 0; j < groups[i].count; j++) {
            uint32_t ordinal = groups[i].decisions[j];
            RefDecision *decision = &decisions[ordinal];
            RefTypeRef ref;
            int64_t relative;
            int64_t end;
            if (!ref_sub_i64(decision->chunk.address.offset,
                             groups[i].base.offset, &relative) || relative < 0 ||
                !ref_chunk_end(&decision->chunk, &end) ||
                !ref_pointer_type_ref(decision, candidates, groups, group_count,
                                      arrays, array_count, &ref)) goto done;
            if (field_count == field_capacity) {
                uint32_t next = field_capacity == 0 ? 16u : field_capacity * 2u;
                RefFieldWork *grown = realloc(fields, (size_t)next * sizeof(*fields));
                if (grown == NULL) goto done;
                fields = grown;
                field_capacity = next;
            }
            RefFieldWork *field = &fields[field_count++];
            memset(field, 0, sizeof(*field));
            field->field.chunk = decision->chunk;
            field->field.relative_offset = (uint64_t)relative;
            field->field.value_type_id = object_work[ordinal];
            field->field.posterior = candidates[decision->role].value.posterior;
            field->field.support = candidates[decision->role].value.direct_support;
            field->field.source_rule_bits = candidates[decision->role].value.source_rule_bits;
            field->value_ref = ref;
            if ((uint64_t)relative > UINT64_MAX - decision->chunk.size ||
                (uint64_t)relative + decision->chunk.size > (uint64_t)INT64_MAX) {
                goto done;
            }
            if (value.size < (uint64_t)relative + decision->chunk.size) {
                value.size = (uint64_t)relative + decision->chunk.size;
            }
            if (value.posterior > field->field.posterior) value.posterior =
                field->field.posterior;
            if (value.direct_support > field->field.support) value.direct_support =
                field->field.support;
            value.source_rule_bits |= field->field.source_rule_bits;
        }
        if (value.size == 0) goto done;
        if (value.direct_support == UINT64_MAX) value.direct_support = 0;
        if (!ref_type_append(&works, &work_count, &work_capacity, &value, NULL)) {
            goto done;
        }
    }

    for (uint32_t i = 0; i < array_count; i++) {
        RefArray *array = &arrays[i];
        RefTypeWork value;
        RefTypeRef common;
        bool homogeneous = array->member_count == array->count;
        memset(&value, 0, sizeof(value));
        memset(&common, 0, sizeof(common));
        value.kind = OSPREY_TYPE_ARRAY;
        value.canonical_base.region = array->region;
        value.canonical_base.offset = array->lo;
        value.size = (uint64_t)(array->hi - array->lo);
        value.element_count = array->count;
        value.element_size = array->stride;
        value.posterior = candidates[array->candidate].value.posterior;
        value.direct_support = candidates[array->candidate].value.direct_support;
        value.source_rule_bits = candidates[array->candidate].value.source_rule_bits;
        for (uint32_t j = 0; homogeneous && j < array->member_count; j++) {
            uint32_t ordinal = array->members[j];
            RefDecision *decision = &decisions[ordinal];
            RefTypeRef ref;
            int64_t delta;
            uint64_t position;
            if (decision->chunk.size != array->stride ||
                !ref_sub_i64(decision->chunk.address.offset, array->lo,
                             &delta) || delta < 0 ||
                (uint64_t)delta % array->stride != 0 ||
                !ref_pointer_type_ref(decision, candidates, groups, group_count,
                                      arrays, array_count, &ref)) {
                homogeneous = false;
                break;
            }
            position = (uint64_t)delta / array->stride;
            if (position >= array->count) {
                homogeneous = false;
                break;
            }
            for (uint32_t k = 0; k < j; k++) {
                int64_t previous_delta;
                if (!ref_sub_i64(decisions[array->members[k]].chunk.address.offset,
                                array->lo, &previous_delta) ||
                    previous_delta == delta) {
                    homogeneous = false;
                    break;
                }
            }
            if (!homogeneous) break;
            if (j == 0) common = ref;
            else if (!ref_type_ref_equal(&common, &ref)) homogeneous = false;
        }
        if (homogeneous && array->member_count != 0) {
            for (uint32_t j = 0; j < array->member_count; j++) {
                RefTypeRef ref;
                if (!ref_pointer_type_ref(&decisions[array->members[j]],
                                          candidates, groups, group_count,
                                          arrays, array_count, &ref) ||
                    !ref_type_ref_equal(&common, &ref)) {
                    homogeneous = false;
                    break;
                }
            }
        }
        if (!homogeneous) {
            if (!ref_type_add_primitive(&works, &work_count, &work_capacity,
                                        array->stride, &value.element_work_id)) goto done;
        } else {
            bool found = false;
            value.element_ref = common;
            for (uint32_t j = 0; j < work_count; j++) {
                RefTypeRef ref;
                memset(&ref, 0, sizeof(ref));
                if (works[j].kind == OSPREY_TYPE_PRIMITIVE) {
                    ref.kind = OSPREY_TYPE_PRIMITIVE;
                    ref.primitive_size = works[j].size;
                } else if (works[j].kind == OSPREY_TYPE_POINTER) {
                    ref.kind = OSPREY_TYPE_POINTER;
                    ref.target_is_void = works[j].target_is_void;
                    ref.target_aggregate_kind = works[j].target_aggregate_kind;
                    ref.target = works[j].canonical_base;
                }
                if (ref_type_ref_equal(&ref, &common)) {
                    value.element_work_id = j;
                    found = true;
                    break;
                }
            }
            if (!found) goto done;
        }
        if (!ref_type_append(&works, &work_count, &work_capacity, &value, NULL)) goto done;
    }

    for (uint32_t i = 1; i < work_count; i++) {
        RefTypeWork value = works[i];
        uint32_t j = i;
        while (j != 0 && ref_type_work_compare(&value, &works[j - 1],
                                               fields) < 0) {
            works[j] = works[j - 1];
            j--;
        }
        works[j] = value;
    }
    for (uint32_t i = 1; i < work_count; i++) {
        if (ref_type_work_compare(&works[i - 1], &works[i], fields) == 0) goto done;
    }
    remap = work_count == 0 ? NULL : malloc((size_t)work_count * sizeof(*remap));
    if (work_count != 0 && remap == NULL) goto done;
    for (uint32_t i = 0; i < work_count; i++) {
        if (works[i].old_index >= work_count) goto done;
        remap[works[i].old_index] = i;
    }
    result->type_count = work_count;
    result->types = work_count == 0 ? NULL : calloc(work_count, sizeof(*result->types));
    result->fields = field_count == 0 ? NULL : calloc(field_count, sizeof(*result->fields));
    result->field_count = field_count;
    if ((work_count != 0 && result->types == NULL) ||
        (field_count != 0 && result->fields == NULL)) goto done;
    for (uint32_t i = 0; i < field_count; i++) {
        result->fields[i] = fields[i].field;
        if (result->fields[i].value_type_id >= work_count) goto done;
        result->fields[i].value_type_id = remap[result->fields[i].value_type_id];
    }
    for (uint32_t i = 0; i < work_count; i++) {
        RefTypeWork *work = &works[i];
        OspreyDecodedType *type = &result->types[i];
        memset(type, 0, sizeof(*type));
        type->id = i;
        type->kind = work->kind;
        type->target_is_void = work->kind == OSPREY_TYPE_POINTER ?
            work->target_is_void : 0;
        type->size = work->size;
        type->element_count = work->element_count;
        type->element_size = work->element_size;
        type->element_type_id = UINT32_MAX;
        type->target_type_id = UINT32_MAX;
        type->canonical_base = work->canonical_base;
        type->field_begin = work->field_begin;
        type->field_count = work->field_count;
        type->evidence_valid = work->kind == OSPREY_TYPE_ARRAY ||
                               work->kind == OSPREY_TYPE_STRUCT;
        type->posterior = type->evidence_valid ? work->posterior : 0.0;
        type->direct_support = type->evidence_valid ? work->direct_support : 0;
        type->source_rule_bits = type->evidence_valid ? work->source_rule_bits : 0;
        if (work->kind == OSPREY_TYPE_ARRAY) {
            if (work->element_work_id >= work_count) goto done;
            type->element_type_id = remap[work->element_work_id];
        }
    }
    for (uint32_t i = 0; i < work_count; i++) {
        RefTypeWork *work = &works[i];
        if (work->kind != OSPREY_TYPE_POINTER || work->target_is_void) continue;
        uint32_t target = UINT32_MAX;
        for (uint32_t j = 0; j < work_count; j++) {
            if ((result->types[j].kind == OSPREY_TYPE_STRUCT ||
                 result->types[j].kind == OSPREY_TYPE_ARRAY) &&
                ref_address_equal(&result->types[j].canonical_base,
                                  &work->canonical_base)) {
                if (target != UINT32_MAX) goto done;
                target = j;
            }
        }
        if (target == UINT32_MAX) goto done;
        result->types[i].target_type_id = target;
    }
    for (uint32_t i = 0; i < decision_count; i++) {
        result->objects[i].value_type_id = remap[object_work[i]];
    }
    result->type_name_count = work_count;
    result->type_names = work_count == 0 ? NULL : calloc(work_count, sizeof(*result->type_names));
    if (work_count != 0 && result->type_names == NULL) goto done;
    for (uint32_t i = 0; i < work_count; i++) {
        char address[128];
        char aggregate[256];
        char name[512];
        OspreyDecodedType *type = &result->types[i];
        uint64_t magnitude = type->canonical_base.offset < 0 ?
            0 - (uint64_t)type->canonical_base.offset :
            (uint64_t)type->canonical_base.offset;
        const char *tag = type->canonical_base.region.kind == OSPREY_REGION_GLOBAL ? "g" :
            (type->canonical_base.region.kind == OSPREY_REGION_HEAP_SITE ? "h" : "s");
        if (snprintf(address, sizeof(address), "%s_i%016" PRIx64 "_s%016" PRIx64
                     "_o%c%016" PRIx64, tag,
                     type->canonical_base.region.code_image_id,
                     type->canonical_base.region.site_offset,
                     type->canonical_base.offset < 0 ? 'n' : 'p', magnitude) < 0) goto done;
        if (type->kind == OSPREY_TYPE_PRIMITIVE) {
            if (snprintf(name, sizeof(name), "prim_b%" PRIu64, type->size) < 0) goto done;
        } else if (type->kind == OSPREY_TYPE_STRUCT) {
            if (snprintf(name, sizeof(name), "struct_%s", address) < 0) goto done;
        } else if (type->kind == OSPREY_TYPE_ARRAY) {
            if (snprintf(name, sizeof(name), "array_%s_stride_%016" PRIx64,
                         address, type->element_size) < 0) goto done;
        } else if (type->kind == OSPREY_TYPE_POINTER) {
            if (type->target_is_void) {
                if (snprintf(name, sizeof(name), "ptr_void_%s", address) < 0) goto done;
            } else {
                if (type->target_type_id >= work_count) goto done;
                OspreyDecodedType *target = &result->types[type->target_type_id];
                uint64_t target_mag = target->canonical_base.offset < 0 ?
                    0 - (uint64_t)target->canonical_base.offset :
                    (uint64_t)target->canonical_base.offset;
                const char *target_tag = target->canonical_base.region.kind ==
                    OSPREY_REGION_GLOBAL ? "g" :
                    (target->canonical_base.region.kind == OSPREY_REGION_HEAP_SITE ? "h" : "s");
                if (target->kind == OSPREY_TYPE_STRUCT) {
                    if (snprintf(aggregate, sizeof(aggregate),
                                 "struct_%s_i%016" PRIx64 "_s%016" PRIx64
                                 "_o%c%016" PRIx64, target_tag,
                                 target->canonical_base.region.code_image_id,
                                 target->canonical_base.region.site_offset,
                                 target->canonical_base.offset < 0 ? 'n' : 'p',
                                 target_mag) < 0) goto done;
                } else if (target->kind == OSPREY_TYPE_ARRAY) {
                    if (snprintf(aggregate, sizeof(aggregate),
                                 "array_%s_i%016" PRIx64 "_s%016" PRIx64
                                 "_o%c%016" PRIx64 "_stride_%016" PRIx64,
                                 target_tag, target->canonical_base.region.code_image_id,
                                 target->canonical_base.region.site_offset,
                                 target->canonical_base.offset < 0 ? 'n' : 'p', target_mag,
                                 target->element_size) < 0) goto done;
                } else goto done;
                if (snprintf(name, sizeof(name), "ptr_to_%s", aggregate) < 0) goto done;
            }
        } else goto done;
        result->type_names[i] = strdup(name);
        if (result->type_names[i] == NULL) goto done;
    }
    /* Build the three semantic indexes independently from the object/type
     * arrays, then sort by their complete keys. */
    result->chunk_index_count = decision_count;
    result->chunk_index = decision_count == 0 ? NULL :
        calloc(decision_count, sizeof(*result->chunk_index));
    if (decision_count != 0 && result->chunk_index == NULL) goto done;
    for (uint32_t i = 0; i < decision_count; i++) {
        OspreyKey key;
        memset(&key, 0, sizeof(key));
        key.tag = 0x43484bULL;
        key.w[0] = result->objects[i].chunk.address.region.kind;
        key.w[1] = result->objects[i].chunk.address.region.code_image_id;
        key.w[2] = result->objects[i].chunk.address.region.site_offset;
        key.w[3] = (uint64_t)result->objects[i].chunk.address.offset;
        key.w[4] = result->objects[i].chunk.size;
        result->chunk_index[i].key = key;
        result->chunk_index[i].ordinal = i;
    }
    result->aggregate_index_count = max_aggregate;
    result->aggregate_index = max_aggregate == 0 ? NULL :
        calloc(max_aggregate, sizeof(*result->aggregate_index));
    if (max_aggregate != 0 && result->aggregate_index == NULL) goto done;
    uint32_t aggregate_position = 0;
    for (uint32_t i = 0; i < work_count; i++) {
        if (result->types[i].kind != OSPREY_TYPE_STRUCT &&
            result->types[i].kind != OSPREY_TYPE_ARRAY) continue;
        OspreyKey key;
        memset(&key, 0, sizeof(key));
        key.tag = 0x414747ULL;
        key.w[0] = result->types[i].canonical_base.region.kind;
        key.w[1] = result->types[i].canonical_base.region.code_image_id;
        key.w[2] = result->types[i].canonical_base.region.site_offset;
        key.w[3] = (uint64_t)result->types[i].canonical_base.offset;
        key.w[4] = result->types[i].kind;
        result->aggregate_index[aggregate_position].key = key;
        result->aggregate_index[aggregate_position++].ordinal = i;
    }
    result->type_index_count = work_count;
    result->type_index = work_count == 0 ? NULL :
        calloc(work_count, sizeof(*result->type_index));
    if (work_count != 0 && result->type_index == NULL) goto done;
    for (uint32_t i = 0; i < work_count; i++) {
        OspreyKey key;
        OspreyDecodedType *type = &result->types[i];
        memset(&key, 0, sizeof(key));
        key.tag = 0x545950ULL;
        key.w[0] = type->kind;
        key.w[1] = type->size;
        if (type->kind != OSPREY_TYPE_PRIMITIVE) {
            key.w[2] = type->canonical_base.region.kind;
            key.w[3] = type->canonical_base.region.code_image_id;
            key.w[4] = type->canonical_base.region.site_offset;
            key.w[5] = (uint64_t)type->canonical_base.offset;
            if (type->kind == OSPREY_TYPE_POINTER) {
                key.w[6] = type->target_is_void;
                key.w[7] = type->target_is_void ? 0 : type->target_type_id;
            } else if (type->kind == OSPREY_TYPE_ARRAY) {
                key.w[6] = type->element_count;
                key.w[7] = type->element_size;
                key.w[8] = type->element_type_id;
            } else {
                key.w[6] = type->field_count;
                key.w[7] = type->field_begin;
            }
        }
        result->type_index[i].key = key;
        result->type_index[i].ordinal = i;
    }
    ok = true;

done:
    free(works);
    free(fields);
    free(object_work);
    free(remap);
    return ok;
}

static int ref_key_compare(const void *ap, const void *bp)
{
    const OspreyKey *a = ap;
    const OspreyKey *b = bp;
    int c = ref_u64(a->tag, b->tag);
    if (c != 0) return c;
    for (size_t i = 0; i < G_N_ELEMENTS(a->w); i++) {
        c = ref_u64(a->w[i], b->w[i]);
        if (c != 0) return c;
    }
    return 0;
}

static int ref_index_compare(const void *ap, const void *bp)
{
    const OspreyModelIndexEntry *a = ap;
    const OspreyModelIndexEntry *b = bp;
    return ref_key_compare(&a->key, &b->key);
}

static bool ref_prepare_candidates(const OspreyReferenceCase *input,
                                   RefCandidate **eligible_out,
                                   uint32_t *eligible_count_out,
                                   OspreyReferenceResult *result)
{
    RefCandidate *eligible = NULL;
    uint32_t count = 0;
    if (input == NULL || eligible_out == NULL || eligible_count_out == NULL ||
        !isfinite(input->report_threshold) || input->report_threshold < 0.0 ||
        input->report_threshold > 1.0 ||
        (input->candidate_count != 0 && input->candidates == NULL) ||
        (input->extent_count != 0 && input->extents == NULL)) return false;
    for (uint32_t i = 0; i < input->extent_count; i++) {
        if (!ref_region_valid(&input->extents[i].region) ||
            input->extents[i].lo > input->extents[i].hi ||
            (i != 0 && ref_region_compare(&input->extents[i - 1].region,
                                           &input->extents[i].region) >= 0)) return false;
    }
    for (uint32_t i = 0; i < input->candidate_count; i++) {
        const OspreyReferenceCandidate *source = &input->candidates[i];
        if (!ref_payload_valid(source->kind, &source->payload) ||
            !ref_belief_valid(source) || source->hard_false > 1) return false;
        if (!ref_is_projected(source->kind)) continue;
        if (source->hard_false && source->kind != OSPREY_PRED_ARRAY) return false;
        if (source->hard_false) {
            result->discarded_hard_false++;
            continue;
        }
        if (source->posterior < input->report_threshold) {
            result->discarded_threshold++;
            continue;
        }
        RefCandidate *grown = realloc(eligible, (size_t)(count + 1u) * sizeof(*eligible));
        if (grown == NULL) {
            free(eligible);
            return false;
        }
        eligible = grown;
        eligible[count].value = *source;
        count++;
    }
    if (count > 1) qsort(eligible, count, sizeof(*eligible), ref_candidate_compare);
    for (uint32_t i = 1; i < count; i++) {
        if (ref_candidate_compare(&eligible[i - 1], &eligible[i]) == 0) {
            free(eligible);
            return false;
        }
    }
    *eligible_out = eligible;
    *eligible_count_out = count;
    return true;
}

static bool ref_build_decisions(const OspreyReferenceCase *input,
                                const RefCandidate *candidates,
                                uint32_t candidate_count,
                                RefDecision **decisions_out,
                                uint32_t *decision_count_out,
                                RefGroup **groups_out,
                                uint32_t *group_count_out)
{
    RefDecision *decisions = NULL;
    uint32_t decision_count = 0;
    RefGroup *groups = NULL;
    uint32_t group_count = 0;
    uint8_t *scheduled = NULL;
    if (decisions_out == NULL || decision_count_out == NULL ||
        groups_out == NULL || group_count_out == NULL) return false;
    for (uint32_t i = 0; i < candidate_count; i++) {
        const OspreyChunk *chunk = ref_candidate_chunk(&candidates[i]);
        if (candidates[i].value.kind != OSPREY_PRED_PRIMITIVE_VAR &&
            candidates[i].value.kind != OSPREY_PRED_SCALAR &&
            candidates[i].value.kind != OSPREY_PRED_FIELD_OF &&
            candidates[i].value.kind != OSPREY_PRED_POINTER) continue;
        if (ref_find_decision(decisions, decision_count, chunk) < 0) {
            RefDecision *grown = realloc(decisions,
                (size_t)(decision_count + 1u) * sizeof(*decisions));
            if (grown == NULL) goto fail;
            decisions = grown;
            memset(&decisions[decision_count], 0, sizeof(*decisions));
            decisions[decision_count].chunk = *chunk;
            decisions[decision_count].primitive = -1;
            decisions[decision_count].scalar = -1;
            decisions[decision_count].field_winner = -1;
            decisions[decision_count].pointer = -1;
            decisions[decision_count].role = -1;
            decision_count++;
        }
    }
    if (decision_count > 1) qsort(decisions, decision_count, sizeof(*decisions),
                                  ref_decision_compare);
    for (uint32_t i = 0; i < decision_count; i++) {
        const OspreyRegionExtent *extent = ref_extent(
            input->extents, input->extent_count,
            &decisions[i].chunk.address.region);
        int64_t end;
        if (extent == NULL || !ref_chunk_end(&decisions[i].chunk, &end) ||
            decisions[i].chunk.address.offset < extent->lo ||
            end > extent->hi) {
            goto fail;
        }
    }
    for (uint32_t i = 0; i < candidate_count; i++) {
        int ordinal;
        const OspreyChunk *chunk = ref_candidate_chunk(&candidates[i]);
        switch (candidates[i].value.kind) {
        case OSPREY_PRED_PRIMITIVE_VAR:
        case OSPREY_PRED_SCALAR:
        case OSPREY_PRED_FIELD_OF:
        case OSPREY_PRED_POINTER:
            ordinal = ref_find_decision(decisions, decision_count, chunk);
            if (ordinal < 0) goto fail;
            if (candidates[i].value.kind == OSPREY_PRED_PRIMITIVE_VAR) {
                if (decisions[ordinal].primitive >= 0) goto fail;
                decisions[ordinal].primitive = (int32_t)i;
            } else if (candidates[i].value.kind == OSPREY_PRED_SCALAR) {
                if (decisions[ordinal].scalar >= 0) goto fail;
                decisions[ordinal].scalar = (int32_t)i;
            } else if (candidates[i].value.kind == OSPREY_PRED_FIELD_OF) {
                if (decisions[ordinal].field_winner < 0 ||
                    ref_candidate_better(&candidates[i],
                        &candidates[decisions[ordinal].field_winner])) {
                    decisions[ordinal].field_winner = (int32_t)i;
                }
            } else {
                if (decisions[ordinal].pointer < 0 ||
                    ref_candidate_better(&candidates[i],
                        &candidates[decisions[ordinal].pointer])) {
                    decisions[ordinal].pointer = (int32_t)i;
                }
            }
            break;
        default:
            break;
        }
    }
    scheduled = candidate_count == 0 ? NULL : calloc(candidate_count, 1);
    if (candidate_count != 0 && scheduled == NULL) goto fail;
    for (uint32_t i = 0; i < decision_count; i++) {
        if (decisions[i].field_winner >= 0) {
            RefCandidate *field = (RefCandidate *)&candidates[decisions[i].field_winner];
            if (!ref_field_geometry_valid(input, field)) goto fail;
            uint32_t group = 0;
            for (; group < group_count; group++) {
                if (ref_address_equal(&groups[group].base,
                    &field->value.payload.attached.base)) break;
            }
            if (group == group_count) {
                RefGroup *grown = realloc(groups,
                    (size_t)(group_count + 1u) * sizeof(*groups));
                if (grown == NULL) goto fail;
                groups = grown;
                memset(&groups[group_count], 0, sizeof(*groups));
                groups[group_count].base = field->value.payload.attached.base;
                group_count++;
            }
            if (!ref_append_u32(&groups[group].decisions,
                                &groups[group].count, i)) goto fail;
        }
    }
    if (group_count > 1) qsort(groups, group_count, sizeof(*groups), ref_group_compare);
    /* Schedule each base independently. */
    for (uint32_t group = 0; group < group_count; group++) {
        uint32_t *field_candidates = NULL;
        uint32_t field_count = 0;
        for (uint32_t j = 0; j < groups[group].count; j++) {
            uint32_t ordinal = groups[group].decisions[j];
            if (!ref_append_u32(&field_candidates, &field_count,
                                (uint32_t)decisions[ordinal].field_winner)) {
                free(field_candidates);
                goto fail;
            }
        }
        if (!ref_schedule_group(candidates, field_candidates, field_count,
                                scheduled, NULL)) {
            free(field_candidates);
            goto fail;
        }
        free(field_candidates);
    }
    for (uint32_t i = 0; i < decision_count; i++) {
        RefDecision *decision = &decisions[i];
        const RefCandidate *primitive = decision->primitive < 0 ? NULL :
            &candidates[decision->primitive];
        const RefCandidate *scalar = decision->scalar < 0 ? NULL :
            &candidates[decision->scalar];
        const RefCandidate *field = decision->field_winner < 0 ||
            !scheduled[decision->field_winner] ? NULL :
            &candidates[decision->field_winner];
        const RefCandidate *role = field;
        if (scalar != NULL && ref_candidate_better(scalar, role)) role = scalar;
        if (role != NULL) {
            decision->role = (int32_t)(role - candidates);
            decision->role_kind = role->value.kind == OSPREY_PRED_SCALAR ?
                OSPREY_STORAGE_SCALAR : OSPREY_STORAGE_FIELD;
            decision->final_kind = decision->role_kind;
            decision->owner = role->value.payload.attached.base;
        } else if (primitive != NULL) {
            decision->role = decision->primitive;
            decision->role_kind = OSPREY_STORAGE_PRIMITIVE;
            decision->final_kind = OSPREY_STORAGE_PRIMITIVE;
        } else {
            decision->role_kind = OSPREY_STORAGE_PRIMITIVE;
            decision->final_kind = OSPREY_STORAGE_PRIMITIVE;
        }
        if (decision->pointer >= 0 &&
            !ref_pointer_geometry_valid(input, &candidates[decision->pointer])) goto fail;
    }
    free(scheduled);
    *decisions_out = decisions;
    *decision_count_out = decision_count;
    *groups_out = groups;
    *group_count_out = group_count;
    return true;
fail:
    free(scheduled);
    free(decisions);
    ref_free_groups(groups, group_count);
    return false;
}

static bool ref_build_arrays(const OspreyReferenceCase *input,
                             const RefCandidate *candidates,
                             uint32_t candidate_count,
                             RefDecision *decisions, uint32_t decision_count,
                             RefArray **arrays_out, uint32_t *array_count_out,
                             uint64_t *discarded_layout)
{
    RefArray *arrays = NULL;
    uint32_t count = 0;
    if (arrays_out == NULL || array_count_out == NULL || discarded_layout == NULL) return false;
    for (uint32_t i = 0; i < candidate_count; i++) {
        if (candidates[i].value.kind != OSPREY_PRED_ARRAY) continue;
        const OspreyAddress *lo = &candidates[i].value.payload.segment.a1;
        const OspreyAddress *hi = &candidates[i].value.payload.segment.a2;
        int64_t span;
        const OspreyRegionExtent *extent = ref_extent(input->extents,
                                                       input->extent_count,
                                                       &lo->region);
        if (!ref_region_equal(&lo->region, &hi->region) || lo->offset >= hi->offset ||
            candidates[i].value.payload.segment.size <= 0 ||
            candidates[i].value.payload.segment.size > INT64_MAX ||
            !ref_sub_i64(hi->offset, lo->offset, &span) || span <= 0 ||
            span < candidates[i].value.payload.segment.size || extent == NULL ||
            lo->offset < extent->lo || hi->offset > extent->hi) return false;
        if (span % candidates[i].value.payload.segment.size != 0) {
            (*discarded_layout)++;
            continue;
        }
        RefArray array;
        memset(&array, 0, sizeof(array));
        array.candidate = i;
        array.region = lo->region;
        array.lo = lo->offset;
        array.hi = hi->offset;
        array.stride = (uint64_t)candidates[i].value.payload.segment.size;
        array.count = (uint64_t)(span / candidates[i].value.payload.segment.size);
        if (!ref_score_probability(&array.score,
                                   candidates[i].value.posterior, false)) return false;
        for (uint32_t d = 0; d < decision_count; d++) {
            int64_t end;
            bool intersects;
            bool legal;
            if (!ref_region_equal(&decisions[d].chunk.address.region,
                                  &array.region)) continue;
            if (!ref_chunk_end(&decisions[d].chunk, &end)) return false;
            intersects = decisions[d].chunk.address.offset < array.hi &&
                         array.lo < end;
            if (!intersects) continue;
            int64_t delta;
            legal = ref_sub_i64(decisions[d].chunk.address.offset, array.lo,
                                &delta) && decisions[d].chunk.address.offset >= array.lo &&
                    end <= array.hi && delta >= 0 &&
                    (uint64_t)delta % array.stride == 0 &&
                    decisions[d].chunk.size <= array.stride;
            if (!legal) {
                free(array.members);
                (*discarded_layout)++;
                goto next_candidate;
            }
            if (!ref_append_u32(&array.members, &array.member_count, d)) {
                free(array.members);
                return false;
            }
            if (decisions[d].role >= 0 &&
                (decisions[d].role_kind == OSPREY_STORAGE_SCALAR ||
                 decisions[d].role_kind == OSPREY_STORAGE_FIELD) &&
                candidates[decisions[d].role].value.posterior > 0.5 &&
                !ref_score_probability(&array.score,
                    candidates[decisions[d].role].value.posterior, true)) {
                free(array.members);
                return false;
            }
        }
        RefArray *grown = realloc(arrays, (size_t)(count + 1u) * sizeof(*arrays));
        if (grown == NULL) {
            free(array.members);
            free(arrays);
            return false;
        }
        arrays = grown;
        arrays[count++] = array;
next_candidate:
        continue;
    }
    if (count == 0) {
        *arrays_out = NULL;
        *array_count_out = 0;
        return true;
    }
    /* Select one non-overlapping schedule per complete region with an
     * independent weighted-interval recurrence. */
    for (uint32_t start = 0; start < count;) {
        uint32_t end = start + 1u;
        while (end < count && ref_region_equal(&arrays[start].region,
                                               &arrays[end].region)) end++;
        if (!ref_schedule_array_region(arrays, start, end - start,
                                       candidates)) {
            for (uint32_t i = 0; i < count; i++) free(arrays[i].members);
            free(arrays);
            return false;
        }
        start = end;
    }
    uint32_t selected_count = 0;
    for (uint32_t i = 0; i < count; i++) if (arrays[i].selected) selected_count++;
    *discarded_layout += count - selected_count;
    *arrays_out = arrays;
    *array_count_out = count;
    return true;
}

static void ref_compact_selected_arrays(RefArray **arrays_io,
                                        uint32_t *count_io)
{
    RefArray *arrays = arrays_io == NULL ? NULL : *arrays_io;
    uint32_t count = count_io == NULL ? 0 : *count_io;
    uint32_t position = 0;
    if (arrays == NULL || count_io == NULL) return;
    for (uint32_t i = 0; i < count; i++) {
        if (!arrays[i].selected) {
            free(arrays[i].members);
            arrays[i].members = NULL;
            continue;
        }
        if (position != i) arrays[position] = arrays[i];
        arrays[position].selected = true;
        position++;
    }
    if (position == 0) {
        free(arrays);
        *arrays_io = NULL;
    } else {
        RefArray *shrunk = realloc(arrays, (size_t)position * sizeof(*arrays));
        *arrays_io = shrunk != NULL ? shrunk : arrays;
    }
    *count_io = position;
}

static bool ref_rebuild_groups_and_roles(const RefCandidate *candidates,
                                         RefDecision *decisions,
                                         uint32_t decision_count,
                                         RefGroup **groups_io,
                                         uint32_t *group_count_io,
                                         RefArray *arrays, uint32_t array_count)
{
    RefGroup *old = *groups_io;
    uint32_t old_count = *group_count_io;
    RefGroup *groups = NULL;
    uint32_t group_count = 0;
    for (uint32_t i = 0; i < array_count; i++) {
        if (!arrays[i].selected) continue;
        for (uint32_t j = 0; j < arrays[i].member_count; j++) {
            RefDecision *decision = &decisions[arrays[i].members[j]];
            decision->final_kind = OSPREY_STORAGE_ARRAY_ELEMENT;
            decision->has_array = 1;
            decision->array_owner = (OspreyAddress){
                .region = arrays[i].region, .offset = arrays[i].lo };
            decision->array_posterior = candidates[arrays[i].candidate].value.posterior;
            decision->array_support = candidates[arrays[i].candidate].value.direct_support;
            decision->array_rules = candidates[arrays[i].candidate].value.source_rule_bits;
        }
    }
    for (uint32_t i = 0; i < old_count; i++) {
        RefGroup group;
        memset(&group, 0, sizeof(group));
        group.base = old[i].base;
        for (uint32_t j = 0; j < old[i].count; j++) {
            uint32_t ordinal = old[i].decisions[j];
            if (decisions[ordinal].final_kind == OSPREY_STORAGE_FIELD &&
                !ref_append_u32(&group.decisions, &group.count, ordinal)) {
                free(group.decisions);
                ref_free_groups(groups, group_count);
                return false;
            }
        }
        if (group.count != 0) {
            RefGroup *grown = realloc(groups,
                                      (size_t)(group_count + 1u) * sizeof(*groups));
            if (grown == NULL) {
                free(group.decisions);
                ref_free_groups(groups, group_count);
                return false;
            }
            groups = grown;
            groups[group_count++] = group;
        } else {
            free(group.decisions);
        }
    }
    if (group_count > 1) qsort(groups, group_count, sizeof(*groups), ref_group_compare);
    ref_free_groups(old, old_count);
    *groups_io = groups;
    *group_count_io = group_count;
    return true;
}

static bool ref_same_candidate(const RefCandidate *a, const RefCandidate *b)
{
    return a != NULL && b != NULL && ref_candidate_compare(a, b) == 0;
}

static uint64_t ref_role_loss_count(const RefCandidate *candidates,
                                    uint32_t candidate_count,
                                    const RefDecision *decisions,
                                    uint32_t decision_count)
{
    uint64_t count = 0;
    for (uint32_t i = 0; i < candidate_count; i++) {
        const RefCandidate *candidate = &candidates[i];
        const OspreyChunk *chunk;
        int ordinal;
        bool retained = false;
        if (candidate->value.kind == OSPREY_PRED_ARRAY) continue;
        chunk = ref_candidate_chunk(candidate);
        ordinal = ref_find_decision(decisions, decision_count, chunk);
        if (ordinal >= 0) {
            const RefDecision *decision = &decisions[ordinal];
            if (candidate->value.kind == OSPREY_PRED_POINTER) {
                retained = decision->pointer >= 0 &&
                    ref_same_candidate(candidate, &candidates[decision->pointer]);
            } else {
                retained = decision->role >= 0 &&
                    ref_same_candidate(candidate, &candidates[decision->role]) &&
                    !(decision->final_kind == OSPREY_STORAGE_ARRAY_ELEMENT);
            }
        }
        if (!retained) count++;
    }
    return count;
}

static uint64_t ref_double_bits(double value)
{
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static bool ref_string_equal(const char *a, const char *b)
{
    return a != NULL && b != NULL && strcmp(a, b) == 0;
}

bool osprey_reference_decode(const OspreyReferenceCase *input,
                             OspreyReferenceResult *result)
{
    RefCandidate *candidates = NULL;
    uint32_t candidate_count = 0;
    RefDecision *decisions = NULL;
    uint32_t decision_count = 0;
    RefGroup *groups = NULL;
    uint32_t group_count = 0;
    RefArray *arrays = NULL;
    uint32_t array_count = 0;
    bool ok = false;

    if (result == NULL) return false;
    memset(result, 0, sizeof(*result));
    result->status = OSPREY_INVALID_MODEL;
    result->error = OSPREY_MODEL_VALIDATION_NONE;
    if (!ref_prepare_candidates(input, &candidates, &candidate_count, result) ||
        !ref_build_decisions(input, candidates, candidate_count, &decisions,
                             &decision_count, &groups, &group_count) ||
        !ref_build_arrays(input, candidates, candidate_count, decisions,
                          decision_count, &arrays, &array_count,
                          &result->discarded_layout)) goto done;
    ref_compact_selected_arrays(&arrays, &array_count);
    if (!ref_rebuild_groups_and_roles(candidates, decisions, decision_count,
                                      &groups, &group_count, arrays,
                                      array_count)) goto done;
    for (uint32_t i = 0; i < group_count; i++) {
        for (uint32_t j = 1; j < groups[i].count; j++) {
            uint32_t value = groups[i].decisions[j];
            uint32_t k = j;
            while (k != 0 && ref_chunk_compare(&decisions[value].chunk,
                       &decisions[groups[i].decisions[k - 1]].chunk) < 0) {
                groups[i].decisions[k] = groups[i].decisions[k - 1];
                k--;
            }
            groups[i].decisions[k] = value;
        }
    }
    for (uint32_t i = 0; i < group_count; i++) {
        for (uint32_t j = 0; j < array_count; j++) {
            if (!arrays[j].selected) continue;
            OspreyAddress base = { .region = arrays[j].region,
                                   .offset = arrays[j].lo };
            if (ref_address_equal(&groups[i].base, &base)) goto done;
        }
    }
    result->discarded_role = ref_role_loss_count(candidates, candidate_count,
                                                 decisions, decision_count);
    if (!ref_build_model(input, candidates, candidate_count, decisions,
                         decision_count, groups, group_count, arrays,
                         array_count, result)) goto done;
    if (result->type_index_count > 1) qsort(result->type_index,
                                             result->type_index_count,
                                             sizeof(*result->type_index),
                                             ref_index_compare);
    if (result->chunk_index_count > 1) qsort(result->chunk_index,
                                              result->chunk_index_count,
                                              sizeof(*result->chunk_index),
                                              ref_index_compare);
    if (result->aggregate_index_count > 1) qsort(result->aggregate_index,
                                                  result->aggregate_index_count,
                                                  sizeof(*result->aggregate_index),
                                                  ref_index_compare);
    result->status = OSPREY_OK;
    ok = true;

done:
    if (!ok) {
        result->status = OSPREY_INVALID_MODEL;
        osprey_reference_result_free(result);
    }
    free(candidates);
    free(decisions);
    ref_free_groups(groups, group_count);
    if (arrays != NULL) {
        for (uint32_t i = 0; i < array_count; i++) free(arrays[i].members);
    }
    free(arrays);
    return ok;
}

void osprey_reference_result_free(OspreyReferenceResult *result)
{
    if (result == NULL) return;
    free(result->objects);
    free(result->types);
    free(result->fields);
    free(result->chunk_index);
    free(result->aggregate_index);
    free(result->type_index);
    if (result->type_names != NULL) {
        for (uint32_t i = 0; i < result->type_name_count; i++) free(result->type_names[i]);
    }
    free(result->type_names);
    result->objects = NULL;
    result->types = NULL;
    result->fields = NULL;
    result->chunk_index = NULL;
    result->aggregate_index = NULL;
    result->type_index = NULL;
    result->type_names = NULL;
}

static bool ref_object_equal(const OspreyDecodedObject *a,
                             const OspreyDecodedObject *b)
{
    return ref_chunk_equal(&a->chunk, &b->chunk) &&
           a->storage_role == b->storage_role &&
           a->has_pointer_target == b->has_pointer_target &&
           a->value_type_id == b->value_type_id &&
           ref_address_equal(&a->owner_base, &b->owner_base) &&
           ref_address_equal(&a->pointer_target, &b->pointer_target) &&
           ref_double_bits(a->storage_posterior) == ref_double_bits(b->storage_posterior) &&
           ref_double_bits(a->pointer_posterior) == ref_double_bits(b->pointer_posterior) &&
           a->storage_support == b->storage_support &&
           a->storage_source_rule_bits == b->storage_source_rule_bits &&
           a->pointer_support == b->pointer_support &&
           a->pointer_source_rule_bits == b->pointer_source_rule_bits;
}

static bool ref_type_equal(const OspreyDecodedType *a,
                           const OspreyDecodedType *b)
{
    return a->id == b->id && a->kind == b->kind &&
           a->target_is_void == b->target_is_void && a->size == b->size &&
           a->element_count == b->element_count &&
           a->element_size == b->element_size &&
           a->element_type_id == b->element_type_id &&
           a->target_type_id == b->target_type_id &&
           ref_address_equal(&a->canonical_base, &b->canonical_base) &&
           a->field_begin == b->field_begin && a->field_count == b->field_count &&
           a->evidence_valid == b->evidence_valid &&
           ref_double_bits(a->posterior) == ref_double_bits(b->posterior) &&
           a->direct_support == b->direct_support &&
           a->source_rule_bits == b->source_rule_bits;
}

static bool ref_field_equal(const OspreyDecodedField *a,
                            const OspreyDecodedField *b)
{
    return ref_chunk_equal(&a->chunk, &b->chunk) &&
           a->relative_offset == b->relative_offset &&
           a->value_type_id == b->value_type_id &&
           ref_double_bits(a->posterior) == ref_double_bits(b->posterior) &&
           a->support == b->support && a->source_rule_bits == b->source_rule_bits;
}

static bool ref_index_equal(const OspreyModelIndexEntry *a,
                            const OspreyModelIndexEntry *b)
{
    return memcmp(&a->key, &b->key, sizeof(a->key)) == 0 &&
           a->ordinal == b->ordinal;
}

bool osprey_reference_model_equal(const OspreyReferenceResult *reference,
                                  const OspreyModel *model)
{
    if (reference == NULL || model == NULL || reference->status != OSPREY_OK ||
        model->version != OSPREY_MODEL_VERSION ||
        reference->object_count != model->object_count ||
        reference->type_count != model->type_count ||
        reference->field_count != model->field_count ||
        reference->chunk_index_count != model->chunk_index_count ||
        reference->aggregate_index_count != model->aggregate_index_count ||
        reference->type_index_count != model->type_index_count) return false;
    for (uint32_t i = 0; i < reference->object_count; i++) {
        if (!ref_object_equal(&reference->objects[i], &model->objects[i])) return false;
    }
    for (uint32_t i = 0; i < reference->type_count; i++) {
        if (!ref_type_equal(&reference->types[i], &model->types[i]) ||
            !ref_string_equal(reference->type_names[i], model->type_names[i])) return false;
    }
    for (uint32_t i = 0; i < reference->field_count; i++) {
        if (!ref_field_equal(&reference->fields[i], &model->fields[i])) return false;
    }
    for (uint32_t i = 0; i < reference->chunk_index_count; i++) {
        if (!ref_index_equal(&reference->chunk_index[i], &model->chunk_index[i])) return false;
    }
    for (uint32_t i = 0; i < reference->aggregate_index_count; i++) {
        if (!ref_index_equal(&reference->aggregate_index[i], &model->aggregate_index[i])) return false;
    }
    for (uint32_t i = 0; i < reference->type_index_count; i++) {
        if (!ref_index_equal(&reference->type_index[i], &model->type_index[i])) return false;
    }
    return true;
}

static const char *ref_type_kind_name(uint8_t kind)
{
    switch (kind) {
    case OSPREY_TYPE_PRIMITIVE: return "primitive";
    case OSPREY_TYPE_POINTER: return "pointer";
    case OSPREY_TYPE_ARRAY: return "array";
    case OSPREY_TYPE_STRUCT: return "struct";
    default: return "invalid";
    }
}

static const char *ref_role_name(uint8_t role)
{
    switch (role) {
    case OSPREY_STORAGE_PRIMITIVE: return "primitive";
    case OSPREY_STORAGE_SCALAR: return "scalar";
    case OSPREY_STORAGE_FIELD: return "field";
    case OSPREY_STORAGE_ARRAY_ELEMENT: return "array-element";
    default: return "invalid";
    }
}

static bool ref_dump_address(FILE *out, const char *label,
                             const OspreyAddress *address)
{
    return out != NULL && label != NULL && address != NULL &&
           fprintf(out, "[%s-region %u] [%s-image 0x%016" PRIx64 "] "
                   "[%s-site 0x%016" PRIx64 "] [%s-offset %" PRId64 "]",
                   label, address->region.kind, label,
                   address->region.code_image_id, label,
                   address->region.site_offset, label, address->offset) >= 0;
}

bool osprey_reference_model_dump_file(const OspreyReferenceResult *reference,
                                      FILE *out)
{
    if (reference == NULL || out == NULL || reference->status != OSPREY_OK ||
        (reference->object_count != 0 && reference->objects == NULL) ||
        (reference->type_count != 0 &&
         (reference->types == NULL || reference->type_names == NULL)) ||
        (reference->field_count != 0 && reference->fields == NULL)) return false;
    if (fprintf(out, "[model-version %u] [objects %u] [types %u] [fields %u]\n",
                OSPREY_MODEL_VERSION, reference->object_count,
                reference->type_count, reference->field_count) < 0) return false;
    for (uint32_t i = 0; i < reference->type_count; i++) {
        const OspreyDecodedType *type = &reference->types[i];
        if (fprintf(out, "[type] [id %u] [kind %s] [name %s] [size %" PRIu64
                    "] [count %" PRIu64 "] [element-size %" PRIu64
                    "] [element-type %u] [target-void %u] [target-type %u] "
                    "[field-begin %u] [field-count %u] [evidence %u] "
                    "[posterior-bits 0x%016" PRIx64 "] [support %" PRIu64
                    "] [rules 0x%016" PRIx64 "] ", i,
                    ref_type_kind_name(type->kind), reference->type_names[i],
                    type->size, type->element_count, type->element_size,
                    type->element_type_id, type->target_is_void,
                    type->target_type_id, type->field_begin, type->field_count,
                    type->evidence_valid, ref_double_bits(type->posterior),
                    type->direct_support, type->source_rule_bits) < 0 ||
            !ref_dump_address(out, "base", &type->canonical_base) ||
            fputc('\n', out) == EOF) return false;
    }
    for (uint32_t i = 0; i < reference->field_count; i++) {
        const OspreyDecodedField *field = &reference->fields[i];
        OspreyAddress owner;
        memset(&owner, 0, sizeof(owner));
        for (uint32_t j = 0; j < reference->type_count; j++) {
            const OspreyDecodedType *type = &reference->types[j];
            if (type->kind == OSPREY_TYPE_STRUCT &&
                i >= type->field_begin &&
                i - type->field_begin < type->field_count) {
                owner = type->canonical_base;
                break;
            }
        }
        if (fprintf(out, "[field] [id %u] [owner ", i) < 0 ||
            !ref_dump_address(out, "owner", &owner) ||
            fputs("] [relative ", out) == EOF ||
            fprintf(out, "%" PRIu64 "] [value-type %u] "
                    "[posterior-bits 0x%016" PRIx64 "] [support %" PRIu64
                    "] [rules 0x%016" PRIx64 "] ", field->relative_offset,
                    field->value_type_id, ref_double_bits(field->posterior),
                    field->support, field->source_rule_bits) < 0 ||
            !ref_dump_address(out, "chunk", &field->chunk.address) ||
            fprintf(out, " [size %" PRIu64 "]\n", field->chunk.size) < 0) {
            return false;
        }
    }
    for (uint32_t i = 0; i < reference->type_count; i++) {
        const OspreyDecodedType *array = &reference->types[i];
        int64_t array_end;
        if (array->kind != OSPREY_TYPE_ARRAY) continue;
        if (array->size > (uint64_t)INT64_MAX ||
            !ref_add_i64(array->canonical_base.offset,
                         (int64_t)array->size, &array_end)) return false;
        if (fprintf(out, "[array] [id %u] [lo %" PRId64 "] [hi %" PRId64
                    "] [size %" PRIu64 "] [stride %" PRIu64 "] [count %" PRIu64
                    "] [element-type %u] [posterior-bits 0x%016" PRIx64
                    "] [support %" PRIu64 "] [rules 0x%016" PRIx64 "] [base ",
                    i, array->canonical_base.offset, array_end, array->size,
                    array->element_size, array->element_count,
                    array->element_type_id, ref_double_bits(array->posterior),
                    array->direct_support, array->source_rule_bits) < 0 ||
            !ref_dump_address(out, "base", &array->canonical_base) ||
            fputs("]\n", out) == EOF) return false;
    }
    for (uint32_t i = 0; i < reference->object_count; i++) {
        const OspreyDecodedObject *object = &reference->objects[i];
        if (fprintf(out, "[object] [id %u] [role %s] [value-type %u] "
                    "[pointer %u] [storage-posterior-bits 0x%016" PRIx64
                    "] [storage-support %" PRIu64 "] [storage-rules 0x%016"
                    PRIx64 "] [pointer-posterior-bits 0x%016" PRIx64
                    "] [pointer-support %" PRIu64 "] [pointer-rules 0x%016"
                    PRIx64 "] ", i, ref_role_name(object->storage_role),
                    object->value_type_id, object->has_pointer_target,
                    ref_double_bits(object->storage_posterior),
                    object->storage_support, object->storage_source_rule_bits,
                    ref_double_bits(object->pointer_posterior),
                    object->pointer_support, object->pointer_source_rule_bits) < 0 ||
            !ref_dump_address(out, "chunk", &object->chunk.address) ||
            fprintf(out, " [size %" PRIu64 "] [owner ", object->chunk.size) < 0 ||
            !ref_dump_address(out, "owner", &object->owner_base) ||
            fputs("]", out) == EOF) return false;
        if (object->has_pointer_target &&
            (fputs(" [target ", out) == EOF ||
             !ref_dump_address(out, "target", &object->pointer_target) ||
             fputs("]", out) == EOF)) return false;
        if (fputc('\n', out) == EOF) return false;
    }
    return ferror(out) == 0;
}
