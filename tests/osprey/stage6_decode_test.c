#include "osprey.h"
#include "osprey-internal.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned failures;
static unsigned registered;
static unsigned executed;

#define CHECK(condition, message) do {                                      \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__);     \
        failures++;                                                          \
    }                                                                            \
} while (0)

#define RUN(test) do {                                                       \
    registered++;                                                            \
    test();                                                                  \
    executed++;                                                              \
} while (0)

static OspreyConfig decode_config(void)
{
    OspreyConfig config;
    memset(&config, 0, sizeof(config));
    config.enabled = true;
    config.shared_bytes = 1u << 20;
    config.max_facts = 1024;
    config.max_chunks_per_region = 128;
    config.max_candidates_per_kind_region = 4096;
    config.max_variables = 1024;
    config.max_factors = 4096;
    config.max_exact_clique_vars = 20;
    config.max_exact_table_bytes = 1u << 20;
    config.max_bp_table_bytes = 1u << 20;
    config.report_threshold = 0.6;
    return config;
}

static OspreyRegionId make_region(OspreyRegionKind kind, uint64_t image,
                                  uint64_t site)
{
    OspreyRegionId region;
    memset(&region, 0, sizeof(region));
    region.kind = kind;
    region.code_image_id = image;
    region.site_offset = site;
    return region;
}

static OspreyAddress make_address(OspreyRegionId region, int64_t offset)
{
    OspreyAddress address;
    memset(&address, 0, sizeof(address));
    address.region = region;
    address.offset = offset;
    return address;
}

static OspreyChunk make_chunk(OspreyRegionId region, int64_t offset,
                              uint64_t size)
{
    OspreyChunk chunk;
    memset(&chunk, 0, sizeof(chunk));
    chunk.address = make_address(region, offset);
    chunk.size = size;
    return chunk;
}

static OspreyContext *new_decode_context(void)
{
    OspreyConfig config = decode_config();
    OspreyContext *ctx = osprey_new(&config);
    if (ctx != NULL) {
        ctx->graph = osprey_graph_new();
        if (ctx->graph != NULL) ctx->graph->extents_built = true;
    }
    return ctx;
}

static void add_extent(OspreyContext *ctx, OspreyRegionId region,
                       int64_t lo, int64_t hi)
{
    OspreyRegionExtent extent;
    memset(&extent, 0, sizeof(extent));
    extent.region = region;
    extent.lo = lo;
    extent.hi = hi;
    g_array_append_val(ctx->graph->extents, extent);
}

static uint32_t add_payload(OspreyContext *ctx, uint8_t kind,
                            const OspreyVarPayload *payload)
{
    OspreyInternResult result = osprey_intern_var(ctx, kind, payload);
    CHECK(result.id != UINT32_MAX, "decoder fixture variable inserted");
    return result.id;
}

static uint32_t add_chunk_var(OspreyContext *ctx, uint8_t kind,
                              OspreyChunk chunk)
{
    OspreyVarPayload payload;
    memset(&payload, 0, sizeof(payload));
    payload.chunk = chunk;
    return add_payload(ctx, kind, &payload);
}

static void set_belief(OspreyContext *ctx, uint32_t id, double belief)
{
    OspreyVar *variable = &g_array_index(ctx->graph->vars, OspreyVar, id);
    variable->belief = belief;
    variable->belief_valid = 1;
}

static void set_all_beliefs(OspreyContext *ctx, double belief)
{
    for (guint i = 0; i < ctx->graph->vars->len; i++) {
        set_belief(ctx, i, belief);
    }
}

static char *dump_input(const OspreyDecodeInput *input)
{
    char *data = NULL;
    size_t length = 0;
    FILE *out = open_memstream(&data, &length);
    if (out == NULL) return NULL;
    if (!osprey_decode_input_dump_file(input, out) || fclose(out) != 0) {
        free(data);
        return NULL;
    }
    return data;
}

static bool build_input(OspreyContext *ctx, OspreyDecodeInput **out)
{
    OspreyStatus status = osprey_decode_input_build(ctx, out);
    CHECK(status == OSPREY_OK, "valid decoder input builds");
    return status == OSPREY_OK;
}

static OspreyContext *make_projection_context(unsigned order)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId global = make_region(OSPREY_REGION_GLOBAL, 0x101, 0x500);
    OspreyRegionId heap = make_region(OSPREY_REGION_HEAP_SITE, 0x202, 0x120);
    OspreyRegionId stack = make_region(OSPREY_REGION_STACK_FUNCTION,
                                       0x303, 0x80);
    OspreyChunk primitive = make_chunk(global, 32, 8);
    OspreyChunk scalar = make_chunk(global, 0, 4);
    OspreyChunk field_a = make_chunk(heap, 8, 8);
    OspreyChunk field_b = make_chunk(heap, 24, 4);
    OspreyChunk pointer = make_chunk(global, 96, sizeof(target_ulong));
    OspreyVarPayload payloads[12];
    uint8_t kinds[12];
    size_t count = 0;

    memset(payloads, 0, sizeof(payloads));
    kinds[count] = OSPREY_PRED_PRIMITIVE_VAR;
    payloads[count++].chunk = primitive;
    kinds[count] = OSPREY_PRED_SCALAR;
    payloads[count++].chunk = scalar;
    kinds[count] = OSPREY_PRED_ARRAY;
    payloads[count].segment.a1 = make_address(global, 0);
    payloads[count].segment.a2 = make_address(global, 18);
    payloads[count++].segment.size = 8; /* retained: graph-valid, non-divisible */
    kinds[count] = OSPREY_PRED_FIELD_OF;
    payloads[count].attached.chunk = field_a;
    payloads[count++].attached.base = make_address(heap, 0);
    kinds[count] = OSPREY_PRED_FIELD_OF;
    payloads[count].attached.chunk = field_b;
    payloads[count++].attached.base = make_address(heap, 0);
    kinds[count] = OSPREY_PRED_FIELD_OF;
    payloads[count].attached.chunk = make_chunk(stack, -8, 8);
    payloads[count++].attached.base = make_address(stack, -16);
    kinds[count] = OSPREY_PRED_POINTER;
    payloads[count].attached.chunk = pointer;
    payloads[count++].attached.base = make_address(heap, 0);

    /* P02-P06 are complete graph inputs, but not decoder families. */
    kinds[count] = OSPREY_PRED_PRIMITIVE_ACCESS;
    payloads[count].prim_access.chunk = primitive;
    payloads[count++].prim_access.insn_pc = 0x44;
    kinds[count] = OSPREY_PRED_UNFOLDABLE_HEAP;
    payloads[count].heap_fold.region = heap;
    payloads[count++].heap_fold.size = 16;
    kinds[count] = OSPREY_PRED_FOLDABLE_HEAP;
    payloads[count].heap_fold.region = heap;
    payloads[count++].heap_fold.size = 0;
    kinds[count] = OSPREY_PRED_HOMO_SEGMENT;
    payloads[count].segment.a1 = make_address(global, 0);
    payloads[count].segment.a2 = make_address(heap, 0);
    payloads[count++].segment.size = 8;
    kinds[count] = OSPREY_PRED_ARRAY_START;
    payloads[count++].addr = make_address(global, 0);

    CHECK(ctx != NULL, "projection context allocated");
    if (ctx == NULL) return NULL;
    add_extent(ctx, stack, -32, 32);
    add_extent(ctx, global, 0, 128);
    add_extent(ctx, heap, 0, 32);
    if (order == 1) {
        for (size_t i = count; i-- > 0;) add_payload(ctx, kinds[i], &payloads[i]);
    } else if (order == 2) {
        static const uint8_t permutation[] = {
            5, 0, 11, 3, 8, 1, 10, 4, 6, 2, 9, 7,
        };
        CHECK(count == G_N_ELEMENTS(permutation),
              "shuffled projection covers every source row");
        for (size_t i = 0; i < count; i++) {
            size_t source = permutation[i];
            add_payload(ctx, kinds[source], &payloads[source]);
        }
    } else {
        for (size_t i = 0; i < count; i++) add_payload(ctx, kinds[i], &payloads[i]);
    }
    set_all_beliefs(ctx, 0.8);
    for (guint i = 0; i < ctx->graph->vars->len; i++) {
        OspreyVar *variable = &g_array_index(ctx->graph->vars, OspreyVar, i);
        variable->direct_support = 3;
        variable->source_rule_bits = UINT64_C(1) << variable->kind;
    }
    return ctx;
}

static void test_valid_projection_and_indexes(void)
{
    OspreyContext *ctx = make_projection_context(false);
    OspreyDecodeInput *input = NULL;
    CHECK(ctx != NULL && build_input(ctx, &input),
          "projection fixture builds canonical input");
    if (input != NULL) {
        CHECK(input->primitive_count == 1 && input->scalar_count == 1 &&
                  input->array_count == 1 && input->field_count == 3 &&
                  input->pointer_count == 1,
              "all projected families retain eligible candidates");
        CHECK(input->discarded_hard_false == 0 &&
                  input->discarded_threshold == 0,
              "valid projection has no discarded candidates");
        CHECK(input->extent_count == 3 && input->extents[0].region.kind ==
                  OSPREY_REGION_GLOBAL,
              "extents are owned and canonically sorted");
        CHECK(input->chunk_range_count == 6 &&
                  input->field_base_range_count == 2 &&
                  input->array_region_range_count == 1 &&
                  input->array_region_stride_range_count == 1,
              "complete chunk/base/region/stride views are built");
        CHECK(input->array_candidates[0].payload.segment.a2.offset == 18,
              "non-divisible graph-valid array reaches Stage 6.3");
        CHECK(input->field_base_ranges[0].count == 2 &&
                  input->field_base_ranges[1].count == 1,
              "base view stores non-contiguous field ordinals as ranges");
        char *dump = dump_input(input);
        CHECK(dump != NULL && strstr(dump, "[posterior-bits") != NULL &&
                  strstr(dump, "[source-rules") != NULL &&
                  strstr(dump, "[extent]") != NULL,
              "input dump contains exact evidence and extents");
        free(dump);
        osprey_decode_input_free(input);
    }
    osprey_free(ctx);
}

static void test_fixed_canonical_dump(void)
{
    static const char expected[] =
        "[discarded-hard-false 0] [discarded-threshold 0]\n"
        "[primitive] [kind 1] [key 0x0000000000564152"
        " 0x0000000000000001 0x0000000000000000"
        " 0x0000000000000001 0x0000000000000002"
        " 0xfffffffffffffff8 0x0000000000000004"
        " 0x0000000000000000 0x0000000000000000"
        " 0x0000000000000000 0x0000000000000000]"
        " [posterior-bits 0x3fe8000000000000] [support 5]"
        " [source-rules 0x0000000000000002]\n"
        "[extent] [region 0] [image 0x0000000000000001]"
        " [site 0x0000000000000002] [lo -16] [hi 0]\n"
        "[chunk-range] [key 0x0000000000564152"
        " 0x0000000000000001 0x0000000000000000"
        " 0x0000000000000001 0x0000000000000002"
        " 0xfffffffffffffff8 0x0000000000000004"
        " 0x0000000000000000 0x0000000000000000"
        " 0x0000000000000000 0x0000000000000000]\n";
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    uint32_t id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                make_chunk(region, -8, 4));
    set_belief(ctx, id, 0.75);
    OspreyVar *variable = &g_array_index(ctx->graph->vars, OspreyVar, id);
    variable->direct_support = 5;
    variable->source_rule_bits = 2;
    add_extent(ctx, region, -16, 0);
    OspreyDecodeInput *input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL,
          "fixed canonical dump fixture builds");
    char *dump = input == NULL ? NULL : dump_input(input);
    CHECK(dump != NULL && strcmp(dump, expected) == 0,
          "canonical dump matches the fixed complete-key record");
    free(dump);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_region_identity_collisions(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId regions[] = {
        make_region(OSPREY_REGION_STACK_FUNCTION, 1, 2),
        make_region(OSPREY_REGION_STACK_FUNCTION, 1, 3),
        make_region(OSPREY_REGION_STACK_FUNCTION, 2, 2),
    };
    for (size_t i = 0; i < G_N_ELEMENTS(regions); i++) {
        uint32_t id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                    make_chunk(regions[i], -16, 8));
        set_belief(ctx, id, 0.9);
        add_extent(ctx, regions[i], -32, 0);
    }
    OspreyDecodeInput *input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL &&
              input->primitive_count == 3 && input->chunk_range_count == 3 &&
              input->extent_count == 3,
          "negative offsets and image/site collisions retain full region identity");
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_projection_permutation(void)
{
    OspreyContext *left = make_projection_context(0);
    OspreyContext *right = make_projection_context(1);
    OspreyContext *shuffled = make_projection_context(2);
    OspreyDecodeInput *a = NULL;
    OspreyDecodeInput *b = NULL;
    OspreyDecodeInput *c = NULL;
    char *dump_a = NULL;
    char *dump_b = NULL;
    char *dump_c = NULL;
    CHECK(left != NULL && right != NULL && shuffled != NULL &&
              build_input(left, &a) && build_input(right, &b) &&
              build_input(shuffled, &c),
          "forward, reverse, and shuffled graph insertions build");
    if (a != NULL && b != NULL && c != NULL) {
        dump_a = dump_input(a);
        dump_b = dump_input(b);
        dump_c = dump_input(c);
        CHECK(dump_a != NULL && dump_b != NULL && dump_c != NULL &&
                  strcmp(dump_a, dump_b) == 0 &&
                  strcmp(dump_a, dump_c) == 0,
              "canonical input dump ignores graph insertion order");
    }
    free(dump_a);
    free(dump_b);
    free(dump_c);
    osprey_decode_input_free(a);
    osprey_decode_input_free(b);
    osprey_decode_input_free(c);
    osprey_free(left);
    osprey_free(right);
    osprey_free(shuffled);
}

static void test_filter_threshold_and_hard_false(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 0x11, 0x22);
    OspreyChunk low = make_chunk(region, 0, 8);
    OspreyChunk equal = make_chunk(region, 8, 8);
    OspreyVarPayload array_payload;
    uint32_t low_id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, low);
    uint32_t equal_id = add_chunk_var(ctx, OSPREY_PRED_SCALAR, equal);
    memset(&array_payload, 0, sizeof(array_payload));
    array_payload.segment.a1 = make_address(region, 16);
    array_payload.segment.a2 = make_address(region, 24);
    array_payload.segment.size = 8;
    uint32_t array_id = add_payload(ctx, OSPREY_PRED_ARRAY, &array_payload);
    set_belief(ctx, low_id, 0.5999999999999999);
    set_belief(ctx, equal_id, 0.6);
    set_belief(ctx, array_id, 1.0);
    OspreyFactorResult hard = osprey_factor_add_hard_false(
        ctx, OSPREY_RULE_CB06, OSPREY_GRAPH_SECONDARY, array_id);
    CHECK(hard.status == OSPREY_OK, "hard-false factor inserted");
    g_array_index(ctx->graph->vars, OspreyVar, array_id).hard_false = 1;
    OspreyDecodeInput *input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL,
          "threshold fixture builds");
    if (input != NULL) {
        CHECK(input->scalar_count == 1 && input->array_count == 0 &&
                  input->primitive_count == 0 && input->discarded_hard_false == 1 &&
                  input->discarded_threshold == 1,
              "hard-false precedes threshold and equality is retained");
        osprey_decode_input_free(input);
    }
    osprey_free(ctx);
}

static void test_graph_boundary_rejections(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    OspreyChunk chunk = make_chunk(region, 0, 8);
    uint32_t id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, chunk);
    set_belief(ctx, id, 0.9);
    add_extent(ctx, region, 0, 8);
    OspreyDecodeInput *input = NULL;

    g_array_free(ctx->graph->factors, TRUE);
    ctx->graph->factors = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "missing factor storage is rejected");
    osprey_free(ctx);

    ctx = new_decode_context();
    id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, chunk);
    set_belief(ctx, id, 0.9);
    add_extent(ctx, region, 0, 8);
    OspreyKey key = osprey_var_key(OSPREY_PRED_PRIMITIVE_VAR,
                                   &(OspreyVarPayload){ .chunk = chunk });
    g_hash_table_remove(ctx->graph->var_index, &key);
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "variable equality index mismatch is rejected");
    osprey_free(ctx);

    ctx = new_decode_context();
    id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, chunk);
    set_belief(ctx, id, 0.9);
    add_extent(ctx, region, 0, 8);
    g_array_index(ctx->graph->vars, OspreyVar, id).belief_valid = 0;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "missing exact belief validity is rejected");
    osprey_free(ctx);

    ctx = new_decode_context();
    id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, chunk);
    set_belief(ctx, id, NAN);
    add_extent(ctx, region, 0, 8);
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "NaN belief is rejected");
    osprey_free(ctx);
}

static void test_exact_belief_boundaries(void)
{
    static const double beliefs[] = {
        0.0, DBL_MIN, 0.6, 1.0,
    };
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 7, 9);
    ctx->config.report_threshold = 0.0;
    for (size_t i = 0; i < G_N_ELEMENTS(beliefs); i++) {
        uint32_t id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                    make_chunk(region, (int64_t)i * 8, 8));
        set_belief(ctx, id, beliefs[i]);
    }
    uint32_t below_id = add_chunk_var(ctx, OSPREY_PRED_SCALAR,
                                      make_chunk(region, 32, 8));
    uint32_t above_id = add_chunk_var(ctx, OSPREY_PRED_SCALAR,
                                      make_chunk(region, 40, 8));
    set_belief(ctx, below_id, nextafter(0.6, 0.0));
    set_belief(ctx, above_id, nextafter(0.6, 1.0));
    add_extent(ctx, region, 0, 48);
    OspreyDecodeInput *input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL,
          "exact finite belief boundaries build without clamping");
    if (input != NULL) {
        CHECK(input->primitive_count == G_N_ELEMENTS(beliefs) &&
                  input->scalar_count == 2,
              "zero, DBL_MIN, threshold-adjacent, and one beliefs survive at zero threshold");
        uint64_t zero_bits = UINT64_MAX;
        uint64_t one_bits = 0;
        memcpy(&zero_bits, &input->primitive_candidates[0].posterior,
               sizeof(zero_bits));
        memcpy(&one_bits, &input->primitive_candidates[3].posterior,
               sizeof(one_bits));
        CHECK(input->primitive_candidates[0].posterior_bits == zero_bits &&
                  input->primitive_candidates[3].posterior_bits == one_bits,
              "candidate records preserve exact posterior bits");
        osprey_decode_input_free(input);
    }
    osprey_free(ctx);
}

static void test_invalid_belief_values(void)
{
    const double invalid[] = {
        NAN, INFINITY, -INFINITY, -DBL_MIN, nextafter(1.0, INFINITY),
    };
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    for (size_t i = 0; i < G_N_ELEMENTS(invalid); i++) {
        OspreyContext *ctx = new_decode_context();
        uint32_t id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                    make_chunk(region, 0, 8));
        set_belief(ctx, id, invalid[i]);
        add_extent(ctx, region, 0, 8);
        OspreyDecodeInput *input = (OspreyDecodeInput *)(uintptr_t)1;
        CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
                  input == NULL,
              "non-finite and out-of-range beliefs are rejected");
        osprey_free(ctx);
    }
}

static void test_variable_identity_rejections(void)
{
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    OspreyChunk chunk = make_chunk(region, 0, 8);
    OspreyDecodeInput *input = NULL;

    OspreyContext *ctx = new_decode_context();
    uint32_t id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, chunk);
    set_belief(ctx, id, 0.9);
    add_extent(ctx, region, 0, 8);
    g_array_index(ctx->graph->vars, OspreyVar, id).id = id + 1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "mismatched variable ordinal is rejected");
    osprey_free(ctx);

    ctx = new_decode_context();
    id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, chunk);
    set_belief(ctx, id, 0.9);
    add_extent(ctx, region, 0, 8);
    OspreyVar duplicate = g_array_index(ctx->graph->vars, OspreyVar, id);
    duplicate.id = id + 1;
    g_array_append_val(ctx->graph->vars, duplicate);
    input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "duplicate semantic variable identity is rejected");
    osprey_free(ctx);
}

static void test_hard_false_contract_rejections(void)
{
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    OspreyVarPayload payload;
    memset(&payload, 0, sizeof(payload));
    payload.segment.a1 = make_address(region, 0);
    payload.segment.a2 = make_address(region, 8);
    payload.segment.size = 8;
    OspreyDecodeInput *input = NULL;

    OspreyContext *ctx = new_decode_context();
    uint32_t array_id = add_payload(ctx, OSPREY_PRED_ARRAY, &payload);
    set_belief(ctx, array_id, 0.9);
    add_extent(ctx, region, 0, 8);
    g_array_index(ctx->graph->vars, OspreyVar, array_id).hard_false = 1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL,
          "hard-false bit without a factor is rejected");
    osprey_free(ctx);

    ctx = new_decode_context();
    array_id = add_payload(ctx, OSPREY_PRED_ARRAY, &payload);
    set_belief(ctx, array_id, 0.9);
    add_extent(ctx, region, 0, 8);
    CHECK(osprey_factor_add_hard_false(ctx, OSPREY_RULE_CB06,
                                      OSPREY_GRAPH_SECONDARY,
                                      array_id).status == OSPREY_OK,
          "hard-false mismatch fixture factor builds");
    input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL,
          "hard-false factor without the bit is rejected");
    osprey_free(ctx);

    ctx = new_decode_context();
    array_id = add_payload(ctx, OSPREY_PRED_ARRAY, &payload);
    set_belief(ctx, array_id, 0.9);
    add_extent(ctx, region, 0, 8);
    CHECK(osprey_factor_add_hard_false(ctx, OSPREY_RULE_CB06,
                                      OSPREY_GRAPH_SECONDARY,
                                      array_id).status == OSPREY_OK,
          "duplicate hard-false fixture factor builds");
    g_array_index(ctx->graph->vars, OspreyVar, array_id).hard_false = 1;
    OspreyFactor *original = g_array_index(ctx->graph->factors,
                                           OspreyFactor *, 0);
    OspreyFactor *duplicate = g_new0(OspreyFactor, 1);
    *duplicate = *original;
    duplicate->id = 1;
    duplicate->var_ids = g_new(uint32_t, original->num_vars);
    memcpy(duplicate->var_ids, original->var_ids,
           original->num_vars * sizeof(original->var_ids[0]));
    g_array_append_val(ctx->graph->factors, duplicate);
    input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL,
          "duplicate hard-false factor is rejected");
    osprey_free(ctx);

    ctx = new_decode_context();
    uint32_t scalar_id = add_chunk_var(ctx, OSPREY_PRED_SCALAR,
                                       make_chunk(region, 0, 8));
    set_belief(ctx, scalar_id, 0.9);
    add_extent(ctx, region, 0, 8);
    CHECK(osprey_factor_add_hard_false(ctx, OSPREY_RULE_CB06,
                                      OSPREY_GRAPH_SECONDARY,
                                      scalar_id).status == OSPREY_OK,
          "wrong-kind hard-false fixture factor builds");
    input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL,
          "hard-false on a non-array variable is rejected");
    osprey_free(ctx);

    ctx = new_decode_context();
    uint32_t ids[2];
    ids[0] = add_payload(ctx, OSPREY_PRED_ARRAY, &payload);
    ids[1] = add_chunk_var(ctx, OSPREY_PRED_SCALAR,
                           make_chunk(region, 0, 8));
    set_all_beliefs(ctx, 0.9);
    add_extent(ctx, region, 0, 8);
    CHECK(osprey_factor_add_ex(ctx, OSPREY_RULE_CB06,
                               OSPREY_GRAPH_SECONDARY,
                               OSPREY_POTENTIAL_IMPLICATION, 1, false,
                               0.8, ids, 2).status == OSPREY_OK,
          "malformed CB06 implication fixture builds");
    input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL,
          "non-hard-false CB06 factor is rejected");
    osprey_free(ctx);
}

static void test_payload_and_extent_rejections(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    OspreyChunk chunk = make_chunk(region, 0, 8);
    uint32_t id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, chunk);
    set_belief(ctx, id, 0.9);
    add_extent(ctx, region, 8, 0);
    OspreyDecodeInput *input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "malformed extent bounds are rejected");
    osprey_free(ctx);

    ctx = new_decode_context();
    id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, chunk);
    set_belief(ctx, id, 0.9);
    add_extent(ctx, region, 0, 8);
    OspreyVarPayload bad;
    memset(&bad, 0, sizeof(bad));
    bad.segment.a1 = make_address(region, 0);
    bad.segment.a2 = make_address(region, 8);
    bad.segment.size = 8;
    OspreyInternResult array = osprey_intern_var(ctx, OSPREY_PRED_ARRAY,
                                                  &bad);
    CHECK(array.id != UINT32_MAX, "payload fixture array inserted");
    set_belief(ctx, array.id, 0.9);
    g_array_index(ctx->graph->vars, OspreyVar, array.id).payload.segment.a2.offset = 0;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "malformed final payload is rejected before projection");
    osprey_free(ctx);

    memset(&bad, 0, sizeof(bad));
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_POINTER, &bad),
          "incomplete attached pointer payload is invalid");
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_PRIMITIVE_VAR, &bad) &&
              !osprey_var_payload_valid(OSPREY_PRED_SCALAR, &bad),
          "zero-width primitive and scalar chunks are invalid");

    bad.prim_access.chunk = make_chunk(region, 0, 0);
    bad.prim_access.insn_pc = 1;
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_PRIMITIVE_ACCESS, &bad),
          "zero-width primitive-access chunk is invalid");

    memset(&bad, 0, sizeof(bad));
    bad.heap_fold.region = region;
    bad.heap_fold.size = 8;
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_UNFOLDABLE_HEAP, &bad) &&
              !osprey_var_payload_valid(OSPREY_PRED_FOLDABLE_HEAP, &bad),
          "heap predicates require a heap-site region");

    memset(&bad, 0, sizeof(bad));
    bad.segment.a1 = make_address(region, 8);
    bad.segment.a2 = make_address(region, 0);
    bad.segment.size = 8;
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_HOMO_SEGMENT, &bad),
          "final homomorphic endpoints must be canonical");
    bad.segment.a1 = make_address(region, 0);
    bad.segment.a2 = make_address(region, 8);
    bad.segment.size = 0;
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_HOMO_SEGMENT, &bad),
          "zero-size homomorphic segment is invalid");

    memset(&bad, 0, sizeof(bad));
    bad.addr = make_address((OspreyRegionId){ .kind = (OspreyRegionKind)99 },
                            0);
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_ARRAY_START, &bad),
          "invalid array-start region kind is rejected");

    memset(&bad, 0, sizeof(bad));
    bad.segment.a1 = make_address(region, 0);
    bad.segment.a2 = make_address(make_region(OSPREY_REGION_GLOBAL, 1, 3), 8);
    bad.segment.size = 8;
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_ARRAY, &bad),
          "array endpoints require complete equal region identity");
    bad.segment.a2 = make_address(region, 8);
    bad.segment.size = 0;
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_ARRAY, &bad),
          "zero-stride array is invalid");
    bad.segment.size = 16;
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_ARRAY, &bad),
          "array span shorter than stride is invalid");

    memset(&bad, 0, sizeof(bad));
    bad.attached.chunk = make_chunk(region, 0, 8);
    bad.attached.base = make_address(make_region(OSPREY_REGION_HEAP_SITE, 0, 4),
                                     0);
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_FIELD_OF, &bad),
          "field chunk and base require one complete region identity");
    bad.attached.base = make_address(region, 8);
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_FIELD_OF, &bad),
          "field base cannot follow its field chunk");
}

static void test_extent_catalog_normalization(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId first = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    OspreyRegionId second = make_region(OSPREY_REGION_HEAP_SITE, 3, 4);
    add_extent(ctx, second, -8, 24);
    add_extent(ctx, first, 0, 16);
    OspreyDecodeInput *input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL &&
              input->extent_count == 2 &&
              input->extents[0].region.kind == OSPREY_REGION_GLOBAL &&
              input->extents[1].region.kind == OSPREY_REGION_HEAP_SITE,
          "unsorted extents are copied into complete canonical order");
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    add_extent(ctx, first, 0, 8);
    add_extent(ctx, first, 0, 16);
    input = (OspreyDecodeInput *)(uintptr_t)1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "duplicate complete-region extents are rejected");
    osprey_free(ctx);
}

static void test_unbuilt_extent_catalog_is_rejected(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    uint32_t id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                make_chunk(region, 0, 8));
    set_belief(ctx, id, 0.9);
    add_extent(ctx, region, 0, 8);
    ctx->graph->extents_built = false;
    OspreyDecodeInput *input = (OspreyDecodeInput *)(uintptr_t)1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "an unfinished extent catalog is rejected");
    osprey_free(ctx);
}

static void test_checked_payload_arithmetic(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    OspreyChunk wrapping = make_chunk(region, INT64_MAX, 1);
    uint32_t id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, wrapping);
    set_belief(ctx, id, 0.9);
    add_extent(ctx, region, INT64_MIN, INT64_MAX);
    OspreyDecodeInput *input = (OspreyDecodeInput *)(uintptr_t)1;
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_PRIMITIVE_VAR,
                                     &(OspreyVarPayload){ .chunk = wrapping }),
          "a chunk with an unrepresentable exclusive end is invalid");
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "decoder rejects unrepresentable chunk arithmetic");
    osprey_free(ctx);

    ctx = new_decode_context();
    OspreyVarPayload heap;
    memset(&heap, 0, sizeof(heap));
    heap.heap_fold.region = make_region(OSPREY_REGION_HEAP_SITE, 0, 3);
    heap.heap_fold.size = (uint64_t)INT64_MAX + 1;
    id = add_payload(ctx, OSPREY_PRED_UNFOLDABLE_HEAP, &heap);
    set_belief(ctx, id, 0.9);
    add_extent(ctx, heap.heap_fold.region, 0, INT64_MAX);
    input = (OspreyDecodeInput *)(uintptr_t)1;
    CHECK(!osprey_var_payload_valid(OSPREY_PRED_UNFOLDABLE_HEAP, &heap),
          "a heap boundary outside the signed canonical domain is invalid");
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "decoder rejects unrepresentable heap arithmetic");
    osprey_free(ctx);
}

static void test_out_of_extent_candidates_are_retained(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    OspreyVarPayload payload;
    uint32_t primitive = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                       make_chunk(region, 64, 8));
    uint32_t scalar = add_chunk_var(ctx, OSPREY_PRED_SCALAR,
                                    make_chunk(region, 72, 8));
    memset(&payload, 0, sizeof(payload));
    payload.segment.a1 = make_address(region, 80);
    payload.segment.a2 = make_address(region, 96);
    payload.segment.size = 8;
    uint32_t array = add_payload(ctx, OSPREY_PRED_ARRAY, &payload);
    memset(&payload, 0, sizeof(payload));
    payload.attached.chunk = make_chunk(region, 104, 8);
    payload.attached.base = make_address(region, 100);
    uint32_t field = add_payload(ctx, OSPREY_PRED_FIELD_OF, &payload);
    memset(&payload, 0, sizeof(payload));
    payload.attached.chunk = make_chunk(region, 112, 4);
    payload.attached.base = make_address(region, 1000);
    uint32_t pointer = add_payload(ctx, OSPREY_PRED_POINTER, &payload);
    set_belief(ctx, primitive, 0.9);
    set_belief(ctx, scalar, 0.9);
    set_belief(ctx, array, 0.9);
    set_belief(ctx, field, 0.9);
    set_belief(ctx, pointer, 0.9);
    add_extent(ctx, region, 0, 8);
    OspreyDecodeInput *input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL,
          "Stage 6.1 preserves candidates for later containment validation");
    if (input != NULL) {
        CHECK(input->primitive_count == 1 && input->scalar_count == 1 &&
                  input->array_count == 1 && input->field_count == 1 &&
                  input->pointer_count == 1 &&
                  input->pointer_candidates[0].payload.attached.chunk.size == 4,
              "out-of-extent and wrong-width pointer candidates remain unchanged");
        osprey_decode_input_free(input);
    }
    osprey_free(ctx);
}

static void test_array_region_view_key_order(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    OspreyVarPayload first;
    OspreyVarPayload second;
    memset(&first, 0, sizeof(first));
    memset(&second, 0, sizeof(second));
    first.segment.a1 = make_address(region, 0);
    first.segment.a2 = make_address(region, 16);
    first.segment.size = 8;
    second.segment.a1 = make_address(region, 8);
    second.segment.a2 = make_address(region, 16);
    second.segment.size = 4;
    uint32_t first_id = add_payload(ctx, OSPREY_PRED_ARRAY, &first);
    uint32_t second_id = add_payload(ctx, OSPREY_PRED_ARRAY, &second);
    set_belief(ctx, first_id, 0.9);
    set_belief(ctx, second_id, 0.9);
    add_extent(ctx, region, 0, 32);
    OspreyDecodeInput *input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL,
          "two-stride array fixture builds");
    if (input != NULL) {
        CHECK(input->array_count == 2 && input->array_region_range_count == 1 &&
                  input->array_region_ranges[0].count == 2 &&
                  input->array_by_region[0] == 0 &&
                  input->array_by_region[1] == 1,
              "region view uses complete candidate-key order, not stride order");
        CHECK(input->array_region_stride_range_count == 2 &&
                  input->array_by_region_stride[0] == 1 &&
                  input->array_by_region_stride[1] == 0,
              "region-stride view retains stride-first order");
        osprey_decode_input_free(input);
    }
    osprey_free(ctx);
}

static void test_rule_stage_mismatch_is_rejected(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    uint32_t ids[2];
    ids[0] = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                           make_chunk(region, 0, 8));
    ids[1] = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                           make_chunk(region, 8, 8));
    set_all_beliefs(ctx, 0.9);
    add_extent(ctx, region, 0, 16);
    OspreyFactorResult factor = osprey_factor_add_ex(
        ctx, OSPREY_RULE_CA02, OSPREY_GRAPH_SECONDARY,
        OSPREY_POTENTIAL_IMPLICATION, 1, false, 0.8, ids, 2);
    CHECK(factor.status == OSPREY_OK,
          "fixture creates a structurally valid wrong-stage factor");
    OspreyDecodeInput *input = (OspreyDecodeInput *)(uintptr_t)1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "decoder rejects a rule/factor stage mismatch");
    osprey_free(ctx);
}

static void test_transaction_is_unchanged(void)
{
    OspreyContext *ctx = make_projection_context(false);
    OspreyGraph *graph = ctx->graph;
    OspreyModel *model = ctx->model;
    OspreyModel *staged = ctx->staged_model;
    OspreyStatus tx_status = ctx->tx_status;
    OspreyStatus last_status = ctx->last_status;
    const char *stage = ctx->tx_stage;
    bool ready = ctx->tx_model_ready;
    OspreyDecodeInput *input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK,
          "transaction success fixture builds");
    CHECK(ctx->graph == graph && ctx->model == model &&
              ctx->staged_model == staged && ctx->tx_status == tx_status &&
              ctx->last_status == last_status && ctx->tx_stage == stage &&
              ctx->tx_model_ready == ready,
          "successful helper leaves transaction ownership unchanged");
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_failure_leaves_transaction_unchanged(void)
{
    OspreyContext *ctx = make_projection_context(false);
    OspreyGraph *graph = ctx->graph;
    OspreyModel *model = ctx->model;
    OspreyModel *staged = ctx->staged_model;
    ctx->tx_status = OSPREY_NON_CONVERGED;
    ctx->tx_stage = "secondary";
    ctx->tx_reason = "fixture";
    ctx->tx_model_ready = true;
    ctx->last_status = OSPREY_GRAPH_ARITHMETIC;
    ctx->last_analyze_time_ms = 77;
    ctx->last_exact_logz = 3.25;
    uint64_t total_samples = ctx->total_samples;
    uint64_t observations = ctx->total_dynamic_observations;
    OspreyDecodeInput *input = (OspreyDecodeInput *)(uintptr_t)1;
    osprey_decode_test_set_alloc_fail_after(0);
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "injected decoder allocation failure returns no input");
    osprey_decode_test_set_alloc_fail_after(-1);
    CHECK(ctx->graph == graph && ctx->model == model &&
              ctx->staged_model == staged &&
              ctx->tx_status == OSPREY_NON_CONVERGED &&
              strcmp(ctx->tx_stage, "secondary") == 0 &&
              strcmp(ctx->tx_reason, "fixture") == 0 &&
              ctx->tx_model_ready &&
              ctx->last_status == OSPREY_GRAPH_ARITHMETIC &&
              ctx->last_analyze_time_ms == 77 &&
              ctx->last_exact_logz == 3.25 &&
              ctx->total_samples == total_samples &&
              ctx->total_dynamic_observations == observations,
          "decoder failure leaves graph/model/transaction diagnostics unchanged");
    osprey_free(ctx);
}

static void test_input_owns_source_data(void)
{
    OspreyContext *ctx = make_projection_context(false);
    OspreyDecodeInput *input = NULL;
    CHECK(ctx != NULL && osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
              input != NULL,
          "ownership fixture builds");
    char *before = input == NULL ? NULL : dump_input(input);
    osprey_free(ctx);
    char *after = input == NULL ? NULL : dump_input(input);
    CHECK(before != NULL && after != NULL && strcmp(before, after) == 0,
          "decoder input remains complete after source graph teardown");
    free(before);
    free(after);
    osprey_decode_input_free(input);
}

static void test_null_and_missing_inputs(void)
{
    OspreyDecodeInput *input = (OspreyDecodeInput *)(uintptr_t)1;
    CHECK(osprey_decode_input_build(NULL, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "NULL context is rejected and clears output");

    OspreyContext *ctx = new_decode_context();
    CHECK(osprey_decode_input_build(ctx, NULL) == OSPREY_INVALID_MODEL,
          "NULL output pointer is rejected");
    OspreyGraph *graph = ctx->graph;
    ctx->graph = NULL;
    input = (OspreyDecodeInput *)(uintptr_t)1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "missing graph is rejected");
    osprey_graph_free(graph);
    osprey_free(ctx);

    ctx = new_decode_context();
    GArray *vars = ctx->graph->vars;
    ctx->graph->vars = NULL;
    input = (OspreyDecodeInput *)(uintptr_t)1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "missing variable storage is rejected");
    ctx->graph->vars = vars;
    osprey_free(ctx);

    ctx = new_decode_context();
    GArray *extents = ctx->graph->extents;
    ctx->graph->extents = NULL;
    input = (OspreyDecodeInput *)(uintptr_t)1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "missing extent storage is rejected");
    ctx->graph->extents = extents;
    osprey_free(ctx);
}

static void test_repeated_build_and_free(void)
{
    OspreyContext *ctx = make_projection_context(false);
    char *reference = NULL;
    for (unsigned i = 0; i < 4; i++) {
        OspreyDecodeInput *input = NULL;
        CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
                  input != NULL,
              "repeated decoder input build succeeds");
        char *dump = input == NULL ? NULL : dump_input(input);
        if (i == 0) {
            reference = dump;
        } else {
            CHECK(reference != NULL && dump != NULL &&
                      strcmp(reference, dump) == 0,
                  "repeated builds remain byte-identical");
            free(dump);
        }
        osprey_decode_input_free(input);
    }
    free(reference);
    osprey_free(ctx);
}

static void test_count_and_limit_boundaries(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    uint32_t id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                make_chunk(region, 0, 8));
    set_belief(ctx, id, 0.9);
    add_extent(ctx, region, 0, 8);
    ctx->config.max_variables = 0;
    OspreyDecodeInput *input = (OspreyDecodeInput *)(uintptr_t)1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "source variable count above the accepted bound is rejected");
    osprey_free(ctx);

    ctx = new_decode_context();
    uint32_t ids[2];
    ids[0] = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                           make_chunk(region, 0, 8));
    ids[1] = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                           make_chunk(region, 8, 8));
    set_all_beliefs(ctx, 0.9);
    add_extent(ctx, region, 0, 16);
    CHECK(osprey_factor_add_ex(ctx, OSPREY_RULE_CA02,
                               OSPREY_GRAPH_BASE_CA,
                               OSPREY_POTENTIAL_IMPLICATION, 1, false,
                               0.8, ids, 2).status == OSPREY_OK,
          "factor-bound fixture builds");
    ctx->config.max_factors = 0;
    input = (OspreyDecodeInput *)(uintptr_t)1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_INVALID_MODEL &&
              input == NULL,
          "source factor count above the accepted bound is rejected");
    osprey_free(ctx);

    ctx = new_decode_context();
    input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL &&
              input->primitive_count == 0 && input->chunk_candidate_count == 0 &&
              input->extent_count == 0,
          "zero-count arrays and indexes require no sentinel allocation");
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_allocation_failures(void)
{
    OspreyContext *ctx = make_projection_context(false);
    bool saw_success = false;
    for (int64_t failure = 0; failure < 64; failure++) {
        OspreyDecodeInput *input = (OspreyDecodeInput *)(uintptr_t)1;
        osprey_decode_test_set_alloc_fail_after(failure);
        OspreyStatus status = osprey_decode_input_build(ctx, &input);
        if (status == OSPREY_OK) {
            CHECK(input != NULL, "allocation hook success returns input");
            osprey_decode_input_free(input);
            saw_success = true;
            break;
        }
        CHECK(status == OSPREY_INVALID_MODEL && input == NULL,
              "allocation failure returns no partial input");
    }
    osprey_decode_test_set_alloc_fail_after(-1);
    CHECK(saw_success, "allocation-failure sweep reaches normal allocation");
    OspreyDecodeInput *input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL,
          "success after allocation failure remains possible");
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_threshold_configuration(void)
{
    const char *saved = g_getenv("BINRADAR_OSPREY_REPORT_THRESHOLD");
    char *copy = saved == NULL ? NULL : g_strdup(saved);
    OspreyConfig config;
    g_unsetenv("BINRADAR_OSPREY_REPORT_THRESHOLD");
    CHECK(osprey_config_from_env(&config) && config.report_threshold == 0.6,
          "report threshold defaults to 0.6");
    const char *valid[] = {
        "0", "-0", "0.6", "1", "2.2250738585072014e-308",
    };
    for (size_t i = 0; i < G_N_ELEMENTS(valid); i++) {
        g_setenv("BINRADAR_OSPREY_REPORT_THRESHOLD", valid[i], TRUE);
        CHECK(osprey_config_from_env(&config) &&
                  isfinite(config.report_threshold) &&
                  config.report_threshold >= 0.0 &&
                  config.report_threshold <= 1.0,
              "report threshold accepts finite boundary strings");
    }
    const char *invalid[] = {
        "nan", "+nan", "inf", "+inf", "-inf", "-0.1", "1.1",
        "garbage", "0.5x", "1e-9999",
    };
    for (size_t i = 0; i < G_N_ELEMENTS(invalid); i++) {
        g_setenv("BINRADAR_OSPREY_REPORT_THRESHOLD", invalid[i], TRUE);
        CHECK(!osprey_config_from_env(&config),
              "report threshold rejects malformed or nonfinite input");
    }
    if (copy != NULL) g_setenv("BINRADAR_OSPREY_REPORT_THRESHOLD", copy, TRUE);
    else g_unsetenv("BINRADAR_OSPREY_REPORT_THRESHOLD");
    g_free(copy);
}

static uint32_t add_field_candidate(OspreyContext *ctx, OspreyChunk chunk,
                                    OspreyAddress base, double belief)
{
    OspreyVarPayload payload;
    memset(&payload, 0, sizeof(payload));
    payload.attached.chunk = chunk;
    payload.attached.base = base;
    uint32_t id = add_payload(ctx, OSPREY_PRED_FIELD_OF, &payload);
    set_belief(ctx, id, belief);
    return id;
}

static uint32_t add_pointer_candidate(OspreyContext *ctx, OspreyChunk chunk,
                                      OspreyAddress target, double belief)
{
    OspreyVarPayload payload;
    memset(&payload, 0, sizeof(payload));
    payload.attached.chunk = chunk;
    payload.attached.base = target;
    uint32_t id = add_payload(ctx, OSPREY_PRED_POINTER, &payload);
    set_belief(ctx, id, belief);
    return id;
}

static uint32_t add_array_candidate(OspreyContext *ctx,
                                    OspreyRegionId region, int64_t lo,
                                    int64_t hi, int64_t stride, double belief)
{
    OspreyVarPayload payload;
    memset(&payload, 0, sizeof(payload));
    payload.segment.a1 = make_address(region, lo);
    payload.segment.a2 = make_address(region, hi);
    payload.segment.size = stride;
    uint32_t id = add_payload(ctx, OSPREY_PRED_ARRAY, &payload);
    set_belief(ctx, id, belief);
    return id;
}

static char *dump_plan(const OspreyDecodePlan *plan)
{
    char *data = NULL;
    size_t length = 0;
    FILE *out = open_memstream(&data, &length);
    if (out == NULL) return NULL;
    if (!osprey_decode_plan_dump_file(plan, out) || fclose(out) != 0) {
        free(data);
        return NULL;
    }
    return data;
}

static bool plan_has_role_loss(const OspreyDecodePlan *plan,
                               const OspreyKey *key)
{
    if (plan == NULL || key == NULL) return false;
    for (uint32_t i = 0; i < plan->role_loss_count; i++) {
        if (memcmp(&plan->role_loss_keys[i], key,
                   sizeof(plan->role_loss_keys[i])) == 0) return true;
    }
    return false;
}

static const OspreyChunkDecision *plan_find_decision(
    const OspreyDecodePlan *plan, const OspreyChunk *chunk)
{
    if (plan == NULL || chunk == NULL) return NULL;
    for (uint32_t i = 0; i < plan->decision_count; i++) {
        if (memcmp(&plan->decisions[i].chunk, chunk,
                   sizeof(*chunk)) == 0) {
            return &plan->decisions[i];
        }
    }
    return NULL;
}

static void test_stage62_orthogonal_roles(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId global = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    OspreyRegionId heap = make_region(OSPREY_REGION_HEAP_SITE, 3, 4);
    OspreyChunk cell = make_chunk(global, 8, sizeof(target_ulong));
    OspreyAddress target0 = make_address(heap, 0);
    OspreyAddress target8 = make_address(heap, 8);
    uint32_t primitive = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, cell);
    uint32_t scalar = add_chunk_var(ctx, OSPREY_PRED_SCALAR, cell);
    uint32_t pointer0 = add_pointer_candidate(ctx, cell, target0, 0.95);
    add_pointer_candidate(ctx, cell, target8, 0.95);
    add_pointer_candidate(ctx, cell, make_address(heap, 100), 0.8);
    add_extent(ctx, global, 0, 32);
    add_extent(ctx, heap, 0, 32);
    set_belief(ctx, primitive, 0.7);
    set_belief(ctx, scalar, 0.8);
    OspreyVar *scalar_var = &g_array_index(ctx->graph->vars, OspreyVar,
                                           scalar);
    OspreyVar *pointer_var = &g_array_index(ctx->graph->vars, OspreyVar,
                                            pointer0);
    scalar_var->direct_support = 11;
    scalar_var->source_rule_bits = UINT64_C(0x12);
    pointer_var->direct_support = 13;
    pointer_var->source_rule_bits = UINT64_C(0x34);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL,
          "Stage 6.2 orthogonal fixture builds input");
    if (input != NULL) {
        CHECK(osprey_decode_roles(ctx, input, &plan) == OSPREY_OK &&
                  plan != NULL && plan->decision_count == 1,
              "Stage 6.2 orthogonal fixture builds plan");
        if (plan != NULL) {
            const OspreyChunkDecision *decision =
                plan_find_decision(plan, &cell);
            OspreyKey primitive_key = osprey_var_key(
                OSPREY_PRED_PRIMITIVE_VAR,
                &(OspreyVarPayload){ .chunk = cell });
            OspreyKey pointer_bad_key = osprey_var_key(
                OSPREY_PRED_POINTER,
                &(OspreyVarPayload){ .attached = {
                    .chunk = cell, .base = make_address(heap, 100) }});
            CHECK(decision != NULL &&
                      decision->provisional_role == OSPREY_STORAGE_SCALAR &&
                      decision->final_role == OSPREY_STORAGE_SCALAR &&
                      decision->role_has_predicate &&
                      decision->role_posterior == 0.8 &&
                      decision->role_support == 11 &&
                      decision->role_source_rule_bits == UINT64_C(0x12) &&
                      decision->has_pointer_target &&
                      decision->pointer_target.offset == 0 &&
                      decision->pointer_posterior == 0.95 &&
                      decision->pointer_support == 13 &&
                      decision->pointer_source_rule_bits == UINT64_C(0x34),
                  "scalar storage and pointer target retain exact evidence");
            OspreyKey pointer_key = osprey_var_key(
                OSPREY_PRED_POINTER,
                &(OspreyVarPayload){ .attached = {
                    .chunk = cell, .base = target0 }});
            CHECK(decision != NULL &&
                      memcmp(&decision->pointer_key, &pointer_key,
                             sizeof(pointer_key)) == 0,
                  "pointer decision retains a selected target record");
            CHECK(plan_has_role_loss(plan, &primitive_key) &&
                      plan_has_role_loss(plan, &pointer_bad_key) &&
                      plan->role_loss_count == 3,
                  "unused primitive and losing pointer targets are counted once");
        }
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage62_baseline_role_matrix(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId global = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    OspreyRegionId heap = make_region(OSPREY_REGION_HEAP_SITE, 3, 4);
    OspreyChunk primitive_chunk = make_chunk(global, 0, 4);
    OspreyChunk scalar_chunk = make_chunk(global, 8, 4);
    OspreyChunk field_pointer_chunk = make_chunk(heap, 8, 8);
    OspreyChunk tied_chunk = make_chunk(heap, 16, 8);
    OspreyChunk pointer_only_chunk = make_chunk(global, 24,
                                                sizeof(target_ulong));
    uint32_t primitive = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                       primitive_chunk);
    uint32_t scalar = add_chunk_var(ctx, OSPREY_PRED_SCALAR, scalar_chunk);
    add_field_candidate(ctx, field_pointer_chunk, make_address(heap, 0), 0.9);
    add_pointer_candidate(ctx, field_pointer_chunk, make_address(heap, 0), 0.8);
    add_field_candidate(ctx, tied_chunk, make_address(heap, 0), 0.8);
    uint32_t tied_scalar = add_chunk_var(ctx, OSPREY_PRED_SCALAR, tied_chunk);
    add_pointer_candidate(ctx, pointer_only_chunk, make_address(heap, 0), 0.9);
    set_belief(ctx, primitive, 0.7);
    set_belief(ctx, scalar, 0.8);
    set_belief(ctx, tied_scalar, 0.8);
    add_extent(ctx, global, 0, 64);
    add_extent(ctx, heap, 0, 32);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
              osprey_decode_roles(ctx, input, &plan) == OSPREY_OK &&
              plan != NULL,
          "baseline role matrix builds");
    if (plan != NULL) {
        const OspreyChunkDecision *primitive_decision =
            plan_find_decision(plan, &primitive_chunk);
        const OspreyChunkDecision *scalar_decision =
            plan_find_decision(plan, &scalar_chunk);
        const OspreyChunkDecision *field_pointer_decision =
            plan_find_decision(plan, &field_pointer_chunk);
        const OspreyChunkDecision *tied_decision =
            plan_find_decision(plan, &tied_chunk);
        const OspreyChunkDecision *pointer_only_decision =
            plan_find_decision(plan, &pointer_only_chunk);
        CHECK(primitive_decision != NULL &&
                  primitive_decision->provisional_role == OSPREY_STORAGE_PRIMITIVE &&
                  primitive_decision->role_has_predicate,
              "P01 provides an explicit primitive fallback");
        CHECK(scalar_decision != NULL &&
                  scalar_decision->provisional_role == OSPREY_STORAGE_SCALAR,
              "P07 provides scalar storage without P01");
        CHECK(field_pointer_decision != NULL &&
                  field_pointer_decision->provisional_role == OSPREY_STORAGE_FIELD &&
                  field_pointer_decision->has_pointer_target,
              "pointer-valued field retains field storage ownership");
        CHECK(tied_decision != NULL &&
                  tied_decision->provisional_role == OSPREY_STORAGE_SCALAR,
              "equal scalar and field posteriors use canonical predicate order");
        CHECK(pointer_only_decision != NULL &&
                  pointer_only_decision->provisional_role == OSPREY_STORAGE_PRIMITIVE &&
                  !pointer_only_decision->role_has_predicate &&
                  pointer_only_decision->has_pointer_target &&
                  pointer_only_decision->role_posterior == 0.0 &&
                  pointer_only_decision->role_support == 0 &&
                  pointer_only_decision->role_source_rule_bits == 0,
              "P10-only cell receives an evidence-free implicit primitive role");
        CHECK(plan->field_group_count == 1 &&
                  plan->field_groups[0].field_count == 1,
              "scalar displacement rebuilds the field group from role winners");
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage62_field_selection(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    OspreyAddress base0 = make_address(region, 0);
    OspreyAddress base4 = make_address(region, 4);
    OspreyChunk chunk = make_chunk(region, 8, 8);
    uint32_t low_base = add_field_candidate(ctx, chunk, base0, 0.8);
    uint32_t high_base = add_field_candidate(ctx, chunk, base4, 0.9);
    add_extent(ctx, region, 0, 32);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
              osprey_decode_roles(ctx, input, &plan) == OSPREY_OK &&
              plan != NULL,
          "one-base field fixture builds");
    if (plan != NULL) {
        OspreyKey low_key = osprey_var_key(
            OSPREY_PRED_FIELD_OF,
            &(OspreyVarPayload){ .attached = {
                .chunk = chunk, .base = base0 }});
        const OspreyChunkDecision *decision =
            plan_find_decision(plan, &chunk);
        CHECK(decision != NULL && decision->provisional_role ==
                  OSPREY_STORAGE_FIELD && decision->owner_base.offset == 4 &&
                  plan->field_group_count == 1 &&
                  plan->field_groups[0].base.offset == 4 &&
                  plan->field_groups[0].field_count == 1,
              "higher-posterior field base wins deterministically");
        CHECK(plan_has_role_loss(plan, &low_key) && plan->role_loss_count == 1,
              "losing field base enters role-loss diagnostics");
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
    (void)low_base;
    (void)high_base;

    ctx = new_decode_context();
    ctx->config.report_threshold = 0.0;
    region = make_region(OSPREY_REGION_GLOBAL, 5, 6);
    base0 = make_address(region, 0);
    OspreyChunk wide = make_chunk(region, 0, 16);
    OspreyChunk narrow = make_chunk(region, 0, 8);
    OspreyChunk adjacent = make_chunk(region, 8, 8);
    uint32_t wide_id = add_field_candidate(ctx, wide, base0, 1.0);
    add_field_candidate(ctx, narrow, base0, 0.5);
    add_field_candidate(ctx, adjacent, base0, 0.5);
    add_extent(ctx, region, 0, 32);
    input = NULL;
    plan = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
              osprey_decode_roles(ctx, input, &plan) == OSPREY_OK &&
              plan != NULL,
          "weighted field schedule fixture builds");
    if (plan != NULL) {
        OspreyKey wide_key = osprey_var_key(
            OSPREY_PRED_FIELD_OF,
            &(OspreyVarPayload){ .attached = {
                .chunk = wide, .base = base0 }});
        CHECK(plan->field_group_count == 1 &&
                  plan->field_groups[0].field_count == 2 &&
                  plan_has_role_loss(plan, &wide_key),
              "compatible adjacent fields beat an equal-score wide field");
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
    (void)wide_id;

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 7, 8);
    base0 = make_address(region, 0);
    OspreyChunk first = make_chunk(region, 0, 8);
    adjacent = make_chunk(region, 8, 8);
    uint32_t first_field = add_field_candidate(ctx, first, base0, 0.8);
    uint32_t second_field = add_field_candidate(ctx, adjacent, base0, 0.8);
    uint32_t second_scalar = add_chunk_var(ctx, OSPREY_PRED_SCALAR, adjacent);
    set_belief(ctx, second_scalar, 0.9);
    add_extent(ctx, region, 0, 32);
    input = NULL;
    plan = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
              osprey_decode_roles(ctx, input, &plan) == OSPREY_OK &&
              plan != NULL,
          "field rebuild fixture builds");
    if (plan != NULL) {
        OspreyKey second_key = osprey_var_key(
            OSPREY_PRED_FIELD_OF,
            &(OspreyVarPayload){ .attached = {
                .chunk = adjacent, .base = base0 }});
        const OspreyChunkDecision *second_decision =
            plan_find_decision(plan, &adjacent);
        CHECK(second_decision != NULL &&
                  second_decision->provisional_role == OSPREY_STORAGE_SCALAR &&
                  plan->field_group_count == 1 &&
                  plan->field_groups[0].field_count == 1 &&
                  plan_has_role_loss(plan, &second_key),
              "scalar displacement rebuilds surviving field groups");
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
    (void)first_field;
    (void)second_field;

    ctx = new_decode_context();
    ctx->config.report_threshold = 0.0;
    region = make_region(OSPREY_REGION_GLOBAL, 9, 10);
    base0 = make_address(region, 0);
    OspreyChunk zero_field = make_chunk(region, 0, 1);
    OspreyChunk positive_field = make_chunk(region, 1, 8);
    add_field_candidate(ctx, zero_field, base0, 0.0);
    add_field_candidate(ctx, positive_field, base0, 1.0);
    add_extent(ctx, region, 0, 16);
    input = NULL;
    plan = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
              osprey_decode_roles(ctx, input, &plan) == OSPREY_OK &&
              plan != NULL && plan->field_group_count == 1 &&
              plan->field_groups[0].field_count == 2,
          "zero-score prefix fields retain canonical maximum schedule");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    ctx->config.report_threshold = 0.0;
    region = make_region(OSPREY_REGION_GLOBAL, 11, 12);
    base0 = make_address(region, 0);
    wide = make_chunk(region, 0, 16);
    narrow = make_chunk(region, 0, 8);
    adjacent = make_chunk(region, 8, 8);
    zero_field = make_chunk(region, 24, 8);
    add_field_candidate(ctx, wide, base0, 0.9);
    add_field_candidate(ctx, narrow, base0, 0.4);
    add_field_candidate(ctx, adjacent, base0, 0.4);
    add_field_candidate(ctx, zero_field, make_address(region, 24), 0.0);
    add_extent(ctx, region, 0, 32);
    input = NULL;
    plan = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
              osprey_decode_roles(ctx, input, &plan) == OSPREY_OK &&
              plan != NULL,
          "wide-field and empty-schedule fixture builds");
    if (plan != NULL) {
        const OspreyChunkDecision *wide_decision =
            plan_find_decision(plan, &wide);
        const OspreyChunkDecision *zero_decision =
            plan_find_decision(plan, &zero_field);
        CHECK(wide_decision != NULL &&
                  wide_decision->provisional_role == OSPREY_STORAGE_FIELD,
              "one wide field beats a lower-score compatible pair");
        CHECK(zero_decision != NULL &&
                  zero_decision->provisional_role == OSPREY_STORAGE_PRIMITIVE &&
                  !zero_decision->role_has_predicate,
              "empty schedule beats an isolated exact-zero field");
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage62_signed_canonical_order(void)
{
    OspreyContext *ctx = new_decode_context();
    ctx->config.report_threshold = 0.0;
    OspreyRegionId region = make_region(OSPREY_REGION_STACK_FUNCTION, 7, 8);
    OspreyAddress schedule_base = make_address(region, -16);
    OspreyChunk tied_chunk = make_chunk(region, 8, 8);
    OspreyChunk negative_wide = make_chunk(region, -8, 16);
    OspreyChunk positive_narrow = make_chunk(region, 0, 8);
    OspreyChunk pointer_cell = make_chunk(region, 16, sizeof(target_ulong));
    add_field_candidate(ctx, tied_chunk, make_address(region, 0), 0.9);
    add_field_candidate(ctx, tied_chunk, make_address(region, -8), 0.9);
    add_field_candidate(ctx, positive_narrow, schedule_base, 0.5);
    add_field_candidate(ctx, negative_wide, schedule_base, 0.5);
    add_pointer_candidate(ctx, pointer_cell, make_address(region, 0), 0.8);
    add_pointer_candidate(ctx, pointer_cell, make_address(region, -8), 0.8);
    add_extent(ctx, region, -32, 32);

    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL,
          "signed-order fixture builds decoder input");
    if (input != NULL) {
        CHECK(input->field_count == 4 &&
                  input->field_candidates[0].payload.attached.chunk.address.offset == -8 &&
                  input->array_count == 0,
              "primary candidates retain signed canonical payload order");
        CHECK(osprey_decode_roles(ctx, input, &plan) == OSPREY_OK &&
                  plan != NULL,
              "signed-order fixture builds role plan");
    }
    if (plan != NULL) {
        const OspreyChunkDecision *field =
            plan_find_decision(plan, &tied_chunk);
        const OspreyChunkDecision *negative =
            plan_find_decision(plan, &negative_wide);
        const OspreyChunkDecision *positive =
            plan_find_decision(plan, &positive_narrow);
        const OspreyChunkDecision *pointer =
            plan_find_decision(plan, &pointer_cell);
        CHECK(field != NULL && field->provisional_role == OSPREY_STORAGE_FIELD &&
                  field->owner_base.offset == -8,
              "equal field-base posterior uses signed canonical target order");
        CHECK(negative != NULL &&
                  negative->provisional_role == OSPREY_STORAGE_FIELD &&
                  positive != NULL &&
                  positive->provisional_role == OSPREY_STORAGE_PRIMITIVE,
              "equal field schedules use signed canonical chunk order");
        CHECK(pointer != NULL && pointer->has_pointer_target &&
                  pointer->pointer_target.offset == -8,
              "equal pointer posterior uses signed canonical target order");
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    OspreyVarPayload negative_array;
    OspreyVarPayload positive_array;
    memset(&negative_array, 0, sizeof(negative_array));
    memset(&positive_array, 0, sizeof(positive_array));
    negative_array.segment.a1 = make_address(region, -16);
    negative_array.segment.a2 = make_address(region, -8);
    negative_array.segment.size = 8;
    positive_array.segment.a1 = make_address(region, 0);
    positive_array.segment.a2 = make_address(region, 8);
    positive_array.segment.size = 8;
    uint32_t positive_id = add_payload(ctx, OSPREY_PRED_ARRAY,
                                       &positive_array);
    uint32_t negative_id = add_payload(ctx, OSPREY_PRED_ARRAY,
                                       &negative_array);
    set_belief(ctx, positive_id, 0.9);
    set_belief(ctx, negative_id, 0.9);
    add_extent(ctx, region, -32, 32);
    input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL &&
              input->array_count == 2 && input->array_by_region[0] == 0 &&
              input->array_by_region[1] == 1 &&
              input->array_by_region_stride[0] == 0 &&
              input->array_by_region_stride[1] == 1,
          "array indexes use signed canonical candidate order");
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage62_checked_field_end(void)
{
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    OspreyContext *ctx = new_decode_context();
    OspreyChunk overflowing = make_chunk(region, -1, 1);
    add_field_candidate(ctx, overflowing, make_address(region, INT64_MIN), 1.0);
    add_extent(ctx, region, INT64_MIN, 1);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = (OspreyDecodePlan *)(uintptr_t)1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL &&
              osprey_decode_roles(ctx, input, &plan) == OSPREY_INVALID_MODEL &&
              plan == NULL,
          "selected field rejects relative-end signed overflow");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    OspreyChunk boundary = make_chunk(region, INT64_MAX - 1, 1);
    add_field_candidate(ctx, boundary,
                        make_address(region, INT64_MAX - 1), 1.0);
    add_extent(ctx, region, INT64_MAX - 1, INT64_MAX);
    input = NULL;
    plan = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
              osprey_decode_roles(ctx, input, &plan) == OSPREY_OK &&
              plan != NULL && plan->field_group_count == 1,
          "selected field accepts exact INT64_MAX exclusive end");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage62_rejects_selected_geometry(void)
{
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    OspreyContext *ctx = new_decode_context();
    OspreyChunk cell = make_chunk(region, 0, 8);
    add_field_candidate(ctx, cell, make_address(region, -8), 0.9);
    add_extent(ctx, region, 0, 16);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = (OspreyDecodePlan *)(uintptr_t)1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
              osprey_decode_roles(ctx, input, &plan) == OSPREY_INVALID_MODEL &&
              plan == NULL,
          "selected field outside its extent rejects without a plan");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    OspreyChunk wrong_width = make_chunk(region, 0, 4);
    add_pointer_candidate(ctx, wrong_width, make_address(region, 0), 1.0);
    add_extent(ctx, region, 0, 16);
    input = NULL;
    plan = (OspreyDecodePlan *)(uintptr_t)1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
              osprey_decode_roles(ctx, input, &plan) == OSPREY_INVALID_MODEL &&
              plan == NULL,
          "selected pointer cell with wrong width rejects atomically");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    OspreyChunk pointer_cell = make_chunk(region, 0, sizeof(target_ulong));
    add_pointer_candidate(ctx, pointer_cell, make_address(region, 16), 1.0);
    add_extent(ctx, region, 0, 16);
    input = NULL;
    plan = (OspreyDecodePlan *)(uintptr_t)1;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
              osprey_decode_roles(ctx, input, &plan) == OSPREY_INVALID_MODEL &&
              plan == NULL,
          "selected pointer target at the exclusive extent end rejects");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    add_field_candidate(ctx, cell, make_address(region, 0), 0.9);
    add_field_candidate(ctx, cell, make_address(region, -8), 0.8);
    add_extent(ctx, region, 0, 16);
    input = NULL;
    plan = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
              osprey_decode_roles(ctx, input, &plan) == OSPREY_OK &&
              plan != NULL,
          "invalid losing field geometry does not reject a valid winner");
    if (plan != NULL) {
        OspreyKey losing_key = osprey_var_key(
            OSPREY_PRED_FIELD_OF,
            &(OspreyVarPayload){ .attached = {
                .chunk = cell, .base = make_address(region, -8) }});
        CHECK(plan_has_role_loss(plan, &losing_key),
              "invalid losing field remains a discarded-role candidate");
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    OspreyRegionId target_a = make_region(OSPREY_REGION_HEAP_SITE, 2, 4);
    OspreyRegionId target_b = make_region(OSPREY_REGION_HEAP_SITE, 2, 5);
    add_pointer_candidate(ctx, pointer_cell, make_address(target_b, 0), 0.9);
    add_pointer_candidate(ctx, pointer_cell, make_address(target_a, 0), 0.9);
    add_extent(ctx, region, 0, 16);
    add_extent(ctx, target_a, 0, 8);
    add_extent(ctx, target_b, 0, 8);
    input = NULL;
    plan = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK &&
              osprey_decode_roles(ctx, input, &plan) == OSPREY_OK &&
              plan != NULL,
          "pointer region-identity tie fixture builds");
    if (plan != NULL) {
        const OspreyChunkDecision *decision =
            plan_find_decision(plan, &pointer_cell);
        CHECK(decision != NULL && decision->has_pointer_target &&
                  decision->pointer_target.region.site_offset == 4,
              "pointer target tie compares complete region identity");
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage62_rejects_noncanonical_slices(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    uint32_t first = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                   make_chunk(region, 0, 8));
    uint32_t second = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                    make_chunk(region, 8, 8));
    set_belief(ctx, first, 0.9);
    set_belief(ctx, second, 0.9);
    add_extent(ctx, region, 0, 16);
    OspreyDecodeInput *input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL &&
              input->chunk_candidate_count == 2 &&
              input->chunk_range_count == 2,
          "noncanonical-slice fixture builds input");
    if (input != NULL && input->chunk_candidate_count == 2 &&
        input->chunk_range_count == 2) {
        OspreyDecodeCandidateRef ref = input->chunk_candidates[0];
        input->chunk_candidates[0] = input->chunk_candidates[1];
        input->chunk_candidates[1] = ref;
        input->chunk_ranges[0].begin = 1;
        input->chunk_ranges[1].begin = 0;
        OspreyDecodePlan *plan = (OspreyDecodePlan *)(uintptr_t)1;
        CHECK(osprey_decode_roles(ctx, input, &plan) == OSPREY_INVALID_MODEL &&
                  plan == NULL,
              "semantically matching but noncanonical range slices reject");
        osprey_decode_plan_free(plan);
    }
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage62_rejects_array_candidate_alias(void)
{
    OspreyContext *ctx = new_decode_context();
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 1, 2);
    uint32_t primitive_id = add_chunk_var(
        ctx, OSPREY_PRED_PRIMITIVE_VAR, make_chunk(region, 0, 8));
    OspreyVarPayload array_payload;
    memset(&array_payload, 0, sizeof(array_payload));
    array_payload.segment.a1 = make_address(region, 0);
    array_payload.segment.a2 = make_address(region, 8);
    array_payload.segment.size = 8;
    uint32_t array_id = add_payload(ctx, OSPREY_PRED_ARRAY, &array_payload);
    set_belief(ctx, primitive_id, 0.9);
    set_belief(ctx, array_id, 0.9);
    add_extent(ctx, region, 0, 8);
    OspreyDecodeInput *input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL &&
              input->primitive_count == 1 && input->array_count == 1,
          "array-alias fixture builds input");
    if (input != NULL && input->primitive_count == 1 && input->array_count == 1) {
        const size_t candidate_size = sizeof(OspreyDecodeCandidate);
        const size_t overlap = sizeof(uint64_t);
        uint8_t *storage = g_malloc0(candidate_size * 2);
        OspreyDecodeCandidate primitive = input->primitive_candidates[0];
        OspreyDecodeCandidate array = input->array_candidates[0];
        OspreyDecodeCandidate *saved_primitive = input->primitive_candidates;
        OspreyDecodeCandidate *saved_array = input->array_candidates;
        primitive.source_rule_bits = array.key.tag;
        memcpy(storage, &primitive, candidate_size);
        memcpy(storage + candidate_size - overlap, &array, candidate_size);
        input->primitive_candidates = (OspreyDecodeCandidate *)storage;
        input->array_candidates = (OspreyDecodeCandidate *)(
            storage + candidate_size - overlap);
        OspreyDecodePlan *plan = (OspreyDecodePlan *)(uintptr_t)1;
        CHECK(osprey_decode_roles(ctx, input, &plan) == OSPREY_INVALID_MODEL &&
                  plan == NULL,
              "overlapping array candidate ownership rejects atomically");
        osprey_decode_plan_free(plan);
        input->primitive_candidates = saved_primitive;
        input->array_candidates = saved_array;
        g_free(storage);
    }
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage62_plan_permutation_and_failures(void)
{
    OspreyContext *left = make_projection_context(0);
    OspreyContext *right = make_projection_context(1);
    OspreyContext *shuffled = make_projection_context(2);
    OspreyDecodeInput *left_input = NULL;
    OspreyDecodeInput *right_input = NULL;
    OspreyDecodeInput *shuffled_input = NULL;
    OspreyDecodePlan *left_plan = NULL;
    OspreyDecodePlan *right_plan = NULL;
    OspreyDecodePlan *shuffled_plan = NULL;
    CHECK(osprey_decode_input_build(left, &left_input) == OSPREY_OK &&
              osprey_decode_input_build(right, &right_input) == OSPREY_OK &&
              osprey_decode_input_build(shuffled, &shuffled_input) == OSPREY_OK &&
              osprey_decode_roles(left, left_input, &left_plan) == OSPREY_OK &&
              osprey_decode_roles(right, right_input, &right_plan) == OSPREY_OK &&
              osprey_decode_roles(shuffled, shuffled_input,
                                  &shuffled_plan) == OSPREY_OK,
          "permutation role plans build");
    char *left_dump = left_plan == NULL ? NULL : dump_plan(left_plan);
    char *right_dump = right_plan == NULL ? NULL : dump_plan(right_plan);
    char *shuffled_dump = shuffled_plan == NULL ? NULL :
        dump_plan(shuffled_plan);
    CHECK(left_dump != NULL && right_dump != NULL && shuffled_dump != NULL &&
              strcmp(left_dump, right_dump) == 0 &&
              strcmp(left_dump, shuffled_dump) == 0,
          "role decisions and diagnostics ignore graph insertion order");
    free(left_dump);
    free(right_dump);
    free(shuffled_dump);
    osprey_decode_plan_free(left_plan);
    osprey_decode_plan_free(right_plan);
    osprey_decode_plan_free(shuffled_plan);
    osprey_decode_input_free(left_input);
    osprey_decode_input_free(right_input);
    osprey_decode_input_free(shuffled_input);
    osprey_free(left);
    osprey_free(right);
    osprey_free(shuffled);

    OspreyContext *ctx = make_projection_context(0);
    OspreyDecodeInput *input = NULL;
    CHECK(osprey_decode_input_build(ctx, &input) == OSPREY_OK && input != NULL,
          "allocation sweep input builds");
    char *input_before = input == NULL ? NULL : dump_input(input);
    OspreyGraph *saved_graph = ctx->graph;
    OspreyModel *saved_model = ctx->model;
    OspreyModel *saved_staged_model = ctx->staged_model;
    OspreyStatus saved_tx_status = ctx->tx_status;
    OspreyStatus saved_last_status = ctx->last_status;
    const char *saved_tx_stage = ctx->tx_stage;
    const char *saved_tx_reason = ctx->tx_reason;
    bool saved_tx_model_ready = ctx->tx_model_ready;
    bool saw_success = false;
    for (int64_t failure = 0; failure < 256; failure++) {
        OspreyDecodePlan *plan = (OspreyDecodePlan *)(uintptr_t)1;
        osprey_decode_test_set_alloc_fail_after(failure);
        OspreyStatus status = osprey_decode_roles(ctx, input, &plan);
        if (status == OSPREY_OK) {
            CHECK(plan != NULL, "role allocation hook success returns plan");
            osprey_decode_plan_free(plan);
            saw_success = true;
            break;
        }
        CHECK(status == OSPREY_INVALID_MODEL && plan == NULL,
              "role allocation failure returns no partial plan");
    }
    osprey_decode_test_set_alloc_fail_after(-1);
    CHECK(saw_success, "role allocation sweep reaches normal allocation");
    char *input_after = input == NULL ? NULL : dump_input(input);
    CHECK(input_before != NULL && input_after != NULL &&
              strcmp(input_before, input_after) == 0,
          "role success and allocation failures leave decoder input unchanged");
    CHECK(ctx->graph == saved_graph && ctx->model == saved_model &&
              ctx->staged_model == saved_staged_model &&
              ctx->tx_status == saved_tx_status &&
              ctx->last_status == saved_last_status &&
              ctx->tx_stage == saved_tx_stage &&
              ctx->tx_reason == saved_tx_reason &&
              ctx->tx_model_ready == saved_tx_model_ready,
          "role success and allocation failures leave transaction state unchanged");
    free(input_before);
    free(input_after);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static bool build_array_input_and_roles(OspreyContext *ctx,
                                        OspreyDecodeInput **input_out,
                                        OspreyDecodePlan **plan_out)
{
    *input_out = NULL;
    *plan_out = NULL;
    return osprey_decode_input_build(ctx, input_out) == OSPREY_OK &&
           osprey_decode_roles(ctx, *input_out, plan_out) == OSPREY_OK;
}

static bool build_array_plan(OspreyContext *ctx,
                             OspreyDecodeInput **input_out,
                             OspreyDecodePlan **plan_out)
{
    return build_array_input_and_roles(ctx, input_out, plan_out) &&
           osprey_decode_select_arrays(ctx, *input_out, *plan_out) ==
               OSPREY_OK;
}

static void test_stage63_arrays_and_roles(void)
{
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 0x601, 0x701);
    OspreyContext *ctx = new_decode_context();
    OspreyChunk first = make_chunk(region, 0, 8);
    OspreyChunk second = make_chunk(region, 8, 8);
    OspreyChunk third = make_chunk(region, 16, 8);
    uint32_t first_id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, first);
    uint32_t second_id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, second);
    uint32_t third_id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, third);
    set_belief(ctx, first_id, 0.7);
    set_belief(ctx, second_id, 0.7);
    set_belief(ctx, third_id, 0.7);
    uint32_t exact_array = add_array_candidate(ctx, region, 0, 24, 8, 0.9);
    add_extent(ctx, region, 0, 32);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL,
          "Stage 6.3 exact array plan builds");
    if (plan != NULL) {
        CHECK(plan->array_count == 1 && plan->discarded_layout == 0 &&
                  plan->arrays[0].lo == 0 && plan->arrays[0].hi == 24 &&
                  plan->arrays[0].stride == 8 && plan->arrays[0].count == 3 &&
                  plan->arrays[0].member_count == 3,
              "array geometry and whole-element count are exact");
        CHECK(plan->decisions[0].final_role == OSPREY_STORAGE_ARRAY_ELEMENT &&
                  plan->decisions[1].final_role == OSPREY_STORAGE_ARRAY_ELEMENT &&
                  plan->decisions[2].final_role == OSPREY_STORAGE_ARRAY_ELEMENT,
              "every aligned observed chunk becomes an array element");
        CHECK(plan->arrays[0].source_rule_bits ==
                  g_array_index(ctx->graph->vars, OspreyVar, exact_array)
                      .source_rule_bits,
              "selected array evidence is preserved");
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x602, 0x702);
    add_array_candidate(ctx, region, 0, 10, 8, 0.9);
    add_extent(ctx, region, 0, 32);
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 0 && plan->discarded_layout == 1,
          "non-divisible span is excluded without a partial element");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x603, 0x703);
    add_array_candidate(ctx, region, 0, 8, 8, 0.9);
    add_array_candidate(ctx, region, 8, 16, 8, 0.9);
    add_extent(ctx, region, 0, 16);
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 2 && plan->arrays[0].hi == 8 &&
              plan->arrays[1].lo == 8,
          "half-open adjacent arrays are both selected");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x604, 0x704);
    add_array_candidate(ctx, region, 0, 24, 8, 0.9);
    add_array_candidate(ctx, region, 0, 8, 8, 0.8);
    add_array_candidate(ctx, region, 8, 16, 8, 0.8);
    add_array_candidate(ctx, region, 16, 24, 8, 0.8);
    add_extent(ctx, region, 0, 24);
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 3 && plan->discarded_layout == 1,
          "compatible narrow arrays beat one lower-score wide array");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x605, 0x705);
    add_array_candidate(ctx, region, 0, 16, 8, 0.9);
    add_array_candidate(ctx, region, 0, 15, 5, 0.95);
    add_extent(ctx, region, 0, 16);
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->arrays[0].stride == 5 &&
              plan->discarded_layout == 1,
          "different strides compete in one region schedule");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x606, 0x706);
    OspreyRegionId target_region = make_region(OSPREY_REGION_HEAP_SITE,
                                               0x607, 0x707);
    OspreyChunk pointer_cell = make_chunk(region, 0, sizeof(target_ulong));
    uint32_t scalar_id = add_chunk_var(ctx, OSPREY_PRED_SCALAR, pointer_cell);
    set_belief(ctx, scalar_id, 0.7);
    add_pointer_candidate(ctx, pointer_cell, make_address(target_region, 0),
                          0.95);
    add_array_candidate(ctx, region, 0, 8, 8, 1.0);
    add_extent(ctx, region, 0, 16);
    add_extent(ctx, target_region, 0, 8);
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->decisions[0].final_role ==
                  OSPREY_STORAGE_ARRAY_ELEMENT &&
              plan->decisions[0].has_pointer_target &&
              plan->decisions[0].pointer_target.region.kind ==
                  OSPREY_REGION_HEAP_SITE &&
              plan->decisions[0].has_array_owner,
          "array displacement preserves orthogonal pointer evidence");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x608, 0x708);
    OspreyAddress base = make_address(region, 0);
    OspreyChunk field0 = make_chunk(region, 0, 8);
    OspreyChunk field1 = make_chunk(region, 16, 8);
    add_field_candidate(ctx, field0, base, 0.9);
    add_field_candidate(ctx, field1, base, 0.9);
    add_array_candidate(ctx, region, 0, 8, 8, 1.0);
    add_extent(ctx, region, 0, 32);
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->field_group_count == 1 &&
              plan->field_groups[0].field_count == 1 &&
              plan->decisions[0].final_role == OSPREY_STORAGE_ARRAY_ELEMENT &&
              plan->decisions[1].final_role == OSPREY_STORAGE_FIELD,
          "array displacement rebuilds and retains nonempty fields");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x609, 0x709);
    base = make_address(region, 0);
    field0 = make_chunk(region, 0, 8);
    add_field_candidate(ctx, field0, base, 0.9);
    add_array_candidate(ctx, region, 0, 8, 8, 1.0);
    add_extent(ctx, region, 0, 16);
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->field_group_count == 0,
          "array displacement drops empty structures");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    ctx->config.report_threshold = 0.0;
    region = make_region(OSPREY_REGION_GLOBAL, 0x60a, 0x70a);
    add_array_candidate(ctx, region, 0, 8, 8, 0.0);
    add_extent(ctx, region, 0, 8);
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 0 && plan->discarded_layout == 1,
          "exact zero array score loses to empty schedule");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage63_invalid_geometry_atomic(void)
{
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 0x620, 0x720);
    OspreyContext *ctx = new_decode_context();
    add_array_candidate(ctx, region, 0, 8, 8, 0.9);
    add_extent(ctx, region, 0, 8);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    CHECK(build_array_input_and_roles(ctx, &input, &plan) && plan != NULL,
          "out-of-extent array fixture builds Stage 6.2 plan");
    char *before = plan == NULL ? NULL : dump_plan(plan);
    if (input != NULL && plan != NULL && input->array_count == 1) {
        input->array_candidates[0].payload.segment.a2.offset = 16;
        input->array_candidates[0].key = osprey_var_key(
            OSPREY_PRED_ARRAY, &input->array_candidates[0].payload);
        CHECK(osprey_decode_select_arrays(ctx, input, plan) ==
                  OSPREY_INVALID_MODEL,
              "out-of-extent array rejects the complete transaction");
        char *after = dump_plan(plan);
        CHECK(before != NULL && after != NULL && strcmp(before, after) == 0,
              "out-of-extent failure leaves the incoming plan unchanged");
        free(after);
    }
    free(before);
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x621, 0x721);
    add_array_candidate(ctx, region, 0, 8, 8, 0.9);
    add_extent(ctx, region, 0, 8);
    input = NULL;
    plan = NULL;
    CHECK(build_array_input_and_roles(ctx, &input, &plan) && input != NULL &&
              plan != NULL,
          "array identity fixture builds Stage 6.2 plan");
    before = plan == NULL ? NULL : dump_plan(plan);
    if (input != NULL && input->array_count == 1) {
        input->array_candidates[0].payload.segment.a2.region =
            make_region(OSPREY_REGION_HEAP_SITE, 0x621, 0x721);
        input->array_candidates[0].key = osprey_var_key(
            OSPREY_PRED_ARRAY, &input->array_candidates[0].payload);
        CHECK(osprey_decode_select_arrays(ctx, input, plan) ==
                  OSPREY_INVALID_MODEL,
              "mismatched array endpoint identity rejects atomically");
        char *after = dump_plan(plan);
        CHECK(before != NULL && after != NULL && strcmp(before, after) == 0,
              "endpoint identity failure leaves the plan unchanged");
        free(after);
    }
    free(before);
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x622, 0x722);
    add_array_candidate(ctx, region, 0, 8, 8, 0.9);
    add_extent(ctx, region, 0, 8);
    input = NULL;
    plan = NULL;
    CHECK(build_array_input_and_roles(ctx, &input, &plan) && input != NULL &&
              plan != NULL,
          "short-span array fixture builds graph input");
    before = plan == NULL ? NULL : dump_plan(plan);
    if (input != NULL && plan != NULL && input->array_count == 1) {
        input->array_candidates[0].payload.segment.a2.offset = 4;
        input->array_candidates[0].key = osprey_var_key(
            OSPREY_PRED_ARRAY, &input->array_candidates[0].payload);
        CHECK(osprey_decode_select_arrays(ctx, input, plan) ==
                  OSPREY_INVALID_MODEL,
              "span smaller than stride rejects instead of truncating");
        char *after = dump_plan(plan);
        CHECK(before != NULL && after != NULL && strcmp(before, after) == 0,
              "invalid span failure leaves the plan unchanged");
        free(after);
    }
    free(before);
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage63_membership_boundaries(void)
{
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 0x623, 0x723);
    OspreyRegionId other = make_region(OSPREY_REGION_GLOBAL, 0x624, 0x724);
    OspreyContext *ctx = new_decode_context();
    OspreyChunk inside_chunk = make_chunk(region, 0, 8);
    OspreyChunk before_chunk = make_chunk(region, -8, 8);
    OspreyChunk after_chunk = make_chunk(region, 8, 8);
    OspreyChunk other_chunk = make_chunk(other, 0, 8);
    uint32_t inside = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                    inside_chunk);
    uint32_t before = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                    before_chunk);
    uint32_t after = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                   after_chunk);
    uint32_t other_id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                      other_chunk);
    set_belief(ctx, inside, 0.8);
    set_belief(ctx, before, 0.8);
    set_belief(ctx, after, 0.8);
    set_belief(ctx, other_id, 0.8);
    add_array_candidate(ctx, region, 0, 8, 8, 1.0);
    add_extent(ctx, region, -8, 16);
    add_extent(ctx, other, 0, 16);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->arrays[0].member_count == 1,
          "aligned membership ignores half-open boundaries and other regions");
    if (plan != NULL) {
        const OspreyChunkDecision *inside_decision =
            plan_find_decision(plan, &inside_chunk);
        const OspreyChunkDecision *before_decision =
            plan_find_decision(plan, &before_chunk);
        const OspreyChunkDecision *after_decision =
            plan_find_decision(plan, &after_chunk);
        const OspreyChunkDecision *other_decision =
            plan_find_decision(plan, &other_chunk);
        CHECK(inside_decision != NULL &&
                  inside_decision->final_role == OSPREY_STORAGE_ARRAY_ELEMENT,
              "aligned boundary fixture retains its inside decision");
        CHECK(before_decision != NULL &&
                  before_decision->final_role == OSPREY_STORAGE_PRIMITIVE &&
                  after_decision != NULL &&
                  after_decision->final_role == OSPREY_STORAGE_PRIMITIVE &&
                  other_decision != NULL &&
                  other_decision->final_role == OSPREY_STORAGE_PRIMITIVE,
              "adjacent and same-offset other-region chunks stay independent");
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x625, 0x725);
    uint32_t unaligned = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                       make_chunk(region, 4, 4));
    set_belief(ctx, unaligned, 0.8);
    add_array_candidate(ctx, region, 0, 16, 8, 0.9);
    add_extent(ctx, region, 0, 16);
    input = NULL;
    plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 0 && plan->discarded_layout == 1 &&
              plan->decisions[0].final_role == OSPREY_STORAGE_PRIMITIVE,
          "unaligned overlap is a hard array conflict");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x626, 0x726);
    uint32_t oversized = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                       make_chunk(region, 0, 16));
    set_belief(ctx, oversized, 0.8);
    add_array_candidate(ctx, region, 0, 8, 8, 0.9);
    add_extent(ctx, region, 0, 16);
    input = NULL;
    plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 0 && plan->discarded_layout == 1,
          "oversized overlap is excluded as a hard conflict");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x632, 0x732);
    uint32_t crossing = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                      make_chunk(region, 4, 8));
    set_belief(ctx, crossing, 0.8);
    add_array_candidate(ctx, region, 0, 8, 8, 0.9);
    add_extent(ctx, region, 0, 16);
    input = NULL;
    plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 0 && plan->discarded_layout == 1,
          "crossing-end overlap is a hard array conflict");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage63_overlap_and_ties(void)
{
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 0x627, 0x727);
    OspreyContext *ctx = new_decode_context();
    add_array_candidate(ctx, region, 0, 16, 8, 0.8);
    add_array_candidate(ctx, region, 0, 8, 8, 0.8);
    add_extent(ctx, region, 0, 16);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->arrays[0].hi == 8 &&
              plan->discarded_layout == 1,
          "same-stride overlap selects the canonical winning interval");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_STACK_FUNCTION, 0x635, 0x735);
    add_array_candidate(ctx, region, -8, 8, 8, 0.8);
    add_array_candidate(ctx, region, 0, 8, 8, 0.8);
    add_extent(ctx, region, -16, 16);
    input = NULL;
    plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->arrays[0].lo == -8,
          "array ties use signed canonical endpoint order");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x628, 0x728);
    add_array_candidate(ctx, region, 0, 8, 8, 0.8);
    add_array_candidate(ctx, region, 0, 8, 4, 0.8);
    add_extent(ctx, region, 0, 8);
    input = NULL;
    plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->arrays[0].stride == 4,
          "exact cross-stride score ties use the complete key order");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    ctx->config.report_threshold = 0.5;
    region = make_region(OSPREY_REGION_GLOBAL, 0x634, 0x734);
    add_array_candidate(ctx, region, 0, 8, 8, 0.5);
    add_array_candidate(ctx, region, 8, 16, 8, 0.9);
    add_extent(ctx, region, 0, 16);
    input = NULL;
    plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 2 && plan->arrays[0].lo == 0 &&
              plan->arrays[1].lo == 8,
          "global key tie retains a zero-score canonical prefix");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage63_adjusted_scores(void)
{
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 0x629, 0x729);
    OspreyContext *ctx = new_decode_context();
    OspreyChunk chunk = make_chunk(region, 0, 8);
    uint32_t scalar = add_chunk_var(ctx, OSPREY_PRED_SCALAR, chunk);
    set_belief(ctx, scalar, 0.6);
    add_array_candidate(ctx, region, 0, 8, 8, 0.9);
    add_extent(ctx, region, 0, 8);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->arrays[0].displacement_count == 1 &&
              plan->decisions[0].final_role == OSPREY_STORAGE_ARRAY_ELEMENT,
          "scalar displacement contributes one adjusted-score penalty");
    if (plan != NULL) {
        OspreyKey scalar_key = osprey_var_key(
            OSPREY_PRED_SCALAR, &(OspreyVarPayload){ .chunk = chunk });
        CHECK(memcmp(&plan->arrays[0].displacement_keys[0], &scalar_key,
                     sizeof(scalar_key)) == 0 &&
                  plan_has_role_loss(plan, &scalar_key),
              "scalar displacement key and role loss are canonical");
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x62a, 0x72a);
    chunk = make_chunk(region, 0, 8);
    add_field_candidate(ctx, chunk, make_address(region, 0), 0.6);
    add_array_candidate(ctx, region, 0, 8, 8, 0.9);
    add_extent(ctx, region, 0, 8);
    input = NULL;
    plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->arrays[0].displacement_count == 1 &&
              plan->field_group_count == 0,
          "field displacement contributes one adjusted-score penalty");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x62b, 0x72b);
    chunk = make_chunk(region, 0, 8);
    uint32_t losing_scalar = add_chunk_var(ctx, OSPREY_PRED_SCALAR, chunk);
    set_belief(ctx, losing_scalar, 0.9);
    add_array_candidate(ctx, region, 0, 8, 8, 0.7);
    add_extent(ctx, region, 0, 8);
    input = NULL;
    plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 0 && plan->discarded_layout == 1 &&
              plan->decisions[0].final_role == OSPREY_STORAGE_SCALAR,
          "negative adjusted score loses to the empty schedule");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage63_score_boundaries(void)
{
    OspreyDecodeScore score = { .infinity_balance = INT64_MAX };
    OspreyDecodeScore positive_one = { .infinity_balance = 1 };
    CHECK(!osprey_decode_score_add(&score, &positive_one) &&
              score.infinity_balance == INT64_MAX && score.finite == 0.0,
          "infinity-balance addition is checked at INT64_MAX");
    score = (OspreyDecodeScore){ .infinity_balance = INT64_MIN };
    OspreyDecodeScore negative_one = { .infinity_balance = -1 };
    CHECK(!osprey_decode_score_add(&score, &negative_one) &&
              score.infinity_balance == INT64_MIN,
          "infinity-balance addition is checked at INT64_MIN");
    score = (OspreyDecodeScore){ 0 };
    OspreyDecodeScore negative_infinite = { .negative_infinite = 1 };
    CHECK(osprey_decode_score_add(&score, &negative_infinite) &&
              osprey_decode_score_compare(&score, &(OspreyDecodeScore){ 0 }) < 0,
          "negative-infinite score remains below the empty schedule");

    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 0x62c, 0x72c);
    OspreyContext *ctx = new_decode_context();
    OspreyChunk chunk = make_chunk(region, 0, 8);
    uint32_t scalar = add_chunk_var(ctx, OSPREY_PRED_SCALAR, chunk);
    set_belief(ctx, scalar, 1.0);
    add_array_candidate(ctx, region, 0, 8, 8, 1.0);
    add_extent(ctx, region, 0, 8);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 0 && plan->discarded_layout == 1 &&
              plan->decisions[0].final_role == OSPREY_STORAGE_SCALAR,
          "exact one array and exact one displacement cancel to zero");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage63_residual_order(void)
{
    OspreyRegionId region = make_region(OSPREY_REGION_STACK_FUNCTION,
                                         0x62d, 0x72d);
    OspreyContext *ctx = new_decode_context();
    ctx->config.report_threshold = 0.5;
    OspreyChunk chunks[] = {
        make_chunk(region, 0, 8), make_chunk(region, 8, 8),
        make_chunk(region, 16, 8),
    };
    const double probabilities[] = {
        0.500000000000001, 0.9, 0.500000000000002,
    };
    uint32_t ids[3];
    for (size_t i = G_N_ELEMENTS(ids); i-- > 0;) {
        ids[i] = add_chunk_var(ctx, OSPREY_PRED_SCALAR, chunks[i]);
        set_belief(ctx, ids[i], probabilities[i]);
    }
    add_array_candidate(ctx, region, 0, 24, 8, 0.999);
    add_extent(ctx, region, -8, 32);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->arrays[0].displacement_count == 3,
          "residual-order fixture selects all aligned scalar members");
    if (plan != NULL && plan->array_count == 1) {
        OspreyKey expected_keys[3];
        double expected = log(0.999) - log1p(-0.999);
        for (size_t i = 0; i < G_N_ELEMENTS(probabilities); i++) {
            expected -= log(probabilities[i]) - log1p(-probabilities[i]);
            expected_keys[i] = osprey_var_key(
                OSPREY_PRED_SCALAR,
                &(OspreyVarPayload){ .chunk = chunks[i] });
        }
        CHECK(memcmp(&plan->arrays[0].displacement_keys[0],
                     &expected_keys[0], sizeof(expected_keys[0])) == 0 &&
                  memcmp(&plan->arrays[0].displacement_keys[1],
                         &expected_keys[1], sizeof(expected_keys[1])) == 0 &&
                  memcmp(&plan->arrays[0].displacement_keys[2],
                         &expected_keys[2], sizeof(expected_keys[2])) == 0,
              "displacement terms are ordered by complete chunk key");
        CHECK(plan->arrays[0].adjusted_score.finite == expected,
              "finite adjusted score uses the specified binary64 order");
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage63_signed_endpoints(void)
{
    OspreyRegionId region = make_region(OSPREY_REGION_STACK_FUNCTION,
                                         0x633, 0x733);
    OspreyContext *ctx = new_decode_context();
    add_array_candidate(ctx, region, INT64_MIN + 8, INT64_MIN + 24, 8, 0.9);
    add_extent(ctx, region, INT64_MIN, INT64_MIN + 32);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->arrays[0].lo == INT64_MIN + 8 &&
              plan->arrays[0].hi == INT64_MIN + 24 &&
              plan->arrays[0].count == 2,
          "negative stack endpoints preserve checked signed geometry");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static void test_stage63_evidence_and_dump(void)
{
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 0x62e, 0x72e);
    OspreyContext *ctx = new_decode_context();
    uint32_t array = add_array_candidate(ctx, region, 0, 16, 8, 0.9);
    OspreyVar *array_var = &g_array_index(ctx->graph->vars, OspreyVar, array);
    array_var->direct_support = 41;
    array_var->source_rule_bits = UINT64_C(0x1234);
    add_extent(ctx, region, 0, 16);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->arrays[0].member_count == 0 &&
              plan->arrays[0].direct_support == 41 &&
              plan->arrays[0].source_rule_bits == UINT64_C(0x1234),
          "unobserved arrays retain exact geometry and evidence");
    if (plan != NULL) {
        char *dump = dump_plan(plan);
        CHECK(dump != NULL && strstr(dump, "[key ") != NULL &&
                  strstr(dump, "[members]") != NULL &&
                  strstr(dump, "[score-finite-bits ") != NULL &&
                  strstr(dump, "graph") == NULL && strstr(dump, "0x") != NULL,
              "array plan dump contains canonical fields without graph IDs");
        free(dump);
    }
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);

    ctx = new_decode_context();
    region = make_region(OSPREY_REGION_GLOBAL, 0x62f, 0x72f);
    OspreyChunk chunk = make_chunk(region, 0, 8);
    uint32_t primitive = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR, chunk);
    array = add_array_candidate(ctx, region, 0, 8, 8, 0.9);
    OspreyVar *primitive_var = &g_array_index(ctx->graph->vars,
                                              OspreyVar, primitive);
    array_var = &g_array_index(ctx->graph->vars, OspreyVar, array);
    primitive_var->direct_support = 17;
    primitive_var->source_rule_bits = UINT64_C(0x22);
    array_var->direct_support = 19;
    array_var->source_rule_bits = UINT64_C(0x44);
    set_belief(ctx, primitive, 0.8);
    add_extent(ctx, region, 0, 8);
    input = NULL;
    plan = NULL;
    CHECK(build_array_plan(ctx, &input, &plan) && plan != NULL &&
              plan->array_count == 1 && plan->decisions[0].has_array_owner &&
              plan->decisions[0].array_support == 19 &&
              plan->decisions[0].array_source_rule_bits == UINT64_C(0x44) &&
              plan->decisions[0].array_posterior == 0.9,
          "selected array and member evidence propagate independently");
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

static OspreyContext *make_array_permutation_context(unsigned order)
{
    static const unsigned permutations[][5] = {
        { 0, 1, 2, 3, 4 }, { 4, 3, 2, 1, 0 }, { 2, 4, 0, 3, 1 },
    };
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 0x630, 0x730);
    OspreyChunk chunks[] = {
        make_chunk(region, 0, 8), make_chunk(region, 8, 8),
    };
    OspreyContext *ctx = new_decode_context();
    unsigned permutation = order % G_N_ELEMENTS(permutations);
    for (size_t i = 0; i < G_N_ELEMENTS(permutations[0]); i++) {
        switch (permutations[permutation][i]) {
        case 0: {
            uint32_t id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                        chunks[0]);
            set_belief(ctx, id, 0.8);
            break;
        }
        case 1: {
            uint32_t id = add_chunk_var(ctx, OSPREY_PRED_PRIMITIVE_VAR,
                                        chunks[1]);
            set_belief(ctx, id, 0.8);
            break;
        }
        case 2:
            add_array_candidate(ctx, region, 0, 16, 8, 0.8);
            break;
        case 3:
            add_array_candidate(ctx, region, 0, 8, 8, 0.9);
            break;
        case 4:
            add_array_candidate(ctx, region, 8, 16, 8, 0.9);
            break;
        default:
            break;
        }
    }
    add_extent(ctx, region, 0, 16);
    return ctx;
}

static void test_stage63_permutations(void)
{
    OspreyContext *contexts[3] = {
        make_array_permutation_context(0),
        make_array_permutation_context(1),
        make_array_permutation_context(2),
    };
    OspreyDecodeInput *inputs[3] = { NULL, NULL, NULL };
    OspreyDecodePlan *plans[3] = { NULL, NULL, NULL };
    char *dumps[3] = { NULL, NULL, NULL };
    bool built = true;
    for (size_t i = 0; i < G_N_ELEMENTS(contexts); i++) {
        built = built && build_array_plan(contexts[i], &inputs[i], &plans[i]);
        dumps[i] = plans[i] == NULL ? NULL : dump_plan(plans[i]);
    }
    CHECK(built && dumps[0] != NULL && dumps[1] != NULL && dumps[2] != NULL &&
              strcmp(dumps[0], dumps[1]) == 0 &&
              strcmp(dumps[0], dumps[2]) == 0,
          "array schedules and diagnostics ignore insertion order");
    for (size_t i = 0; i < G_N_ELEMENTS(contexts); i++) {
        free(dumps[i]);
        osprey_decode_plan_free(plans[i]);
        osprey_decode_input_free(inputs[i]);
        osprey_free(contexts[i]);
    }
}

static void test_stage63_allocation_atomicity(void)
{
    OspreyRegionId region = make_region(OSPREY_REGION_GLOBAL, 0x631, 0x731);
    OspreyContext *ctx = new_decode_context();
    OspreyChunk member = make_chunk(region, 0, 8);
    OspreyChunk survivor = make_chunk(region, 16, 8);
    uint32_t member_id = add_chunk_var(ctx, OSPREY_PRED_SCALAR, member);
    uint32_t survivor_id = add_chunk_var(ctx, OSPREY_PRED_SCALAR, survivor);
    set_belief(ctx, member_id, 0.7);
    set_belief(ctx, survivor_id, 0.8);
    add_array_candidate(ctx, region, 0, 8, 8, 1.0);
    add_field_candidate(ctx, survivor, make_address(region, 0), 0.9);
    add_extent(ctx, region, 0, 32);
    OspreyDecodeInput *input = NULL;
    OspreyDecodePlan *plan = NULL;
    osprey_decode_test_set_alloc_fail_after(-1);
    CHECK(build_array_input_and_roles(ctx, &input, &plan) && plan != NULL,
          "array allocation sweep fixture builds Stage 6.2 plan");
    char *before = plan == NULL ? NULL : dump_plan(plan);
    bool saw_success = false;
    for (int64_t failure = 0; failure < 512; failure++) {
        osprey_decode_test_set_alloc_fail_after(failure);
        OspreyStatus status = osprey_decode_select_arrays(ctx, input, plan);
        if (status == OSPREY_OK) {
            saw_success = true;
            break;
        }
        CHECK(status == OSPREY_INVALID_MODEL,
              "array allocation failure returns invalid model");
        char *after = dump_plan(plan);
        CHECK(before != NULL && after != NULL && strcmp(before, after) == 0,
              "array allocation failure leaves the plan byte-equivalent");
        free(after);
    }
    osprey_decode_test_set_alloc_fail_after(-1);
    CHECK(saw_success, "array allocation sweep reaches normal allocation");
    free(before);
    osprey_decode_plan_free(plan);
    osprey_decode_input_free(input);
    osprey_free(ctx);
}

int main(void)
{
    RUN(test_valid_projection_and_indexes);
    RUN(test_fixed_canonical_dump);
    RUN(test_region_identity_collisions);
    RUN(test_projection_permutation);
    RUN(test_filter_threshold_and_hard_false);
    RUN(test_graph_boundary_rejections);
    RUN(test_exact_belief_boundaries);
    RUN(test_invalid_belief_values);
    RUN(test_variable_identity_rejections);
    RUN(test_hard_false_contract_rejections);
    RUN(test_payload_and_extent_rejections);
    RUN(test_extent_catalog_normalization);
    RUN(test_unbuilt_extent_catalog_is_rejected);
    RUN(test_checked_payload_arithmetic);
    RUN(test_out_of_extent_candidates_are_retained);
    RUN(test_array_region_view_key_order);
    RUN(test_rule_stage_mismatch_is_rejected);
    RUN(test_transaction_is_unchanged);
    RUN(test_failure_leaves_transaction_unchanged);
    RUN(test_input_owns_source_data);
    RUN(test_null_and_missing_inputs);
    RUN(test_repeated_build_and_free);
    RUN(test_count_and_limit_boundaries);
    RUN(test_allocation_failures);
    RUN(test_threshold_configuration);
    RUN(test_stage62_orthogonal_roles);
    RUN(test_stage62_baseline_role_matrix);
    RUN(test_stage62_field_selection);
    RUN(test_stage62_signed_canonical_order);
    RUN(test_stage62_checked_field_end);
    RUN(test_stage62_rejects_selected_geometry);
    RUN(test_stage62_rejects_noncanonical_slices);
    RUN(test_stage62_rejects_array_candidate_alias);
    RUN(test_stage62_plan_permutation_and_failures);
    RUN(test_stage63_arrays_and_roles);
    RUN(test_stage63_invalid_geometry_atomic);
    RUN(test_stage63_membership_boundaries);
    RUN(test_stage63_overlap_and_ties);
    RUN(test_stage63_adjusted_scores);
    RUN(test_stage63_score_boundaries);
    RUN(test_stage63_residual_order);
    RUN(test_stage63_signed_endpoints);
    RUN(test_stage63_evidence_and_dump);
    RUN(test_stage63_permutations);
    RUN(test_stage63_allocation_atomicity);
    fprintf(stderr, "stage6.3: %u/%u tests passed\n", executed - failures,
            registered);
    return failures == 0 ? 0 : 1;
}
