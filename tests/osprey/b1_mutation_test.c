#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "qemu/osdep.h"
#include "osprey.h"
#include "osprey-internal.h"

static unsigned checks;
static unsigned failures;

#define CHECK(_condition, _message) do { \
    checks++; \
    if (!(_condition)) { \
        failures++; \
        fprintf(stderr, "FAIL: %s\n", (_message)); \
    } \
} while (0)

static OspreyConfig test_config(void)
{
    OspreyConfig config;
    memset(&config, 0, sizeof(config));
    config.enabled = true;
    config.analysis_mode = OSPREY_ANALYSIS_MODE_MUTATION;
    config.shared_bytes = 1u << 20;
    config.max_facts = 1024;
    config.max_chunks_per_region = 128;
    config.max_candidates_per_kind_region = 128;
    config.max_variables = 1024;
    config.max_factors = 2048;
    config.max_exact_clique_vars = 20;
    config.report_threshold = 0.5;
    return config;
}

static OspreyRegionId region(OspreyRegionKind kind, uint64_t site,
                             uint64_t image)
{
    OspreyRegionId result;
    memset(&result, 0, sizeof(result));
    result.kind = kind;
    result.site_offset = site;
    result.code_image_id = image;
    return result;
}

static OspreyAddress address(OspreyRegionId region_id, int64_t offset)
{
    OspreyAddress result;
    result.region = region_id;
    result.offset = offset;
    return result;
}

static OspreyChunk chunk(OspreyRegionId region_id, int64_t offset,
                         uint64_t size)
{
    OspreyChunk result;
    result.address = address(region_id, offset);
    result.size = size;
    return result;
}

static void add_point(OspreyContext *ctx, OspreyChunk cell,
                      OspreyAddress target, uint32_t support)
{
    OspreyPointsToFact fact;
    memset(&fact, 0, sizeof(fact));
    fact.pointer_chunk = cell;
    fact.target = target;
    fact.sample_support = support;
    g_array_append_val(ctx->points_facts, fact);
}

static void add_access(OspreyContext *ctx, OspreyChunk value)
{
    OspreyAccessFact fact;
    memset(&fact, 0, sizeof(fact));
    fact.pc = 0x401000;
    fact.chunk = value;
    fact.dynamic_count = 1;
    fact.sample_support = 1;
    fact.op_class = 1;
    g_array_append_val(ctx->access_facts, fact);
}

static void add_base(OspreyContext *ctx, OspreyAddress base,
                     OspreyChunk field, uint64_t pc)
{
    OspreyBaseFact fact;
    memset(&fact, 0, sizeof(fact));
    fact.pc = pc;
    fact.chunk = field;
    fact.base = base;
    fact.sample_support = 1;
    g_array_append_val(ctx->base_facts, fact);
}

static void add_alloc(OspreyContext *ctx, uint64_t site, uint64_t size)
{
    OspreyMallocFact fact;
    memset(&fact, 0, sizeof(fact));
    fact.site_pc = site;
    fact.requested_size = size;
    fact.sample_support = 1;
    g_array_append_val(ctx->alloc_facts, fact);
}

static void add_array(OspreyContext *ctx, OspreyAddress start,
                      uint64_t count, uint64_t element_size)
{
    OspreyMayArrayFact fact;
    memset(&fact, 0, sizeof(fact));
    fact.start = start;
    fact.element_count = count;
    fact.element_size = element_size;
    fact.evidence_kind = OSPREY_MAY_ARRAY_CALLOC_GEOMETRY;
    fact.sample_support = 1;
    g_array_append_val(ctx->mayarray_facts, fact);
}

static OspreyContext *new_context(void)
{
    OspreyConfig config = test_config();
    OspreyContext *ctx = osprey_new(&config);
    CHECK(ctx != NULL, "compact context allocation");
    if (ctx != NULL) osprey_tx_begin(ctx);
    return ctx;
}

static void test_unique_array(void)
{
    OspreyRegionId cell_region = region(OSPREY_REGION_GLOBAL, 0, 7);
    OspreyRegionId heap_region = region(OSPREY_REGION_HEAP_SITE, 0x100, 0);
    OspreyChunk cell = chunk(cell_region, 0x20, sizeof(target_ulong));
    OspreyAddress target = address(heap_region, 0);
    OspreyContext *ctx = new_context();
    const OspreyMutationModel *model;
    const OspreyMutationEntry *entry;

    add_access(ctx, cell);
    add_point(ctx, cell, target, 3);
    add_alloc(ctx, 0x100, 16);
    add_array(ctx, target, 2, 8);
    CHECK(osprey_mutation_model_build(ctx) == OSPREY_OK,
          "unique array compact build");
    model = osprey_mutation_model(ctx);
    CHECK(model != NULL && model->entry_count == 1,
          "unique array publishes one entry");
    entry = model == NULL ? NULL : osprey_mutation_lookup(model, &cell);
    CHECK(entry != NULL && entry->kind == OSPREY_MUTATION_AGGREGATE_ARRAY,
          "unique array kind is explicit");
    CHECK(entry != NULL && entry->target_base.region.code_image_id == 0 &&
              entry->extent == 16 && entry->support == 3 && entry->ordinal == 0,
          "unique array keeps exact target, extent, and support");
    CHECK(model != NULL && model->stats.eligible_cells == 1 &&
              model->stats.published_entries == 1,
          "unique array reports compact counters");
    osprey_free(ctx);
}

static void test_unique_struct(void)
{
    OspreyRegionId cell_region = region(OSPREY_REGION_GLOBAL, 0, 1);
    OspreyRegionId target_region = region(OSPREY_REGION_GLOBAL, 0x50, 9);
    OspreyChunk cell = chunk(cell_region, 0x40, sizeof(target_ulong));
    OspreyAddress target = address(target_region, 0x10);
    OspreyContext *ctx = new_context();
    const OspreyMutationModel *model;
    const OspreyMutationEntry *entry;

    add_point(ctx, cell, target, 2);
    add_base(ctx, target, chunk(target_region, 0x10, 4), 0x10);
    add_base(ctx, target, chunk(target_region, 0x18, 8), 0x20);
    CHECK(osprey_mutation_model_build(ctx) == OSPREY_OK,
          "unique struct compact build");
    model = osprey_mutation_model(ctx);
    entry = model == NULL ? NULL : osprey_mutation_lookup(model, &cell);
    CHECK(entry != NULL && entry->kind == OSPREY_MUTATION_AGGREGATE_STRUCT,
          "two exact F02 fields create a struct");
    CHECK(entry != NULL && entry->extent == 16 &&
              entry->target_base.region.code_image_id == 9,
          "struct envelope preserves full canonical target identity");
    osprey_free(ctx);
}

static void test_target_tie_and_local_abstention(void)
{
    OspreyRegionId cell_region = region(OSPREY_REGION_GLOBAL, 0, 0);
    OspreyRegionId target_region = region(OSPREY_REGION_GLOBAL, 0x70, 0);
    OspreyChunk cell = chunk(cell_region, 0x60, sizeof(target_ulong));
    OspreyAddress first = address(target_region, 0);
    OspreyAddress second = address(target_region, 0x40);
    OspreyContext *ctx = new_context();
    const OspreyMutationModel *model;

    add_point(ctx, cell, first, 2);
    add_point(ctx, cell, second, 2);
    add_base(ctx, first, chunk(target_region, 0, 4), 1);
    add_base(ctx, first, chunk(target_region, 4, 4), 2);
    add_base(ctx, second, chunk(target_region, 0x40, 4), 3);
    add_base(ctx, second, chunk(target_region, 0x44, 4), 4);
    CHECK(osprey_mutation_model_build(ctx) == OSPREY_OK,
          "target tie remains a successful transaction");
    model = osprey_mutation_model(ctx);
    CHECK(model != NULL && model->entry_count == 0 &&
              model->stats.abstentions[OSPREY_MUTATION_ABSTAIN_TARGET_TIE] == 1,
          "equal direct F04 support abstains locally");
    CHECK(ctx->tx_status == OSPREY_OK,
          "local target abstention does not reject transaction");
    osprey_free(ctx);
}

static void test_conflicting_geometry_and_fallback(void)
{
    OspreyRegionId cell_region = region(OSPREY_REGION_GLOBAL, 0, 0);
    OspreyRegionId target_region = region(OSPREY_REGION_HEAP_SITE, 0x180, 0);
    OspreyChunk cell = chunk(cell_region, 0x80, sizeof(target_ulong));
    OspreyAddress target = address(target_region, 0);
    OspreyContext *ctx = new_context();
    const OspreyMutationModel *model;

    add_point(ctx, cell, target, 1);
    add_alloc(ctx, 0x180, 32);
    add_array(ctx, target, 2, 8);
    add_array(ctx, target, 4, 4);
    CHECK(osprey_mutation_model_build(ctx) == OSPREY_OK,
          "conflicting geometry keeps transaction live");
    model = osprey_mutation_model(ctx);
    CHECK(model != NULL && model->entry_count == 0 &&
              model->stats.abstentions[
                  OSPREY_MUTATION_ABSTAIN_GEOMETRY_CONFLICT] == 1,
          "conflicting F06 geometry abstains without a prefix");
    osprey_free(ctx);
}

static void test_budget_rejection_hides_model(void)
{
    OspreyRegionId region_id = region(OSPREY_REGION_GLOBAL, 0, 0);
    OspreyChunk cell = chunk(region_id, 0, sizeof(target_ulong));
    OspreyAddress target = address(region_id, 16);
    OspreyConfig config = test_config();
    OspreyContext *ctx;

    config.max_mutation_input = 1;
    ctx = osprey_new(&config);
    osprey_tx_begin(ctx);
    add_access(ctx, cell);
    add_point(ctx, cell, target, 1);
    CHECK(osprey_mutation_model_build(ctx) == OSPREY_LIMIT_EXCEEDED,
          "input cap rejects before compact allocation");
    CHECK(osprey_mutation_model(ctx) == NULL &&
              strcmp(ctx->tx_reason, "mutation-input-cap") == 0,
          "input cap hides model with stable reason");
    osprey_free(ctx);

    config = test_config();
    config.max_mutation_bytes = 1;
    ctx = osprey_new(&config);
    osprey_tx_begin(ctx);
    add_point(ctx, cell, target, 1);
    CHECK(osprey_mutation_model_build(ctx) == OSPREY_LIMIT_EXCEEDED,
          "owned-byte cap rejects compact staging");
    CHECK(osprey_mutation_model(ctx) == NULL &&
              strcmp(ctx->tx_reason, "mutation-bytes-cap") == 0,
          "owned-byte cap has stable reason");
    osprey_free(ctx);
}

static void test_malformed_fact_rejects(void)
{
    OspreyRegionId region_id = region(OSPREY_REGION_GLOBAL, 0, 0);
    OspreyConfig config = test_config();
    OspreyContext *ctx = osprey_new(&config);
    OspreyPointsToFact fact;

    osprey_tx_begin(ctx);
    memset(&fact, 0, sizeof(fact));
    fact.pointer_chunk = chunk(region_id, 0, 4);
    fact.target = address(region_id, 8);
    fact.sample_support = 1;
    g_array_append_val(ctx->points_facts, fact);
    CHECK(osprey_mutation_model_build(ctx) == OSPREY_LIMIT_EXCEEDED,
          "wrong-width F04 rejects aggregate input");
    CHECK(osprey_mutation_model(ctx) == NULL &&
              strcmp(ctx->tx_reason, "mutation-malformed-input") == 0,
          "malformed input reports stable rejection reason");
    osprey_free(ctx);

    config = test_config();
    ctx = osprey_new(&config);
    osprey_tx_begin(ctx);
    OspreyChunk cell = chunk(region_id, 0x18, sizeof(target_ulong));
    OspreyAddress target = address(region_id, 0x40);
    add_point(ctx, cell, target, 1);
    add_array(ctx, target, UINT64_MAX, 2);
    CHECK(osprey_mutation_model_build(ctx) == OSPREY_LIMIT_EXCEEDED,
          "overflowing F06 rejects aggregate input");
    CHECK(osprey_mutation_model(ctx) == NULL &&
              strcmp(ctx->tx_reason, "mutation-malformed-input") == 0,
          "overflowing F06 has no partial publication");
    osprey_free(ctx);
}

static void test_cell_local_support(void)
{
    OspreyContext *ctx = new_context();
    OspreyRegionId cell_region = region(OSPREY_REGION_GLOBAL, 0, 0);
    OspreyRegionId array_region = region(OSPREY_REGION_GLOBAL, 0x100, 0);
    OspreyRegionId struct_region = region(OSPREY_REGION_GLOBAL, 0x200, 0);
    OspreyChunk cell_a = chunk(cell_region, 0x10, sizeof(target_ulong));
    OspreyChunk cell_b = chunk(cell_region, 0x18, sizeof(target_ulong));
    OspreyAddress array_target = address(array_region, 0);
    OspreyAddress struct_target = address(struct_region, 0);
    const OspreyMutationModel *model;
    const OspreyMutationEntry *entry_a;
    const OspreyMutationEntry *entry_b;

    add_point(ctx, cell_a, array_target, 1);
    add_point(ctx, cell_a, struct_target, 2);
    add_point(ctx, cell_b, array_target, 100);
    add_array(ctx, array_target, 2, 8);
    add_base(ctx, struct_target, chunk(struct_region, 0, 4), 1);
    add_base(ctx, struct_target, chunk(struct_region, 8, 4), 2);
    CHECK(osprey_mutation_model_build(ctx) == OSPREY_OK,
          "cell-local support fixture builds");
    model = osprey_mutation_model(ctx);
    entry_a = model == NULL ? NULL : osprey_mutation_lookup(model, &cell_a);
    entry_b = model == NULL ? NULL : osprey_mutation_lookup(model, &cell_b);
    CHECK(entry_a != NULL && entry_a->target_base.region.site_offset == 0x200 &&
              entry_a->support == 2,
          "cell-local support selects the local maximum");
    CHECK(entry_b != NULL && entry_b->target_base.region.site_offset == 0x100 &&
              entry_b->support == 100,
          "another cell cannot change local target selection");
    osprey_free(ctx);
}

static OspreyContext *permutation_context(bool reverse)
{
    OspreyContext *ctx = new_context();
    OspreyRegionId cell_region = region(OSPREY_REGION_GLOBAL, 0, 0);
    OspreyRegionId array_region = region(OSPREY_REGION_HEAP_SITE, 0x200, 0);
    OspreyRegionId struct_region = region(OSPREY_REGION_GLOBAL, 0x80, 2);
    OspreyChunk cell_a = chunk(cell_region, 0x20, sizeof(target_ulong));
    OspreyChunk cell_b = chunk(cell_region, 0x28, sizeof(target_ulong));
    OspreyAddress array_target = address(array_region, 0);
    OspreyAddress struct_target = address(struct_region, 0);
    if (!reverse) {
        add_point(ctx, cell_a, array_target, 3);
        add_point(ctx, cell_b, struct_target, 2);
        add_alloc(ctx, 0x200, 16);
        add_array(ctx, array_target, 2, 8);
        add_base(ctx, struct_target, chunk(struct_region, 0, 4), 2);
        add_base(ctx, struct_target, chunk(struct_region, 8, 8), 1);
    } else {
        add_base(ctx, struct_target, chunk(struct_region, 8, 8), 1);
        add_base(ctx, struct_target, chunk(struct_region, 0, 4), 2);
        add_array(ctx, array_target, 2, 8);
        add_alloc(ctx, 0x200, 16);
        add_point(ctx, cell_b, struct_target, 2);
        add_point(ctx, cell_a, array_target, 3);
    }
    return ctx;
}

static void test_permutation_independence(void)
{
    OspreyContext *forward = permutation_context(false);
    OspreyContext *reverse = permutation_context(true);
    const OspreyMutationModel *a;
    const OspreyMutationModel *b;
    CHECK(osprey_mutation_model_build(forward) == OSPREY_OK &&
              osprey_mutation_model_build(reverse) == OSPREY_OK,
          "permuted compact fixtures build");
    a = osprey_mutation_model(forward);
    b = osprey_mutation_model(reverse);
    CHECK(a != NULL && b != NULL && a->entry_count == b->entry_count &&
              a->stats.work_units == b->stats.work_units &&
              a->stats.owned_bytes == b->stats.owned_bytes,
          "permuted compact work and counts are identical");
    CHECK(a != NULL && b != NULL && a->entry_count == 2 &&
              memcmp(a->entries, b->entries,
                     a->entry_count * sizeof(*a->entries)) == 0,
          "permuted compact entries are byte-identical");
    osprey_free(forward);
    osprey_free(reverse);
}

static void test_work_budget_rejects(void)
{
    OspreyConfig config = test_config();
    OspreyContext *ctx;
    OspreyRegionId r = region(OSPREY_REGION_GLOBAL, 0, 0);
    config.max_mutation_work = 1;
    ctx = osprey_new(&config);
    osprey_tx_begin(ctx);
    add_point(ctx, chunk(r, 0, sizeof(target_ulong)), address(r, 8), 1);
    CHECK(osprey_mutation_model_build(ctx) == OSPREY_LIMIT_EXCEEDED,
          "compact work budget rejects before staging");
    CHECK(ctx->tx_reason != NULL &&
              strcmp(ctx->tx_reason, "mutation-work-budget") == 0,
          "compact work budget reason is stable");
    osprey_free(ctx);
}

static void test_publication_validation(void)
{
    OspreyContext *ctx = new_context();
    OspreyRegionId r = region(OSPREY_REGION_GLOBAL, 0, 0);
    OspreyChunk cell = chunk(r, 0, sizeof(target_ulong));
    OspreyAddress target = address(r, 16);
    OspreyMutationModel *owned;
    add_point(ctx, cell, target, 1);
    add_base(ctx, target, chunk(r, 16, 4), 1);
    add_base(ctx, target, chunk(r, 24, 4), 2);
    CHECK(osprey_mutation_model_build(ctx) == OSPREY_OK,
          "publication validation fixture builds");
    owned = ctx->mutation_model;
    CHECK(owned != NULL && osprey_mutation_model_validate(owned),
          "published compact model validates");
    if (owned != NULL) {
        owned->publication_valid = 0;
        CHECK(!osprey_mutation_model_validate(owned),
              "publication bit gates compact model visibility");
        owned->publication_valid = 1;
    }
    osprey_free(ctx);
}

int main(void)
{
    test_unique_array();
    test_unique_struct();
    test_target_tie_and_local_abstention();
    test_conflicting_geometry_and_fallback();
    test_budget_rejection_hides_model();
    test_malformed_fact_rejects();
    test_cell_local_support();
    test_permutation_independence();
    test_work_budget_rejects();
    test_publication_validation();
    printf("b1_mutation: %u/%u checks passed\n", checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
