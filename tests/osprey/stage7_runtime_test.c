/* Package C focused tests: exact compact cell/pointer runtime lookup. */

#include "osprey.h"
#include "osprey-internal.h"
#include "stage7_mutation_reference.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

static unsigned failures;
static unsigned checks;

#define CHECK(condition, message) do {                                    \
    checks++;                                                             \
    if (!(condition)) {                                                   \
        fprintf(stderr, "FAIL: %s (line %d)\n", (message), __LINE__); \
        failures++;                                                       \
    }                                                                      \
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
    config.max_candidates_per_kind_region = 4096;
    config.max_variables = 1024;
    config.max_factors = 4096;
    config.max_exact_clique_vars = 20;
    config.max_exact_table_bytes = 1u << 20;
    config.max_bp_table_bytes = 1u << 20;
    config.report_threshold = 0.6;
    return config;
}

static OspreyRegionId make_region(OspreyRegionKind kind, uint64_t site)
{
    OspreyRegionId region;
    memset(&region, 0, sizeof(region));
    region.kind = kind;
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

static bool address_equal(const OspreyAddress *a, const OspreyAddress *b)
{
    return a != NULL && b != NULL &&
           a->region.kind == b->region.kind &&
           a->region.code_image_id == b->region.code_image_id &&
           a->region.site_offset == b->region.site_offset &&
           a->offset == b->offset;
}

static bool address_ref_equal(const OspreyRuntimeAddressRef *a,
                              const OspreyRuntimeAddressRef *b)
{
    return a != NULL && b != NULL && address_equal(&a->address, &b->address) &&
           a->raw == b->raw && a->instance_id == b->instance_id &&
           a->prov_object_id == b->prov_object_id &&
           a->prov_generation == b->prov_generation &&
           a->valid == b->valid &&
           memcmp(a->reserved, b->reserved, sizeof(a->reserved)) == 0;
}

static bool cell_resolution_equal(const OspreyRuntimeCellResolution *a,
                                  const OspreyRuntimeCellResolution *b)
{
    return a->status == b->status &&
           a->entry_ordinal == b->entry_ordinal &&
           a->instance_valid == b->instance_valid &&
           a->aggregate_kind == b->aggregate_kind &&
           memcmp(a->reserved, b->reserved, sizeof(a->reserved)) == 0 &&
           a->cell.size == b->cell.size &&
           address_ref_equal(&a->cell.start, &b->cell.start) &&
           address_equal(&a->target_base, &b->target_base) &&
           a->target_extent == b->target_extent;
}

static bool pointer_resolution_equal(
    const OspreyRuntimePointerResolution *a,
    const OspreyRuntimePointerResolution *b)
{
    return a->status == b->status &&
           a->entry_ordinal == b->entry_ordinal &&
           a->target_extent == b->target_extent &&
           a->aggregate_kind == b->aggregate_kind &&
           a->has_runtime_target == b->has_runtime_target &&
           a->target_valid == b->target_valid &&
           a->reserved == b->reserved &&
           cell_resolution_equal(&a->cell, &b->cell) &&
           address_equal(&a->target_base, &b->target_base) &&
           address_ref_equal(&a->target, &b->target);
}

typedef struct ResolverInputSnapshot {
    OspreyRegionInstance *regions;
    size_t region_count;
    OspreyMutationEntry entry;
} ResolverInputSnapshot;

static ResolverInputSnapshot resolver_snapshot_take(
    const OspreyContext *ctx, const OspreyMutationModel *model)
{
    ResolverInputSnapshot snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.region_count = ctx->region_instances->len;
    if (snapshot.region_count != 0) {
        size_t bytes = snapshot.region_count * sizeof(OspreyRegionInstance);
        snapshot.regions = g_malloc(bytes);
        memcpy(snapshot.regions, ctx->region_instances->data, bytes);
    }
    if (model->entry_count == 1) snapshot.entry = model->entries[0];
    return snapshot;
}

static bool resolver_snapshot_unchanged(
    const ResolverInputSnapshot *snapshot, const OspreyContext *ctx,
    const OspreyMutationModel *model)
{
    return snapshot->region_count == ctx->region_instances->len &&
           (snapshot->region_count == 0 ||
            memcmp(snapshot->regions, ctx->region_instances->data,
                   snapshot->region_count * sizeof(OspreyRegionInstance)) == 0) &&
           model->entry_count == 1 &&
           memcmp(&snapshot->entry, model->entries,
                  sizeof(snapshot->entry)) == 0;
}

static void resolver_snapshot_free(ResolverInputSnapshot *snapshot)
{
    g_free(snapshot->regions);
    memset(snapshot, 0, sizeof(*snapshot));
}

static void append_instance(OspreyContext *ctx, OspreyRegionId region,
                            uint64_t instance_id, uint64_t raw_base,
                            uint64_t raw_min, uint64_t raw_max,
                            uint64_t object_id, uint32_t generation)
{
    OspreyRegionInstance instance;
    memset(&instance, 0, sizeof(instance));
    instance.region = region;
    instance.instance_id = instance_id;
    instance.raw_base = raw_base;
    instance.raw_min = raw_min;
    instance.raw_max = raw_max;
    instance.prov_object_id = object_id;
    instance.prov_generation = generation;
    instance.sample_support = 1;
    g_array_append_val(ctx->region_instances, instance);
}

typedef struct Stage7Fixture {
    OspreyContext *ctx;
    OspreyMutationModel model;
    OspreyMutationEntry entry;
    OspreyRegionId global;
    OspreyRegionId target_region;
    OspreyRuntimeChunkRef cell;
    OspreyRuntimeAddressRef target;
} Stage7Fixture;

static void fixture_init(Stage7Fixture *fixture, unsigned order)
{
    OspreyConfig config = test_config();
    memset(fixture, 0, sizeof(*fixture));
    fixture->ctx = osprey_new(&config);
    fixture->global = make_region(OSPREY_REGION_GLOBAL, 0);
    fixture->target_region = make_region(OSPREY_REGION_HEAP_SITE, 0x200);

    if (order == 0) {
        append_instance(fixture->ctx, fixture->global, 0,
                        0x1000, 0x1000, 0x1100, 0, 0);
        append_instance(fixture->ctx, fixture->target_region, 7,
                        0x2000, 0x2000, 0x2040, 0x111, 1);
        append_instance(fixture->ctx, fixture->target_region, 8,
                        0x3000, 0x3000, 0x3040, 0x222, 2);
    } else {
        append_instance(fixture->ctx, fixture->target_region, 8,
                        0x3000, 0x3000, 0x3040, 0x222, 2);
        append_instance(fixture->ctx, fixture->global, 0,
                        0x1000, 0x1000, 0x1100, 0, 0);
        append_instance(fixture->ctx, fixture->target_region, 7,
                        0x2000, 0x2000, 0x2040, 0x111, 1);
    }
    append_instance(fixture->ctx,
                    make_region(OSPREY_REGION_STACK_FUNCTION, 0x300), 9,
                    0x4000, 0x3f00, 0x4000, 0, 0);

    fixture->entry.cell = make_chunk(fixture->global, 8,
                                     sizeof(target_ulong));
    fixture->entry.target_base = make_address(fixture->target_region, 0);
    fixture->entry.extent = 16;
    fixture->entry.support = 1;
    fixture->entry.ordinal = 0;
    fixture->entry.kind = OSPREY_MUTATION_AGGREGATE_STRUCT;
    fixture->model.version = OSPREY_MUTATION_MODEL_VERSION;
    fixture->model.entry_count = 1;
    fixture->model.publication_valid = 1;
    fixture->model.entries = &fixture->entry;

    fixture->cell.start.valid = 1;
    fixture->cell.start.address = fixture->entry.cell.address;
    fixture->cell.start.raw = 0x1008;
    fixture->cell.start.instance_id = 0;
    fixture->cell.size = sizeof(target_ulong);

    fixture->target.valid = 1;
    fixture->target.address = fixture->entry.target_base;
    fixture->target.raw = 0x3000;
    fixture->target.instance_id = 8;
    fixture->target.prov_object_id = 0x222;
    fixture->target.prov_generation = 2;

    fixture->ctx->mutation_model = &fixture->model;
    fixture->ctx->mutation_model_ready = true;
    fixture->ctx->tx_status = OSPREY_OK;
    fixture->ctx->tx_model_ready = true;
    CHECK(osprey_runtime_index_build(fixture->ctx),
          order == 0 ? "runtime index builds in insertion order"
                     : "runtime index builds in reverse order");
    CHECK(osprey_runtime_mutation_prepare(fixture->ctx),
          "compact runtime publication validates once");
}

static void fixture_reprepare(Stage7Fixture *fixture)
{
    fixture->ctx->mutation_model_ready = true;
    fixture->ctx->mutation_runtime_ready = false;
    fixture->ctx->tx_status = OSPREY_OK;
    (void)osprey_runtime_mutation_prepare(fixture->ctx);
}

static void fixture_free(Stage7Fixture *fixture)
{
    if (fixture->ctx != NULL) {
        fixture->ctx->mutation_model = NULL;
        fixture->ctx->mutation_model_ready = false;
        fixture->ctx->mutation_runtime_ready = false;
        osprey_free(fixture->ctx);
        fixture->ctx = NULL;
    }
}

static bool cell_matches_reference(
    const OspreyContext *ctx, const OspreyMutationModel *model,
    const OspreyRuntimeChunkRef *locator,
    OspreyRuntimeResolveStatus expected,
    OspreyRuntimeCellResolution *actual_out)
{
    OspreyRuntimeCellResolution actual;
    OspreyRuntimeCellResolution reference;
    ResolverInputSnapshot snapshot = resolver_snapshot_take(ctx, model);
    OspreyRuntimeResolveStatus actual_status = osprey_runtime_resolve_cell(
        ctx, model, locator, &actual);
    OspreyRuntimeResolveStatus reference_status =
        stage7_reference_resolve_cell(ctx, model, locator, &reference);
    bool unchanged = resolver_snapshot_unchanged(&snapshot, ctx, model);
    resolver_snapshot_free(&snapshot);
    if (actual_out != NULL) *actual_out = actual;
    return actual_status == expected && reference_status == expected &&
           cell_resolution_equal(&actual, &reference) && unchanged;
}

static bool pointer_matches_reference(
    const OspreyContext *ctx, const OspreyMutationModel *model,
    const OspreyRuntimeChunkRef *cell_locator, target_ulong concrete_value,
    const OspreyRuntimeAddressRef *target_locator,
    OspreyRuntimeResolveStatus expected,
    OspreyRuntimePointerResolution *actual_out)
{
    OspreyRuntimePointerResolution actual;
    OspreyRuntimePointerResolution reference;
    ResolverInputSnapshot snapshot = resolver_snapshot_take(ctx, model);
    OspreyRuntimeResolveStatus actual_status = osprey_runtime_resolve_pointer(
        ctx, model, cell_locator, concrete_value, target_locator, &actual);
    OspreyRuntimeResolveStatus reference_status =
        stage7_reference_resolve_pointer(ctx, model, cell_locator,
                                         concrete_value, target_locator,
                                         &reference);
    bool unchanged = resolver_snapshot_unchanged(&snapshot, ctx, model);
    resolver_snapshot_free(&snapshot);
    if (actual_out != NULL) *actual_out = actual;
    return actual_status == expected && reference_status == expected &&
           pointer_resolution_equal(&actual, &reference) && unchanged;
}

static void test_exact_lookup_and_permutation(unsigned order)
{
    Stage7Fixture fixture;
    OspreyRuntimeCellResolution cell;
    OspreyRuntimePointerResolution pointer;
    OspreyRuntimeCounters counters;

    fixture_init(&fixture, order);
    osprey_runtime_test_reset_counters();
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_RESOLVED, &cell),
          "exact compact cell agrees with independent scanner");
    CHECK(cell.entry_ordinal == 0 && cell.instance_valid == 1 &&
              cell.aggregate_kind == OSPREY_MUTATION_AGGREGATE_STRUCT &&
              cell.target_extent == 16 && cell.cell.start.raw == 0x1008,
          "compact cell returns ordinal, kind, extent, and identity");
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000,
              &fixture.target, OSPREY_RUNTIME_RESOLVED, &pointer),
          "exact compact pointer agrees with independent scanner");
    CHECK(pointer.entry_ordinal == 0 && pointer.target_extent == 16 &&
              pointer.aggregate_kind == OSPREY_MUTATION_AGGREGATE_STRUCT &&
              pointer.target.prov_object_id == 0x222 &&
              pointer.target.raw == 0x3000,
          "pointer selects exact target instance and extent");
    osprey_runtime_test_get_counters(&counters);
    CHECK(counters.reception_validation_passes == 0 &&
              counters.cell_key_comparisons > 0 &&
              counters.cell_key_comparisons <= 4 &&
              counters.runtime_instance_comparisons > 0,
          "repeated resolver work uses prepared logarithmic indexes");
    fixture_free(&fixture);
}

static void test_null_and_cell_rejections(void)
{
    Stage7Fixture fixture;
    OspreyRuntimePointerResolution pointer;
    OspreyRuntimeCellResolution cell;
    OspreyRuntimeAddressRef bad_target;

    fixture_init(&fixture, 0);
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0, NULL,
              OSPREY_RUNTIME_RESOLVED, &pointer) &&
              !pointer.has_runtime_target && !pointer.target_valid &&
              pointer.target_extent == 16,
          "NULL compact pointer exposes type extent without fake target");
    bad_target = fixture.target;
    bad_target.valid = 1;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0, &bad_target,
              OSPREY_RUNTIME_MALFORMED, &pointer),
          "NULL pointer rejects nonzero target locator");

    fixture.cell.start.raw++;
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_STALE_INSTANCE, &cell),
          "stale cell raw value is unavailable");
    fixture.cell.start.raw = 0x1008;
    fixture.cell.start.reserved[1] = 1;
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_MALFORMED, &cell),
          "reserved locator bytes are rejected");
    fixture.cell.start.reserved[1] = 0;
    fixture.cell.size = sizeof(target_ulong) - 1;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0, NULL,
              OSPREY_RUNTIME_NO_CELL_OBJECT, &pointer),
          "unindexed width-changing cell is unavailable");
    fixture_free(&fixture);
}

static void test_target_identity_and_bounds(void)
{
    Stage7Fixture fixture;
    Stage7Fixture overflow_fixture;
    OspreyRuntimePointerResolution pointer;
    OspreyRuntimeAddressRef bad;
    OspreyRegionInstance *target_instance;

    fixture_init(&fixture, 0);
    bad = fixture.target;
    bad.address.region = fixture.global;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000, &bad,
              OSPREY_RUNTIME_STALE_INSTANCE, &pointer),
          "cross-region target identity is unavailable");
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000, NULL,
              OSPREY_RUNTIME_NO_LOCATOR, &pointer),
          "non-NULL pointer requires a target locator");
    bad = fixture.target;
    bad.raw = 0x2000;
    bad.instance_id = 7;
    bad.prov_object_id = 0x111;
    bad.prov_generation = 1;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000, &bad,
              OSPREY_RUNTIME_STALE_INSTANCE, &pointer),
          "target locator identity mismatch is unavailable");
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3001,
              &fixture.target, OSPREY_RUNTIME_STALE_INSTANCE, &pointer),
          "concrete pointer mismatch is unavailable");
    fixture.entry.extent = 65;
    fixture_reprepare(&fixture);
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000,
              &fixture.target, OSPREY_RUNTIME_OUT_OF_BOUNDS, &pointer),
          "target extent crossing runtime bound is unavailable");
    fixture.entry.extent = 16;
    fixture_reprepare(&fixture);
    fixture.target.raw = UINT64_MAX - 7;
    fixture.target.address = fixture.entry.target_base;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell,
              (target_ulong)fixture.target.raw, &fixture.target,
              OSPREY_RUNTIME_STALE_INSTANCE, &pointer),
          "unindexed overflowing target is unavailable");

    target_instance = &g_array_index(fixture.ctx->region_instances,
                                     OspreyRegionInstance, 2);
    target_instance->raw_base = UINT64_MAX - 7;
    target_instance->raw_min = UINT64_MAX - 7;
    target_instance->raw_max = UINT64_MAX;
    fixture.target.raw = UINT64_MAX - 7;
    CHECK(osprey_runtime_index_build(fixture.ctx),
          "runtime index accepts near-maximum target base");
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell,
              (target_ulong)fixture.target.raw, &fixture.target,
              OSPREY_RUNTIME_OUT_OF_BOUNDS, &pointer),
          "target addition overflow is unavailable");
    fixture_free(&fixture);

    fixture_init(&overflow_fixture, 0);
    OspreyRegionInstance *global_instance = &g_array_index(
        overflow_fixture.ctx->region_instances, OspreyRegionInstance, 0);
    global_instance->raw_base = UINT64_MAX - 7;
    global_instance->raw_min = UINT64_MAX - 7;
    global_instance->raw_max = UINT64_MAX;
    overflow_fixture.entry.cell.address.offset = 0;
    overflow_fixture.cell.start.address.offset = 0;
    overflow_fixture.cell.start.raw = UINT64_MAX - 7;
    fixture_reprepare(&overflow_fixture);
    CHECK(osprey_runtime_index_build(overflow_fixture.ctx),
          "runtime index accepts near-maximum cell base");
    CHECK(cell_matches_reference(
              overflow_fixture.ctx, &overflow_fixture.model,
              &overflow_fixture.cell, OSPREY_RUNTIME_OUT_OF_BOUNDS, NULL),
          "cell addition overflow is unavailable");
    fixture_free(&overflow_fixture);
}

static void test_signed_stack_and_heap_reuse(void)
{
    Stage7Fixture fixture;
    OspreyRuntimeCellResolution cell;
    OspreyRegionId stack = make_region(OSPREY_REGION_STACK_FUNCTION, 0x300);

    fixture_init(&fixture, 0);
    fixture.entry.cell = make_chunk(stack, -8, 8);
    fixture.cell.start.address = fixture.entry.cell.address;
    fixture.cell.start.raw = 0x3ff8;
    fixture.cell.start.instance_id = 9;
    fixture.cell.start.prov_object_id = 0;
    fixture.cell.start.prov_generation = 0;
    fixture.cell.size = 8;
    fixture_reprepare(&fixture);
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_RESOLVED, &cell) &&
              cell.cell.start.address.offset == -8,
          "signed recursive-stack locator resolves exactly");
    fixture_free(&fixture);
}

static void test_heap_identity_and_index_failures(void)
{
    Stage7Fixture fixture;
    OspreyRuntimePointerResolution pointer;
    OspreyRuntimeAddressRef target;
    OspreyRegionInstance duplicate;
    guint original_count;

    fixture_init(&fixture, 0);
    target = fixture.target;
    target.raw = 0x2000;
    target.instance_id = 7;
    target.prov_object_id = 0x111;
    target.prov_generation = 1;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x2000, &target,
              OSPREY_RUNTIME_RESOLVED, &pointer),
          "same-site heap reuse selects the exact generation");
    target.prov_generation = 2;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x2000, &target,
              OSPREY_RUNTIME_STALE_INSTANCE, &pointer),
          "heap provenance generation mismatch is stale");
    target = fixture.target;
    fixture.entry.target_base.offset = 63;
    target.address.offset = 63;
    target.raw = 0x303f;
    fixture_reprepare(&fixture);
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x303f, &target,
              OSPREY_RUNTIME_OUT_OF_BOUNDS, &pointer),
          "upper half-open target boundary is enforced");
    fixture.entry.target_base.offset = 0;
    fixture_reprepare(&fixture);

    original_count = fixture.ctx->region_instances->len;
    duplicate = g_array_index(fixture.ctx->region_instances,
                              OspreyRegionInstance, 1);
    g_array_append_val(fixture.ctx->region_instances, duplicate);
    CHECK(!osprey_runtime_index_build(fixture.ctx),
          "duplicate runtime identity rejects index construction");
    CHECK(osprey_runtime_resolve_pointer(
              fixture.ctx, &fixture.model, &fixture.cell, 0, NULL,
              &pointer) == OSPREY_RUNTIME_AMBIGUOUS_INSTANCE,
          "duplicate runtime identity is typed-unavailable");
    g_array_set_size(fixture.ctx->region_instances, original_count);
    CHECK(osprey_runtime_index_build(fixture.ctx),
          "runtime index recovers after duplicate removal");
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0, NULL,
              OSPREY_RUNTIME_RESOLVED, &pointer),
          "runtime lookup recovers after index rebuild");
    osprey_runtime_index_clear(fixture.ctx);
    CHECK(osprey_runtime_resolve_pointer(
              fixture.ctx, &fixture.model, &fixture.cell, 0, NULL,
              &pointer) == OSPREY_RUNTIME_INDEX_UNAVAILABLE,
          "missing runtime index is typed-unavailable");
    fixture_free(&fixture);
}

static unsigned ceil_log2_u32(uint32_t value)
{
    unsigned bits = 0;
    uint32_t bound = 1;
    while (bound < value) {
        bound <<= 1;
        bits++;
    }
    return bits;
}

static void test_logarithmic_lookup_growth(void)
{
    static const uint32_t sizes[] = { 1, 2, 4, 8, 16, 32, 64 };
    const OspreyRegionId global = make_region(OSPREY_REGION_GLOBAL, 0);

    for (size_t n_index = 0; n_index < G_N_ELEMENTS(sizes); n_index++) {
        uint32_t n = sizes[n_index];
        OspreyConfig config = test_config();
        OspreyContext *ctx = osprey_new(&config);
        OspreyMutationModel model;
        OspreyMutationEntry *entries = g_new0(OspreyMutationEntry, n);
        OspreyRuntimeChunkRef locator;
        OspreyRuntimeCellResolution resolution;
        OspreyRuntimeCounters counters;
        uint64_t offset = (uint64_t)(n - 1) * sizeof(target_ulong) + 8;

        append_instance(ctx, global, 0, 0x1000, 0x1000,
                        0x1000 + (uint64_t)n * sizeof(target_ulong) + 8,
                        0, 0);
        for (uint32_t i = 0; i < n; i++) {
            entries[i].cell = make_chunk(
                global, 8 + (int64_t)i * sizeof(target_ulong),
                sizeof(target_ulong));
            entries[i].target_base = make_address(global, 0);
            entries[i].extent = sizeof(target_ulong) * 2;
            entries[i].support = 1;
            entries[i].ordinal = i;
            entries[i].kind = OSPREY_MUTATION_AGGREGATE_STRUCT;
        }
        memset(&model, 0, sizeof(model));
        model.version = OSPREY_MUTATION_MODEL_VERSION;
        model.entry_count = n;
        model.publication_valid = 1;
        model.entries = entries;
        memset(&locator, 0, sizeof(locator));
        locator.start.valid = 1;
        locator.start.address = entries[n - 1].cell.address;
        locator.start.raw = 0x1000 + offset;
        locator.start.instance_id = 0;
        locator.size = sizeof(target_ulong);
        ctx->mutation_model = &model;
        ctx->mutation_model_ready = true;
        ctx->tx_status = OSPREY_OK;
        ctx->tx_model_ready = true;
        CHECK(osprey_runtime_index_build(ctx),
              "cell-growth runtime index builds");
        osprey_runtime_test_reset_counters();
        CHECK(osprey_runtime_mutation_prepare(ctx),
              "cell-growth compact model prepares");
        CHECK(osprey_runtime_resolve_cell(ctx, &model, &locator,
                                          &resolution) ==
                  OSPREY_RUNTIME_RESOLVED &&
              resolution.entry_ordinal == n - 1,
              "cell-growth lookup resolves the last ordinal");
        osprey_runtime_test_get_counters(&counters);
        CHECK(counters.reception_validation_passes == 1 &&
              counters.cell_key_comparisons <=
                  ceil_log2_u32(n) + 2 &&
              (n < 4 || counters.cell_key_comparisons < n),
              "cell lookup comparisons grow logarithmically");
        ctx->mutation_model = NULL;
        ctx->mutation_model_ready = false;
        ctx->mutation_runtime_ready = false;
        osprey_free(ctx);
        g_free(entries);
    }

    for (size_t r_index = 0; r_index < G_N_ELEMENTS(sizes); r_index++) {
        uint32_t r = sizes[r_index];
        OspreyConfig config = test_config();
        OspreyContext *ctx = osprey_new(&config);
        OspreyMutationModel model;
        OspreyMutationEntry entry;
        OspreyRuntimeChunkRef locator;
        OspreyRuntimeCellResolution resolution;
        OspreyRuntimeCounters counters;

        append_instance(ctx, global, 0, 0x1000, 0x1000, 0x1100, 0, 0);
        for (uint32_t i = 0; i < r; i++) {
            append_instance(ctx,
                            make_region(OSPREY_REGION_STACK_FUNCTION,
                                        0x100 + i),
                            1, 0x2000 + (uint64_t)i * 0x100,
                            0x2000 + (uint64_t)i * 0x100,
                            0x2080 + (uint64_t)i * 0x100, 0, 0);
        }
        memset(&entry, 0, sizeof(entry));
        entry.cell = make_chunk(global, 8, sizeof(target_ulong));
        entry.target_base = make_address(global, 0);
        entry.extent = sizeof(target_ulong) * 2;
        entry.support = 1;
        entry.kind = OSPREY_MUTATION_AGGREGATE_STRUCT;
        memset(&model, 0, sizeof(model));
        model.version = OSPREY_MUTATION_MODEL_VERSION;
        model.entry_count = 1;
        model.publication_valid = 1;
        model.entries = &entry;
        memset(&locator, 0, sizeof(locator));
        locator.start.valid = 1;
        locator.start.address = entry.cell.address;
        locator.start.raw = 0x1008;
        locator.start.instance_id = 0;
        locator.size = sizeof(target_ulong);
        ctx->mutation_model = &model;
        ctx->mutation_model_ready = true;
        ctx->tx_status = OSPREY_OK;
        ctx->tx_model_ready = true;
        CHECK(osprey_runtime_index_build(ctx),
              "runtime-growth index builds");
        osprey_runtime_test_reset_counters();
        CHECK(osprey_runtime_mutation_prepare(ctx),
              "runtime-growth compact model prepares");
        CHECK(osprey_runtime_resolve_cell(ctx, &model, &locator,
                                          &resolution) ==
                  OSPREY_RUNTIME_RESOLVED,
              "runtime-growth lookup resolves the global cell");
        osprey_runtime_test_get_counters(&counters);
        CHECK(counters.reception_validation_passes == 1 &&
              counters.runtime_instance_comparisons <=
                  2 * (ceil_log2_u32(r + 1) + 1) &&
              (r < 4 || counters.runtime_instance_comparisons < r + 1),
              "runtime-instance comparisons grow logarithmically");
        ctx->mutation_model = NULL;
        ctx->mutation_model_ready = false;
        ctx->mutation_runtime_ready = false;
        osprey_free(ctx);
    }
}

static void expect_reception_reject(Stage7Fixture *fixture,
                                    const char *message)
{
    fixture->ctx->mutation_runtime_ready = false;
    CHECK(!osprey_runtime_mutation_prepare(fixture->ctx), message);
    CHECK(osprey_runtime_mutation_model(fixture->ctx) == NULL,
          "failed reception hides compact model");
}

static void test_reception_and_counter_gate(void)
{
    Stage7Fixture fixture;
    OspreyRuntimeCounters counters;

    osprey_runtime_test_reset_counters();
    fixture_init(&fixture, 0);
    osprey_runtime_test_get_counters(&counters);
    CHECK(counters.reception_validation_passes == 1,
          "compact publication is validated once at reception");
    CHECK(osprey_runtime_mutation_prepare(fixture.ctx),
          "prepared compact model remains available");
    CHECK(osprey_runtime_mutation_prepare(fixture.ctx),
          "second reception call is a no-op");
    osprey_runtime_test_get_counters(&counters);
    CHECK(counters.reception_validation_passes == 1,
          "already prepared model is not revalidated");

    fixture.entry.ordinal = 3;
    expect_reception_reject(&fixture,
                            "bad compact ordinal fails reception validation");
    fixture.entry.ordinal = 0;
    fixture_reprepare(&fixture);
    fixture.entry.cell.size = 4;
    expect_reception_reject(&fixture,
                            "wrong compact pointer width fails reception validation");
    fixture.entry.cell.size = sizeof(target_ulong);
    fixture_reprepare(&fixture);
    fixture.entry.support = 0;
    expect_reception_reject(&fixture,
                            "missing compact support fails reception validation");
    fixture.entry.support = 1;
    fixture_reprepare(&fixture);
    fixture.entry.kind = 0;
    expect_reception_reject(&fixture,
                            "invalid compact aggregate kind fails reception validation");
    fixture.entry.kind = OSPREY_MUTATION_AGGREGATE_STRUCT;
    fixture_reprepare(&fixture);
    fixture.entry.extent = 0;
    expect_reception_reject(&fixture,
                            "zero compact extent fails reception validation");
    fixture.entry.extent = 16;
    fixture_reprepare(&fixture);
    fixture.entry.extent = OSPREY_MUTATION_MAX_EXTENT + 1;
    expect_reception_reject(&fixture,
                            "over-cap compact extent fails reception validation");
    fixture.entry.extent = 16;
    fixture_reprepare(&fixture);
    fixture.entry.target_base.region.kind = (OspreyRegionKind)99;
    expect_reception_reject(&fixture,
                            "invalid compact target identity fails reception validation");
    fixture.entry.target_base.region.kind = fixture.target_region.kind;
    fixture_reprepare(&fixture);
    fixture.model.entries = NULL;
    expect_reception_reject(&fixture,
                            "null compact entry array fails reception validation");
    fixture.model.entries = &fixture.entry;
    fixture_reprepare(&fixture);
    fixture.model.publication_valid = 0;
    expect_reception_reject(&fixture,
                            "unpublished compact model fails reception validation");
    fixture.model.publication_valid = 1;
    fixture_reprepare(&fixture);
    fixture.model.version = 0;
    expect_reception_reject(&fixture,
                            "wrong compact model version fails reception validation");
    fixture_free(&fixture);
}

int main(void)
{
    test_exact_lookup_and_permutation(0);
    test_exact_lookup_and_permutation(1);
    test_null_and_cell_rejections();
    test_target_identity_and_bounds();
    test_signed_stack_and_heap_reuse();
    test_heap_identity_and_index_failures();
    test_logarithmic_lookup_growth();
    test_reception_and_counter_gate();
    printf("stage7_runtime: %u/%u checks passed\n",
           checks - failures, checks);
    return failures == 0 ? 0 : 1;
}
