/* OSPREY Stage 7.2 focused tests: exact cell and pointee resolution. */

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
    return a->region.kind == b->region.kind &&
           a->region.code_image_id == b->region.code_image_id &&
           a->region.site_offset == b->region.site_offset &&
           a->offset == b->offset;
}

static bool address_ref_equal(const OspreyRuntimeAddressRef *a,
                              const OspreyRuntimeAddressRef *b)
{
    return address_equal(&a->address, &b->address) &&
           a->raw == b->raw && a->instance_id == b->instance_id &&
           a->prov_object_id == b->prov_object_id &&
           a->prov_generation == b->prov_generation &&
           a->valid == b->valid &&
           memcmp(a->reserved, b->reserved, sizeof(a->reserved)) == 0;
}

static bool cell_resolution_equal(const OspreyRuntimeCellResolution *a,
                                  const OspreyRuntimeCellResolution *b)
{
    return a->status == b->status && a->object_index == b->object_index &&
           a->instance_valid == b->instance_valid &&
           memcmp(a->reserved, b->reserved, sizeof(a->reserved)) == 0 &&
           a->cell.size == b->cell.size &&
           address_ref_equal(&a->cell.start, &b->cell.start) &&
           a->object == b->object;
}

static bool pointer_resolution_equal(
    const OspreyRuntimePointerResolution *a,
    const OspreyRuntimePointerResolution *b)
{
    return a->status == b->status &&
           a->target_type_id == b->target_type_id &&
           a->target_extent == b->target_extent &&
           a->has_runtime_target == b->has_runtime_target &&
           a->target_valid == b->target_valid &&
           memcmp(a->reserved, b->reserved, sizeof(a->reserved)) == 0 &&
           cell_resolution_equal(&a->cell, &b->cell) &&
           address_equal(&a->target_base, &b->target_base) &&
           address_ref_equal(&a->target, &b->target);
}

typedef struct ResolverInputSnapshot {
    void *regions;
    size_t region_bytes;
    void *objects;
    size_t object_bytes;
    void *types;
    size_t type_bytes;
    void *chunk_index;
    size_t chunk_index_bytes;
} ResolverInputSnapshot;

static void *copy_bytes(const void *source, size_t bytes)
{
    if (bytes == 0) return NULL;
    void *copy = g_malloc(bytes);
    memcpy(copy, source, bytes);
    return copy;
}

static ResolverInputSnapshot resolver_snapshot_take(
    const OspreyContext *ctx, const OspreyModel *model)
{
    ResolverInputSnapshot snapshot;
    memset(&snapshot, 0, sizeof(snapshot));
    snapshot.region_bytes = ctx->region_instances->len *
                            sizeof(OspreyRegionInstance);
    snapshot.regions = copy_bytes(ctx->region_instances->data,
                                  snapshot.region_bytes);
    snapshot.object_bytes = model->object_count * sizeof(OspreyDecodedObject);
    snapshot.objects = copy_bytes(model->objects, snapshot.object_bytes);
    snapshot.type_bytes = model->type_count * sizeof(OspreyDecodedType);
    snapshot.types = copy_bytes(model->types, snapshot.type_bytes);
    snapshot.chunk_index_bytes = model->chunk_index_count *
                                 sizeof(OspreyModelIndexEntry);
    snapshot.chunk_index = copy_bytes(model->chunk_index,
                                      snapshot.chunk_index_bytes);
    return snapshot;
}

static bool resolver_snapshot_unchanged(
    const ResolverInputSnapshot *snapshot, const OspreyContext *ctx,
    const OspreyModel *model)
{
    return snapshot->region_bytes ==
               ctx->region_instances->len * sizeof(OspreyRegionInstance) &&
           (snapshot->region_bytes == 0 ||
            memcmp(snapshot->regions, ctx->region_instances->data,
                   snapshot->region_bytes) == 0) &&
           snapshot->object_bytes ==
               model->object_count * sizeof(OspreyDecodedObject) &&
           (snapshot->object_bytes == 0 ||
            memcmp(snapshot->objects, model->objects,
                   snapshot->object_bytes) == 0) &&
           snapshot->type_bytes ==
               model->type_count * sizeof(OspreyDecodedType) &&
           (snapshot->type_bytes == 0 ||
            memcmp(snapshot->types, model->types,
                   snapshot->type_bytes) == 0) &&
           snapshot->chunk_index_bytes ==
               model->chunk_index_count * sizeof(OspreyModelIndexEntry) &&
           (snapshot->chunk_index_bytes == 0 ||
            memcmp(snapshot->chunk_index, model->chunk_index,
                   snapshot->chunk_index_bytes) == 0);
}

static void resolver_snapshot_free(ResolverInputSnapshot *snapshot)
{
    g_free(snapshot->regions);
    g_free(snapshot->objects);
    g_free(snapshot->types);
    g_free(snapshot->chunk_index);
    memset(snapshot, 0, sizeof(*snapshot));
}

static bool cell_matches_reference(
    const OspreyContext *ctx, const OspreyModel *model,
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
    const OspreyContext *ctx, const OspreyModel *model,
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
    OspreyModel model;
    OspreyDecodedObject object;
    OspreyDecodedType types[2];
    OspreyModelIndexEntry chunk_index;
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

    memset(&fixture->model, 0, sizeof(fixture->model));
    memset(&fixture->object, 0, sizeof(fixture->object));
    memset(fixture->types, 0, sizeof(fixture->types));
    fixture->object.chunk = make_chunk(fixture->global, 8,
                                       sizeof(target_ulong));
    fixture->object.has_pointer_target = 1;
    fixture->object.value_type_id = 0;
    fixture->object.pointer_target = make_address(fixture->target_region, 0);
    fixture->types[0].id = 0;
    fixture->types[0].kind = OSPREY_TYPE_POINTER;
    fixture->types[0].size = sizeof(target_ulong);
    fixture->types[0].target_type_id = 1;
    fixture->types[0].canonical_base = fixture->object.pointer_target;
    fixture->types[1].id = 1;
    fixture->types[1].kind = OSPREY_TYPE_STRUCT;
    fixture->types[1].size = 16;
    fixture->types[1].canonical_base = fixture->object.pointer_target;
    fixture->chunk_index.key = osprey_chunk_key(&fixture->object.chunk);
    fixture->chunk_index.ordinal = 0;
    fixture->model.version = OSPREY_MODEL_VERSION;
    fixture->model.object_count = 1;
    fixture->model.type_count = 2;
    fixture->model.chunk_index_count = 1;
    fixture->model.objects = &fixture->object;
    fixture->model.types = fixture->types;
    fixture->model.chunk_index = &fixture->chunk_index;

    memset(&fixture->cell, 0, sizeof(fixture->cell));
    fixture->cell.start.valid = 1;
    fixture->cell.start.address = fixture->object.chunk.address;
    fixture->cell.start.raw = 0x1008;
    fixture->cell.start.instance_id = 0;
    fixture->cell.size = sizeof(target_ulong);

    memset(&fixture->target, 0, sizeof(fixture->target));
    fixture->target.valid = 1;
    fixture->target.address = fixture->object.pointer_target;
    fixture->target.raw = 0x3000;
    fixture->target.instance_id = 8;
    fixture->target.prov_object_id = 0x222;
    fixture->target.prov_generation = 2;

    CHECK(osprey_runtime_index_build(fixture->ctx),
          order == 0 ? "runtime index builds in insertion order"
                     : "runtime index builds in reverse order");
}

static void fixture_free(Stage7Fixture *fixture)
{
    if (fixture->ctx != NULL) {
        osprey_free(fixture->ctx);
        fixture->ctx = NULL;
    }
}

static void fixture_select_cell(Stage7Fixture *fixture,
                                OspreyRegionId region, int64_t offset,
                                uint64_t size, uint64_t raw,
                                uint64_t instance_id, uint64_t object_id,
                                uint32_t generation)
{
    fixture->object.chunk = make_chunk(region, offset, size);
    fixture->chunk_index.key = osprey_chunk_key(&fixture->object.chunk);
    fixture->cell.start.address = fixture->object.chunk.address;
    fixture->cell.start.raw = raw;
    fixture->cell.start.instance_id = instance_id;
    fixture->cell.start.prov_object_id = object_id;
    fixture->cell.start.prov_generation = generation;
    fixture->cell.start.valid = 1;
    memset(fixture->cell.start.reserved, 0,
           sizeof(fixture->cell.start.reserved));
    fixture->cell.size = size;
}

static void fixture_select_target(Stage7Fixture *fixture,
                                  OspreyRegionId region, int64_t offset,
                                  uint64_t raw, uint64_t instance_id,
                                  uint64_t object_id, uint32_t generation)
{
    fixture->object.pointer_target = make_address(region, offset);
    fixture->types[0].canonical_base = fixture->object.pointer_target;
    fixture->types[1].canonical_base = fixture->object.pointer_target;
    memset(&fixture->target, 0, sizeof(fixture->target));
    fixture->target.address = fixture->object.pointer_target;
    fixture->target.raw = raw;
    fixture->target.instance_id = instance_id;
    fixture->target.prov_object_id = object_id;
    fixture->target.prov_generation = generation;
    fixture->target.valid = 1;
}

static void test_exact_lookup_and_permutation(unsigned order)
{
    Stage7Fixture fixture;
    OspreyRuntimeCellResolution cell;
    OspreyRuntimePointerResolution pointer;
    OspreyRegionInstance *regions_copy;
    OspreyDecodedObject object_copy;
    OspreyDecodedType types_copy[2];
    OspreyModelIndexEntry index_copy;

    fixture_init(&fixture, order);
    regions_copy = g_malloc(fixture.ctx->region_instances->len *
                            sizeof(OspreyRegionInstance));
    memcpy(regions_copy, fixture.ctx->region_instances->data,
           fixture.ctx->region_instances->len * sizeof(OspreyRegionInstance));
    object_copy = fixture.object;
    memcpy(types_copy, fixture.types, sizeof(types_copy));
    index_copy = fixture.chunk_index;

    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_RESOLVED, &cell),
          "exact cell agrees with independent reference");
    CHECK(cell.object == &fixture.object && cell.object_index == 0 &&
          cell.instance_valid == 1 && cell.cell.start.raw == 0x1008,
          "exact cell returns canonical object and copied identity");

    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000,
              &fixture.target, OSPREY_RUNTIME_RESOLVED, &pointer),
          "non-NULL pointer target agrees with independent reference");
    CHECK(pointer.has_runtime_target == 1 && pointer.target_valid == 1 &&
          pointer.target.instance_id == 8 &&
          pointer.target.prov_object_id == 0x222 &&
          pointer.target_extent == 16 && pointer.target_type_id == 1,
          "target uses selected instance and aggregate extent");
    CHECK(pointer.target.address.region.kind ==
              fixture.object.pointer_target.region.kind &&
          pointer.target.address.region.code_image_id ==
              fixture.object.pointer_target.region.code_image_id &&
          pointer.target.address.region.site_offset ==
              fixture.object.pointer_target.region.site_offset &&
          pointer.target.address.offset == fixture.object.pointer_target.offset,
          "target canonical base is exact");

    memset(&pointer, 0xa5, sizeof(pointer));
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0, NULL,
              OSPREY_RUNTIME_RESOLVED, &pointer),
          "NULL pointer agrees with independent reference");
    CHECK(pointer.target_extent == 16 && pointer.has_runtime_target == 0 &&
          pointer.target_valid == 0 && pointer.target.raw == 0 &&
          pointer.target.instance_id == 0,
          "NULL result has no fabricated runtime target");

    CHECK(memcmp(regions_copy, fixture.ctx->region_instances->data,
                 fixture.ctx->region_instances->len *
                 sizeof(OspreyRegionInstance)) == 0,
          "resolver leaves runtime records unchanged");
    CHECK(memcmp(&object_copy, &fixture.object, sizeof(object_copy)) == 0 &&
          memcmp(types_copy, fixture.types, sizeof(types_copy)) == 0 &&
          memcmp(&index_copy, &fixture.chunk_index, sizeof(index_copy)) == 0,
          "resolver leaves model arrays unchanged");
    g_free(regions_copy);
    fixture_free(&fixture);
}

static void test_target_identity_and_bounds(void)
{
    Stage7Fixture fixture;
    OspreyRuntimePointerResolution result;
    OspreyRuntimeAddressRef stale;

    fixture_init(&fixture, 0);

    stale = fixture.target;
    stale.prov_generation++;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000, &stale,
              OSPREY_RUNTIME_STALE_INSTANCE, &result),
          "stale target generation is rejected");

    stale = fixture.target;
    stale.raw++;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000, &stale,
              OSPREY_RUNTIME_STALE_INSTANCE, &result),
          "concrete target mismatch is rejected");

    stale = fixture.target;
    stale.instance_id = 7;
    stale.prov_object_id = 0x111;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000, &stale,
              OSPREY_RUNTIME_STALE_INSTANCE, &result),
          "raw/identity mismatch is rejected instead of first-instance lookup");

    fixture.types[1].size = 65;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000,
              &fixture.target, OSPREY_RUNTIME_OUT_OF_BOUNDS, &result),
          "decoded target extent crossing runtime bound is rejected");
    fixture.types[1].size = 16;

    fixture.types[1].kind = OSPREY_TYPE_ARRAY;
    fixture.types[1].size = 32;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000,
              &fixture.target, OSPREY_RUNTIME_RESOLVED, &result) &&
          result.target_extent == 32,
          "array target uses decoded aggregate size");
    fixture.types[1].kind = OSPREY_TYPE_STRUCT;
    fixture.types[1].size = 16;

    fixture_free(&fixture);
}

static void test_cell_rejections(void)
{
    Stage7Fixture fixture;
    OspreyRuntimeCellResolution cell;
    OspreyRuntimeChunkRef bad;

    fixture_init(&fixture, 1);

    memset(&bad, 0, sizeof(bad));
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &bad,
                                 OSPREY_RUNTIME_NO_LOCATOR, &cell),
          "missing cell locator is typed-unavailable");

    bad = fixture.cell;
    bad.start.raw++;
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &bad,
                                 OSPREY_RUNTIME_STALE_INSTANCE, &cell),
          "cell raw translation mismatch is rejected");

    bad = fixture.cell;
    bad.size = 0x200;
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &bad,
                                 OSPREY_RUNTIME_OUT_OF_BOUNDS, &cell),
          "cell interval crossing instance bound is rejected");

    bad = fixture.cell;
    bad.start.address.offset = 16;
    bad.start.raw = 0x1010;
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &bad,
                                 OSPREY_RUNTIME_NO_CELL_OBJECT, &cell),
          "valid unmodeled cell remains generic-eligible");

    bad = fixture.cell;
    bad.start.valid = 2;
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &bad,
                                 OSPREY_RUNTIME_MALFORMED, &cell),
          "invalid locator flag is rejected");

    fixture.chunk_index.ordinal = 7;
    CHECK(osprey_runtime_resolve_cell(fixture.ctx, &fixture.model,
                                      &fixture.cell, &cell) ==
              OSPREY_RUNTIME_MALFORMED,
          "corrupt model index is rejected without an unsafe lookup");
    fixture.chunk_index.ordinal = 0;
    fixture_free(&fixture);
}

static void test_stack_cell_identity(void)
{
    Stage7Fixture fixture;
    OspreyRuntimeCellResolution cell;
    OspreyRegionId stack = make_region(OSPREY_REGION_STACK_FUNCTION, 0x300);

    fixture_init(&fixture, 0);
    fixture.object.chunk = make_chunk(stack, -8, 8);
    fixture.chunk_index.key = osprey_chunk_key(&fixture.object.chunk);
    fixture.cell.start.address = fixture.object.chunk.address;
    fixture.cell.start.raw = 0x3ff8;
    fixture.cell.start.instance_id = 9;
    fixture.cell.start.prov_object_id = 0;
    fixture.cell.start.prov_generation = 0;
    fixture.cell.size = 8;
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_RESOLVED, &cell),
          "negative stack offset resolves exactly");

    fixture.object.chunk = make_chunk(stack, -0x100, 1);
    fixture.chunk_index.key = osprey_chunk_key(&fixture.object.chunk);
    fixture.cell.start.address = fixture.object.chunk.address;
    fixture.cell.start.raw = 0x3f00;
    fixture.cell.size = 1;
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_RESOLVED, &cell),
          "stack lower-bound equality resolves");

    fixture.object.chunk = make_chunk(stack, -1, 2);
    fixture.chunk_index.key = osprey_chunk_key(&fixture.object.chunk);
    fixture.cell.start.address = fixture.object.chunk.address;
    fixture.cell.start.raw = 0x3fff;
    fixture.cell.size = 2;
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_MALFORMED, &cell),
          "stack interval crossing the anchor/end is rejected");
    fixture_free(&fixture);
}

static void test_heap_reuse_alias_and_recursion(void)
{
    Stage7Fixture fixture;
    OspreyRuntimeCellResolution cell;
    OspreyRegionId alias_heap = make_region(OSPREY_REGION_HEAP_SITE, 0x201);
    OspreyRegionId stack = make_region(OSPREY_REGION_STACK_FUNCTION, 0x300);
    OspreyRuntimeChunkRef original;

    fixture_init(&fixture, 0);
    append_instance(fixture.ctx, fixture.target_region, 10,
                    0x3000, 0x3000, 0x3040, 0x333, 3);
    append_instance(fixture.ctx, alias_heap, 11,
                    0x1000, 0x1000, 0x1100, 0x444, 4);
    append_instance(fixture.ctx, stack, 10,
                    0x5000, 0x4f00, 0x5000, 0, 0);
    CHECK(osprey_runtime_index_build(fixture.ctx),
          "same-base and recursive identities build one exact index");

    fixture_select_cell(&fixture, fixture.target_region, 8, 8,
                        0x3008, 8, 0x222, 2);
    original = fixture.cell;
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_RESOLVED, &cell),
          "heap cell resolves by complete instance and provenance identity");

    fixture_select_cell(&fixture, fixture.target_region, 8, 8,
                        0x3008, 10, 0x333, 3);
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_RESOLVED, &cell),
          "same-base heap reuse resolves the selected generation");
    CHECK(original.start.raw == fixture.cell.start.raw &&
          original.start.instance_id != fixture.cell.start.instance_id &&
          original.start.prov_object_id != fixture.cell.start.prov_object_id,
          "same-base heap locators differ only in full runtime identity");

    fixture_select_cell(&fixture, fixture.global, 8, 8,
                        0x1008, 0, 0, 0);
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_RESOLVED, &cell),
          "global cell wins despite a heap instance at the same raw address");
    fixture_select_cell(&fixture, alias_heap, 8, 8,
                        0x1008, 11, 0x444, 4);
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_RESOLVED, &cell),
          "heap cell wins despite a global instance at the same raw address");

    fixture_select_cell(&fixture, stack, -8, 8,
                        0x3ff8, 9, 0, 0);
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_RESOLVED, &cell),
          "first recursive stack activation resolves exactly");
    fixture_select_cell(&fixture, stack, -8, 8,
                        0x4ff8, 10, 0, 0);
    CHECK(cell_matches_reference(fixture.ctx, &fixture.model, &fixture.cell,
                                 OSPREY_RUNTIME_RESOLVED, &cell),
          "second recursive stack activation resolves exactly");

    fixture_free(&fixture);
}

static void test_pointer_extremes_and_index_ownership(void)
{
    Stage7Fixture fixture;
    OspreyRuntimePointerResolution pointer;
    OspreyRuntimeCellResolution cell;
    OspreyRegionId overflow_region =
        make_region(OSPREY_REGION_HEAP_SITE, 0x400);
    OspreyRegionId empty_region =
        make_region(OSPREY_REGION_HEAP_SITE, 0x500);
    OspreyRuntimeAddressRef bad_target;
    OspreyRegionInstance *source_copy;
    guint source_count;

    fixture_init(&fixture, 0);

    fixture.types[1].size = 0;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0, NULL,
              OSPREY_RUNTIME_MALFORMED, &pointer),
          "zero-size aggregate target is rejected");
    fixture.types[1].size = 16;

    fixture.types[1].canonical_base.offset = 1;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0, NULL,
              OSPREY_RUNTIME_MALFORMED, &pointer),
          "aggregate base mismatch is rejected");
    fixture.types[1].canonical_base = fixture.object.pointer_target;

    bad_target = fixture.target;
    bad_target.reserved[1] = 1;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000,
              &bad_target, OSPREY_RUNTIME_MALFORMED, &pointer),
          "nonzero target locator reserved state is rejected");

    append_instance(fixture.ctx, overflow_region, 12,
                    UINT64_MAX - 31, UINT64_MAX - 31, UINT64_MAX,
                    0x555, 5);
    CHECK(osprey_runtime_index_build(fixture.ctx),
          "near-address-limit runtime instance builds");
    fixture_select_target(&fixture, overflow_region, 0,
                          UINT64_MAX - 31, 12, 0x555, 5);
    fixture.types[1].size = 32;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell,
              (target_ulong)(UINT64_MAX - 31), &fixture.target,
              OSPREY_RUNTIME_OUT_OF_BOUNDS, &pointer),
          "target end overflow is rejected");

    append_instance(fixture.ctx, empty_region, 13,
                    0x6000, 0x6000, 0x6000, 0x666, 6);
    CHECK(osprey_runtime_index_build(fixture.ctx),
          "zero-size heap runtime instance remains representable");
    fixture_select_target(&fixture, empty_region, 0,
                          0x6000, 13, 0x666, 6);
    fixture.types[1].size = 1;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x6000,
              &fixture.target, OSPREY_RUNTIME_OUT_OF_BOUNDS, &pointer),
          "zero-size runtime target cannot contain one byte");

    /* The production index owns its baseline copy.  Mutating the source
     * GArray after construction cannot redirect an existing lookup. */
    fixture_select_target(&fixture, fixture.target_region, 0,
                          0x3000, 8, 0x222, 2);
    fixture.types[1].size = 16;
    CHECK(osprey_runtime_index_build(fixture.ctx),
          "clean runtime index rebuilds before ownership check");
    source_count = fixture.ctx->region_instances->len;
    source_copy = g_memdup(fixture.ctx->region_instances->data,
                          source_count * sizeof(*source_copy));
    memset(fixture.ctx->region_instances->data, 0,
           source_count * sizeof(*source_copy));
    CHECK(osprey_runtime_resolve_cell(fixture.ctx, &fixture.model,
                                      &fixture.cell, &cell) ==
              OSPREY_RUNTIME_RESOLVED,
          "runtime index does not alias the mutable source GArray");
    memcpy(fixture.ctx->region_instances->data, source_copy,
           source_count * sizeof(*source_copy));
    g_free(source_copy);

    fixture_free(&fixture);
}

static void test_pointer_type_rejections(void)
{
    Stage7Fixture fixture;
    OspreyRuntimePointerResolution result;
    OspreyRuntimeAddressRef bad_target;

    fixture_init(&fixture, 0);

    fixture.object.has_pointer_target = 0;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0, NULL,
              OSPREY_RUNTIME_NO_POINTER_TARGET, &result),
          "pointer without selected target is unavailable");
    fixture.object.has_pointer_target = 1;

    fixture.types[0].target_is_void = 1;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0, NULL,
              OSPREY_RUNTIME_VOID_TARGET, &result),
          "void pointer target is unavailable");
    fixture.types[0].target_is_void = 0;

    fixture.types[1].kind = OSPREY_TYPE_PRIMITIVE;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0, NULL,
              OSPREY_RUNTIME_MISSING_AGGREGATE, &result),
          "primitive target type is unavailable");
    fixture.types[1].kind = OSPREY_TYPE_STRUCT;

    fixture.types[0].target_type_id = 99;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0, NULL,
              OSPREY_RUNTIME_MISSING_TYPE, &result),
          "missing target type is unavailable");
    fixture.types[0].target_type_id = 1;

    fixture.types[0].kind = OSPREY_TYPE_PRIMITIVE;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0, NULL,
              OSPREY_RUNTIME_NON_POINTER, &result),
          "primitive cell value type is not treated as pointer");
    fixture.types[0].kind = OSPREY_TYPE_POINTER;

    bad_target = fixture.target;
    bad_target.address.region = fixture.global;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000,
              &bad_target, OSPREY_RUNTIME_STALE_INSTANCE, &result),
          "same raw address with wrong canonical region is rejected");

    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0x3000, NULL,
              OSPREY_RUNTIME_NO_LOCATOR, &result),
          "non-NULL pointer without target locator is unavailable");

    bad_target = fixture.target;
    bad_target.valid = 1;
    CHECK(pointer_matches_reference(
              fixture.ctx, &fixture.model, &fixture.cell, 0,
              &bad_target, OSPREY_RUNTIME_MALFORMED, &result),
          "NULL pointer with fabricated target identity is rejected");

    fixture_free(&fixture);
}

static void test_index_failure_is_typed_unavailable(void)
{
    Stage7Fixture fixture;
    OspreyRuntimeCellResolution cell;
    OspreyRegionInstance duplicate;

    fixture_init(&fixture, 0);
    duplicate = g_array_index(fixture.ctx->region_instances,
                              OspreyRegionInstance, 2);
    g_array_append_val(fixture.ctx->region_instances, duplicate);
    CHECK(!osprey_runtime_index_build(fixture.ctx),
          "duplicate complete runtime identity rejects index build");
    CHECK(osprey_runtime_resolve_cell(fixture.ctx, &fixture.model,
                                      &fixture.cell, &cell) ==
              OSPREY_RUNTIME_AMBIGUOUS_INSTANCE,
          "ambiguous runtime identity is typed-unavailable");
    g_array_set_size(fixture.ctx->region_instances,
                     fixture.ctx->region_instances->len - 1);
    CHECK(osprey_runtime_index_build(fixture.ctx),
          "index recovers after duplicate removal");
    CHECK(osprey_runtime_resolve_cell(fixture.ctx, &fixture.model,
                                      &fixture.cell, &cell) ==
              OSPREY_RUNTIME_RESOLVED,
          "clean index resolves after recovery");
    fixture_free(&fixture);
}

int main(void)
{
    osprey_collect_enabled = 1;
    for (unsigned order = 0; order < 2; order++) {
        test_exact_lookup_and_permutation(order);
    }
    test_target_identity_and_bounds();
    test_cell_rejections();
    test_stack_cell_identity();
    test_heap_reuse_alias_and_recursion();
    test_pointer_extremes_and_index_ownership();
    test_pointer_type_rejections();
    test_index_failure_is_typed_unavailable();

    if (failures != 0) {
        fprintf(stderr, "stage7-runtime: %u/%u checks failed\n",
                failures, checks);
        return 1;
    }
    printf("stage7-runtime: %u checks passed\n", checks);
    return 0;
}
