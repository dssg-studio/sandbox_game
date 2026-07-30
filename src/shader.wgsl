// Full-screen voxel ray tracer.  Chunk data is a packed storage buffer rather
// than a mesh: this lets a block hold an exact 1/16-th height slab.

struct Uniforms {
    camera_position: vec4<f32>,
    camera_forward: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    sun_direction: vec4<f32>,
    detail_world: vec4<f32>,
    // Physical chunk origin for logical detailed (0, 0), ring side, unused.
    detail_ring: vec4<f32>,
    lod_world: vec4<f32>,
    simulation: vec4<f32>,
    // Exact selected block/slab bounds. selection_min.w is the active flag.
    selection_min: vec4<f32>,
    selection_max: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> detail_blocks: array<u32>;
// 1/16th top height in bits 0..15, top material in 16..23 and cliff
// material in 24..31.  These are compact full-data columns, not a bare
// height-only horizon map.
@group(0) @binding(2) var<storage, read> lod_columns: array<u32>;
// Three vec4s per tree: bounds, segment range/BLAS root, appearance.
@group(0) @binding(3) var<storage, read> trees: array<vec4<f32>>;
// [world origin x, world origin z, cell size, height-buffer offset] per level.
@group(0) @binding(4) var<storage, read> lod_levels: array<vec4<f32>>;
// Eco Machina view of a Dynamic Trees graph: one constant-width rectangular
// spine per HPD wood block, plus separately flagged wood/foliage connectors.
struct TreeSegment {
    start_radius: vec4<f32>,
    end_radius: vec4<f32>,
    style: array<u32, 4>,
};
@group(0) @binding(5) var<storage, read> tree_segments: array<TreeSegment>;
@group(0) @binding(6) var tree_texture: texture_2d<f32>;
@group(0) @binding(7) var tree_sampler: sampler;
struct TreeBvhNode {
    minimum: vec4<f32>,
    maximum: vec4<f32>,
    // Leaf: [first primitive, count, 1, 0]. Inner: [left child, right child, 0, 0].
    data: array<u32, 4>,
};
@group(0) @binding(8) var<storage, read> tree_blas_nodes: array<TreeBvhNode>;
@group(0) @binding(9) var<storage, read> tree_tlas_nodes: array<TreeBvhNode>;
// One packed buffer containing exact non-air occupancy at 1³, 4³, and 16³.
// It only lets DDA skip known-empty cells; every potentially visible block is
// still intersected against its original slab AABB below.
@group(0) @binding(10) var<storage, read> detail_occupancy: array<u32>;
// These are written by the primary pass and read by the shadow/composition
// passes. `textureLoad` keeps the G-buffer values exact: no filtering or
// colour-space conversion takes place between the ray and lighting stages.
@group(0) @binding(11) var primary_geometry: texture_2d<f32>;
@group(0) @binding(12) var primary_surface: texture_2d<f32>;
@group(0) @binding(13) var shadow_mask: texture_2d<u32>;

const AIR: u32 = 0u;
const GRASS: u32 = 1u;
const DIRT: u32 = 2u;
const STONE: u32 = 3u;
const WATER: u32 = 4u;
const OAK_LEAVES: u32 = 6u;
const SPRUCE_LEAVES: u32 = 7u;
const ACACIA_LEAVES: u32 = 8u;
const ROOTY_SOIL: u32 = 9u;
const OAK_BARK: u32 = 10u;
const SPRUCE_BARK: u32 = 11u;
const ACACIA_BARK: u32 = 12u;
const OAK_RINGS: u32 = 13u;
const SPRUCE_RINGS: u32 = 14u;
const ACACIA_RINGS: u32 = 15u;
const FOLIAGE_CONNECTOR_0: u32 = 16u;
const FOLIAGE_CONNECTOR_5: u32 = 21u;
const MAX_LOD_STEPS: u32 = 180u;
const DETAIL_FINE_OCCUPANCY_WORDS: u32 = 31104u;
const DETAIL_BRICK_OCCUPANCY_OFFSET: u32 = DETAIL_FINE_OCCUPANCY_WORDS;
const DETAIL_COARSE_OCCUPANCY_OFFSET: u32 = 31590u;
const DETAIL_BRICKS_PER_CHUNK: u32 = 192u;
const DETAIL_COARSE_LAYERS_PER_CHUNK: u32 = 3u;
const MAX_DETAILED_COARSE_STEPS: u32 = 32u;
const MAX_DETAILED_BRICK_STEPS: u32 = 16u;
const MAX_DETAILED_FINE_STEPS: u32 = 16u;
// A BLAS has at most 512 leaves and the forest TLAS at most 162 leaves, so
// the maximum DFS frontier depth is below 16 entries.
const TREE_BVH_STACK_CAPACITY: u32 = 16u;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct Hit {
    t: f32,
    normal: vec3<f32>,
    material: u32,
    texture_uv: vec2<f32>,
    found: bool,
};

struct RayInterval {
    entry: f32,
    exit: f32,
    found: bool,
};

struct LodColumn {
    minimum: vec2<f32>,
    cell_size: f32,
    height: f32,
    top_material: u32,
    side_material: u32,
    found: bool,
};

struct PrimaryOut {
    // Distance, then geometric normal.
    @location(0) geometry: vec4<f32>,
    // Texture UV, material id and a valid-hit bit.
    @location(1) surface: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOut {
    var points = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: VertexOut;
    output.position = vec4<f32>(points[index], 0.0, 1.0);
    output.uv = points[index] * 0.5 + 0.5;
    return output;
}

fn empty_hit(max_distance: f32) -> Hit {
    return Hit(max_distance, vec3<f32>(0.0), AIR, vec2<f32>(0.0), false);
}

fn block_at(cell: vec3<i32>) -> u32 {
    let detail_x = cell.x - i32(u.detail_world.x);
    let detail_z = cell.z - i32(u.detail_world.y);
    let detail_width = i32(u.detail_world.z);
    let detail_depth = i32(u.detail_world.w);
    if detail_x >= 0 && detail_x < detail_width &&
        detail_z >= 0 && detail_z < detail_depth &&
        cell.y >= 0 && cell.y < i32(u.simulation.z) {
        // Detailed storage is chunk-major and physically ring-buffered, so a
        // stream transition writes only entering chunk ranges. Every chunk
        // remains contiguous in the storage buffer.
        let chunks_per_side = i32(u.detail_ring.z);
        let logical_chunk_x = detail_x / 16;
        let logical_chunk_z = detail_z / 16;
        let physical_chunk_x = (logical_chunk_x + i32(u.detail_ring.x)) % chunks_per_side;
        let physical_chunk_z = (logical_chunk_z + i32(u.detail_ring.y)) % chunks_per_side;
        let local_x = detail_x % 16;
        let local_z = detail_z % 16;
        let index = u32(local_x + 16 * (local_z + 16 * (
            cell.y + i32(u.simulation.z) * (physical_chunk_x + chunks_per_side * physical_chunk_z)
        )));
        return detail_blocks[index];
    }

    return AIR;
}

fn occupancy_is_set(bit: u32) -> bool {
    return (detail_occupancy[bit / 32u] & (1u << (bit % 32u))) != 0u;
}

fn detailed_physical_slot(detail_x: i32, detail_z: i32) -> i32 {
    let chunks_per_side = i32(u.detail_ring.z);
    let logical_chunk_x = detail_x / 16;
    let logical_chunk_z = detail_z / 16;
    let physical_chunk_x = (logical_chunk_x + i32(u.detail_ring.x)) % chunks_per_side;
    let physical_chunk_z = (logical_chunk_z + i32(u.detail_ring.y)) % chunks_per_side;
    return physical_chunk_x + chunks_per_side * physical_chunk_z;
}

fn detailed_fine_occupied(cell: vec3<i32>) -> bool {
    let detail_x = cell.x - i32(u.detail_world.x);
    let detail_z = cell.z - i32(u.detail_world.y);
    if detail_x < 0 || detail_x >= i32(u.detail_world.z) ||
        detail_z < 0 || detail_z >= i32(u.detail_world.w) ||
        cell.y < 0 || cell.y >= i32(u.simulation.z) {
        return false;
    }
    let slot = detailed_physical_slot(detail_x, detail_z);
    let local_x = detail_x % 16;
    let local_z = detail_z % 16;
    let bit = u32(slot * 12288 + local_x + 16 * (local_z + 16 * cell.y));
    return occupancy_is_set(bit);
}

fn detailed_brick_occupied(cell: vec3<i32>) -> bool {
    let detail_x = cell.x * 4 - i32(u.detail_world.x);
    let detail_z = cell.z * 4 - i32(u.detail_world.y);
    let local_y = cell.y;
    if detail_x < 0 || detail_x >= i32(u.detail_world.z) ||
        detail_z < 0 || detail_z >= i32(u.detail_world.w) ||
        local_y < 0 || local_y >= i32(u.simulation.z) / 4 {
        return false;
    }
    let slot = detailed_physical_slot(detail_x, detail_z);
    let local_brick_x = (detail_x % 16) / 4;
    let local_brick_z = (detail_z % 16) / 4;
    let bit_in_chunk = local_brick_x + 4 * (local_brick_z + 4 * local_y);
    let bit = DETAIL_BRICK_OCCUPANCY_OFFSET * 32u
        + u32(slot) * DETAIL_BRICKS_PER_CHUNK + u32(bit_in_chunk);
    return occupancy_is_set(bit);
}

fn detailed_coarse_occupied(cell: vec3<i32>) -> bool {
    let origin_chunk_x = i32(u.detail_world.x) / 16;
    let origin_chunk_z = i32(u.detail_world.y) / 16;
    let detail_chunk_x = cell.x - origin_chunk_x;
    let detail_chunk_z = cell.z - origin_chunk_z;
    if detail_chunk_x < 0 || detail_chunk_x >= i32(u.detail_ring.z) ||
        detail_chunk_z < 0 || detail_chunk_z >= i32(u.detail_ring.z) ||
        cell.y < 0 || cell.y >= i32(u.simulation.z) / 16 {
        return false;
    }
    let physical_chunk_x = (detail_chunk_x + i32(u.detail_ring.x)) % i32(u.detail_ring.z);
    let physical_chunk_z = (detail_chunk_z + i32(u.detail_ring.y)) % i32(u.detail_ring.z);
    let slot = physical_chunk_x + i32(u.detail_ring.z) * physical_chunk_z;
    let bit = DETAIL_COARSE_OCCUPANCY_OFFSET * 32u
        + u32(slot) * DETAIL_COARSE_LAYERS_PER_CHUNK + u32(cell.y);
    return occupancy_is_set(bit);
}

fn leaf_texture_is_opaque(material: u32, point: vec3<f32>, normal: vec3<f32>) -> bool {
    if material < OAK_LEAVES || material > ACACIA_LEAVES {
        return true;
    }
    let alpha = textureSampleLevel(
        tree_texture,
        tree_sampler,
        tree_texture_uv(material, point, normal, vec2<f32>(0.0)),
        0.0,
    ).a;
    return alpha >= 0.5;
}

fn no_lod_column() -> LodColumn {
    return LodColumn(vec2<f32>(0.0), 0.0, 0.0, AIR, AIR, false);
}

// The first *loaded* level that contains the point wins.  While a finer
// full-data source is still building, its already-loaded quadtree parent stays
// visible; there is never a synchronous generation stall or a horizon hole.
fn lod_column_at(point: vec2<f32>) -> LodColumn {
    let grid_side = u.lod_world.x;
    var level = 0u;
    loop {
        if level >= u32(u.lod_world.y) {
            break;
        }
        let info = lod_levels[level];
        let local = point - info.xy;
        let extent = info.z * grid_side;
        if local.x >= 0.0 && local.y >= 0.0 && local.x < extent && local.y < extent {
            let x = u32(floor(local.x / info.z));
            let z = u32(floor(local.y / info.z));
            let index = u32(info.w) + x + u32(grid_side) * z;
            let packed = lod_columns[index];
            if packed != 0u {
                return LodColumn(
                    info.xy + vec2<f32>(f32(x) * info.z, f32(z) * info.z),
                    info.z,
                    f32(packed & 65535u) / 16.0,
                    (packed >> 16u) & 255u,
                    (packed >> 24u) & 255u,
                    true,
                );
            }
        }
        level += 1u;
    }
    return no_lod_column();
}

fn ray_box(ro: vec3<f32>, rd: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>) -> Hit {
    let safe_rd = select(vec3<f32>(0.00001), rd, abs(rd) > vec3<f32>(0.00001));
    let inverse = 1.0 / safe_rd;
    let a = (box_min - ro) * inverse;
    let b = (box_max - ro) * inverse;
    let t_min = min(a, b);
    let t_max = max(a, b);
    let entry = max(max(t_min.x, t_min.y), t_min.z);
    let exit = min(min(t_max.x, t_max.y), t_max.z);
    if exit < max(entry, 0.0) {
        return empty_hit(1.0e30);
    }
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    if entry >= t_min.x && entry >= t_min.y && entry >= t_min.z {
        if entry == t_min.x {
            normal = vec3<f32>(-sign(rd.x), 0.0, 0.0);
        } else if entry == t_min.z {
            normal = vec3<f32>(0.0, 0.0, -sign(rd.z));
        } else {
            normal = vec3<f32>(0.0, -sign(rd.y), 0.0);
        }
    }
    // The far intersection is needed only if a ray starts inside a cell.
    return Hit(select(exit, entry, entry > 0.0001), normal, AIR, vec2<f32>(0.0), true);
}

// This is used only to restrict exact DDA to the loaded detailed window.
// It does not approximate block geometry: each occupied block is still
// intersected with `ray_box` below.
fn ray_aabb_interval(
    ro: vec3<f32>, rd: vec3<f32>, box_min: vec3<f32>, box_max: vec3<f32>
) -> RayInterval {
    let safe_rd = select(vec3<f32>(0.00001), rd, abs(rd) > vec3<f32>(0.00001));
    let inverse = 1.0 / safe_rd;
    let a = (box_min - ro) * inverse;
    let b = (box_max - ro) * inverse;
    let t_min = min(a, b);
    let t_max = max(a, b);
    let entry = max(max(t_min.x, t_min.y), t_min.z);
    let exit = min(min(t_max.x, t_max.y), t_max.z);
    return RayInterval(entry, exit, exit >= max(entry, 0.0));
}

// The three routines below form a nested, exact DDA.  The coarse 16³ and
// brick 4³ maps only skip cells whose child bits prove that they contain air.
// The final loop retains the original slab AABB and foliage alpha test.
fn trace_fine_cells(
    ro: vec3<f32>, rd: vec3<f32>, start_distance: f32, end_distance: f32
) -> Hit {
    let origin_distance = start_distance + 0.0001;
    if origin_distance >= end_distance {
        return empty_hit(end_distance);
    }
    let local_ro = ro + rd * origin_distance;
    let safe_rd = select(vec3<f32>(0.00001), rd, abs(rd) > vec3<f32>(0.00001));
    let delta = abs(1.0 / safe_rd);
    var cell = vec3<i32>(floor(local_ro));
    let step = select(vec3<i32>(-1), vec3<i32>(1), rd >= vec3<f32>(0.0));
    var side = vec3<f32>(0.0);
    if rd.x >= 0.0 {
        side.x = origin_distance + (f32(cell.x + 1) - local_ro.x) * delta.x;
    } else {
        side.x = origin_distance + (local_ro.x - f32(cell.x)) * delta.x;
    }
    if rd.y >= 0.0 {
        side.y = origin_distance + (f32(cell.y + 1) - local_ro.y) * delta.y;
    } else {
        side.y = origin_distance + (local_ro.y - f32(cell.y)) * delta.y;
    }
    if rd.z >= 0.0 {
        side.z = origin_distance + (f32(cell.z + 1) - local_ro.z) * delta.z;
    } else {
        side.z = origin_distance + (local_ro.z - f32(cell.z)) * delta.z;
    }

    var entered = start_distance;
    var step_count = 0u;
    loop {
        if step_count >= MAX_DETAILED_FINE_STEPS || entered > end_distance {
            break;
        }
        let cell_exit = min(side.x, min(side.y, side.z));
        if detailed_fine_occupied(cell) {
            let packed = block_at(cell);
            let material = packed & 255u;
            let height = f32((packed >> 8u) & 255u) / 16.0;
            let candidate = ray_box(
                ro,
                rd,
                vec3<f32>(f32(cell.x), f32(cell.y), f32(cell.z)),
                vec3<f32>(f32(cell.x) + 1.0, f32(cell.y) + height, f32(cell.z) + 1.0),
            );
            if candidate.found && candidate.t >= entered - 0.002 &&
                candidate.t <= cell_exit + 0.002 && candidate.t < end_distance {
                let point = ro + rd * candidate.t;
                if leaf_texture_is_opaque(material, point, candidate.normal) {
                    return Hit(candidate.t, candidate.normal, material, vec2<f32>(0.0), true);
                }
            }
        }
        entered = cell_exit;
        if side.x <= side.y && side.x <= side.z {
            side.x += delta.x;
            cell.x += step.x;
        } else if side.y <= side.z {
            side.y += delta.y;
            cell.y += step.y;
        } else {
            side.z += delta.z;
            cell.z += step.z;
        }
        step_count += 1u;
    }
    return empty_hit(end_distance);
}

fn trace_occupied_bricks(
    ro: vec3<f32>, rd: vec3<f32>, start_distance: f32, end_distance: f32
) -> Hit {
    let origin_distance = start_distance + 0.0001;
    if origin_distance >= end_distance {
        return empty_hit(end_distance);
    }
    let local_ro = ro + rd * origin_distance;
    let safe_rd = select(vec3<f32>(0.00001), rd, abs(rd) > vec3<f32>(0.00001));
    let delta = abs(4.0 / safe_rd);
    var cell = vec3<i32>(floor(local_ro / 4.0));
    let step = select(vec3<i32>(-1), vec3<i32>(1), rd >= vec3<f32>(0.0));
    var side = vec3<f32>(0.0);
    if rd.x >= 0.0 {
        side.x = origin_distance + (f32(cell.x + 1) * 4.0 - local_ro.x) / safe_rd.x;
    } else {
        side.x = origin_distance + (local_ro.x - f32(cell.x) * 4.0) / -safe_rd.x;
    }
    if rd.y >= 0.0 {
        side.y = origin_distance + (f32(cell.y + 1) * 4.0 - local_ro.y) / safe_rd.y;
    } else {
        side.y = origin_distance + (local_ro.y - f32(cell.y) * 4.0) / -safe_rd.y;
    }
    if rd.z >= 0.0 {
        side.z = origin_distance + (f32(cell.z + 1) * 4.0 - local_ro.z) / safe_rd.z;
    } else {
        side.z = origin_distance + (local_ro.z - f32(cell.z) * 4.0) / -safe_rd.z;
    }

    var entered = start_distance;
    var step_count = 0u;
    loop {
        if step_count >= MAX_DETAILED_BRICK_STEPS || entered > end_distance {
            break;
        }
        let cell_exit = min(side.x, min(side.y, side.z));
        if detailed_brick_occupied(cell) {
            let fine_hit = trace_fine_cells(ro, rd, entered, min(cell_exit, end_distance));
            if fine_hit.found {
                return fine_hit;
            }
        }
        entered = cell_exit;
        if side.x <= side.y && side.x <= side.z {
            side.x += delta.x;
            cell.x += step.x;
        } else if side.y <= side.z {
            side.y += delta.y;
            cell.y += step.y;
        } else {
            side.z += delta.z;
            cell.z += step.z;
        }
        step_count += 1u;
    }
    return empty_hit(end_distance);
}

fn trace_blocks(ro: vec3<f32>, rd: vec3<f32>, max_distance: f32) -> Hit {
    let detail_minimum = vec3<f32>(u.detail_world.x, 0.0, u.detail_world.y);
    let detail_maximum = detail_minimum + vec3<f32>(
        u.detail_world.z, u.simulation.z, u.detail_world.w,
    );
    let detail_interval = ray_aabb_interval(ro, rd, detail_minimum, detail_maximum);
    if !detail_interval.found {
        return empty_hit(max_distance);
    }
    let end_distance = min(detail_interval.exit, max_distance);
    let origin_distance = max(detail_interval.entry, 0.0) + 0.0001;
    if origin_distance >= end_distance {
        return empty_hit(max_distance);
    }
    let local_ro = ro + rd * origin_distance;
    let safe_rd = select(vec3<f32>(0.00001), rd, abs(rd) > vec3<f32>(0.00001));
    let delta = abs(16.0 / safe_rd);
    var cell = vec3<i32>(floor(local_ro / 16.0));
    let step = select(vec3<i32>(-1), vec3<i32>(1), rd >= vec3<f32>(0.0));
    var side = vec3<f32>(0.0);
    if rd.x >= 0.0 {
        side.x = origin_distance + (f32(cell.x + 1) * 16.0 - local_ro.x) / safe_rd.x;
    } else {
        side.x = origin_distance + (local_ro.x - f32(cell.x) * 16.0) / -safe_rd.x;
    }
    if rd.y >= 0.0 {
        side.y = origin_distance + (f32(cell.y + 1) * 16.0 - local_ro.y) / safe_rd.y;
    } else {
        side.y = origin_distance + (local_ro.y - f32(cell.y) * 16.0) / -safe_rd.y;
    }
    if rd.z >= 0.0 {
        side.z = origin_distance + (f32(cell.z + 1) * 16.0 - local_ro.z) / safe_rd.z;
    } else {
        side.z = origin_distance + (local_ro.z - f32(cell.z) * 16.0) / -safe_rd.z;
    }

    var entered = origin_distance;
    var step_count = 0u;
    loop {
        if step_count >= MAX_DETAILED_COARSE_STEPS || entered > end_distance {
            break;
        }
        let cell_exit = min(side.x, min(side.y, side.z));
        if detailed_coarse_occupied(cell) {
            let brick_hit = trace_occupied_bricks(ro, rd, entered, min(cell_exit, end_distance));
            if brick_hit.found {
                return brick_hit;
            }
        }
        entered = cell_exit;
        if side.x <= side.y && side.x <= side.z {
            side.x += delta.x;
            cell.x += step.x;
        } else if side.y <= side.z {
            side.y += delta.y;
            cell.y += step.y;
        } else {
            side.z += delta.z;
            cell.z += step.z;
        }
        step_count += 1u;
    }
    return empty_hit(max_distance);
}

fn point_is_inside_detailed_world(point: vec2<f32>) -> bool {
    let minimum = u.detail_world.xy;
    let maximum = minimum + u.detail_world.zw;
    return point.x >= minimum.x && point.y >= minimum.y &&
        point.x < maximum.x && point.y < maximum.y;
}

// Returns the positive distance to leave the detailed X/Z rectangle. It is
// used only when the ray starts inside it, so the first exiting axis is enough.
fn detailed_world_exit_distance(point: vec2<f32>, direction: vec2<f32>) -> f32 {
    let minimum = u.detail_world.xy;
    let maximum = minimum + u.detail_world.zw;
    var exit_x = 1.0e30;
    var exit_z = 1.0e30;
    if direction.x > 0.00001 {
        exit_x = (maximum.x - point.x) / direction.x;
    } else if direction.x < -0.00001 {
        exit_x = (minimum.x - point.x) / direction.x;
    }
    if direction.y > 0.00001 {
        exit_z = (maximum.y - point.y) / direction.y;
    } else if direction.y < -0.00001 {
        exit_z = (minimum.y - point.y) / direction.y;
    }
    return min(exit_x, exit_z);
}

// Traverses the current clipmap cell and then selects a coarser grid as the
// ray leaves each level.  The clipmap deliberately has a hole over the exact
// 9×9 detailed window: the two representations must never overlap there.
// Otherwise a low detailed slab can start *inside* a taller averaged LOD box,
// producing a false self-shadow.
fn trace_lod(ro: vec3<f32>, rd: vec3<f32>, max_distance: f32) -> Hit {
    if abs(rd.x) + abs(rd.z) < 0.002 {
        return empty_hit(max_distance);
    }
    var travelled = 0.0;
    var step_count = 0u;
    loop {
        if step_count >= MAX_LOD_STEPS || travelled > max_distance {
            break;
        }
        let point = ro + rd * travelled;
        if point_is_inside_detailed_world(point.xz) {
            let exit_distance = detailed_world_exit_distance(point.xz, rd.xz);
            if exit_distance >= 1.0e29 {
                break;
            }
            travelled += max(exit_distance, 0.0) + 0.002;
            step_count += 1u;
            continue;
        }
        let column = lod_column_at(point.xz);
        if !column.found {
            break;
        }
        let candidate = ray_box(
            ro,
            rd,
            vec3<f32>(column.minimum.x, 0.0, column.minimum.y),
            vec3<f32>(
                column.minimum.x + column.cell_size,
                column.height,
                column.minimum.y + column.cell_size,
            ),
        );
        if candidate.found && candidate.t >= travelled - 0.01 && candidate.t < max_distance {
            let material = select(
                column.side_material,
                column.top_material,
                candidate.normal.y > 0.5,
            );
            return Hit(candidate.t, candidate.normal, material, vec2<f32>(0.0), true);
        }
        var exit_x = 1.0e30;
        var exit_z = 1.0e30;
        if rd.x > 0.00001 {
            exit_x = (column.minimum.x + column.cell_size - point.x) / rd.x;
        } else if rd.x < -0.00001 {
            exit_x = (column.minimum.x - point.x) / rd.x;
        }
        if rd.z > 0.00001 {
            exit_z = (column.minimum.y + column.cell_size - point.z) / rd.z;
        } else if rd.z < -0.00001 {
            exit_z = (column.minimum.y - point.z) / rd.z;
        }
        travelled += max(min(exit_x, exit_z), 0.001) + 0.002;
        step_count += 1u;
    }
    return empty_hit(max_distance);
}

fn ray_aabb_entry(ro: vec3<f32>, rd: vec3<f32>, minimum: vec3<f32>, maximum: vec3<f32>) -> f32 {
    let safe_rd = select(vec3<f32>(0.00001), rd, abs(rd) > vec3<f32>(0.00001));
    let inverse = 1.0 / safe_rd;
    let a = (minimum - ro) * inverse;
    let b = (maximum - ro) * inverse;
    let entry = max(max(min(a, b).x, min(a, b).y), min(a, b).z);
    let exit = min(min(max(a, b).x, max(a, b).y), max(a, b).z);
    if exit < max(entry, 0.0) {
        return 1.0e30;
    }
    return max(entry, 0.0);
}

fn bark_material(form: u32, end_grain: bool) -> u32 {
    if form == 1u {
        return select(SPRUCE_BARK, SPRUCE_RINGS, end_grain);
    }
    if form == 2u {
        return select(ACACIA_BARK, ACACIA_RINGS, end_grain);
    }
    return select(OAK_BARK, OAK_RINGS, end_grain);
}

// Unwrap the four side faces of an oriented rectangular branch. `v` follows
// the actual segment axis and uses a world-stable phase, so adjacent vertical,
// horizontal, and diagonal spines do not inherit the old world-Y banding.
fn prism_bark_uv(
    local_point: vec3<f32>, local_normal: vec3<f32>, half_width: f32,
    point: vec3<f32>, unit_axis: vec3<f32>,
) -> vec2<f32> {
    let side_span = max(half_width * 2.0, 0.00001);
    var around = 0.0;
    if local_normal.x > 0.5 {
        around = (local_point.z + half_width) / side_span * 0.25;
    } else if local_normal.z > 0.5 {
        around = 0.25 + (local_point.x + half_width) / side_span * 0.25;
    } else if local_normal.x < -0.5 {
        around = 0.50 + (half_width - local_point.z) / side_span * 0.25;
    } else {
        around = 0.75 + (half_width - local_point.x) / side_span * 0.25;
    }
    return vec2<f32>(fract(around), fract(dot(point, unit_axis) * 0.45));
}

fn prism_end_grain_uv(local_point: vec3<f32>, half_width: f32) -> vec2<f32> {
    let side_span = max(half_width * 2.0, 0.00001);
    return vec2<f32>(
        (local_point.x + half_width) / side_span,
        (local_point.z + half_width) / side_span,
    );
}

// Intersect one of the visualizer's constant-width rectangles. Its plane is
// lifted to a square prism for ray tracing, preserving its angular 1/16-step
// silhouette instead of smoothing it into a pipe.
fn trace_hpd_prism(ro: vec3<f32>, rd: vec3<f32>, segment: TreeSegment) -> Hit {
    let start = segment.start_radius.xyz;
    let end = segment.end_radius.xyz;
    let half_width = segment.start_radius.w;
    let axis = end - start;
    let segment_length = length(axis);
    if segment_length < 0.0001 {
        return empty_hit(1.0e30);
    }
    let unit_axis = axis / segment_length;
    let reference = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(unit_axis.y) > 0.92,
    );
    let side = normalize(cross(unit_axis, reference));
    let up = cross(side, unit_axis);
    let relative_origin = ro - start;
    let local_origin = vec3<f32>(
        dot(relative_origin, side),
        dot(relative_origin, unit_axis),
        dot(relative_origin, up),
    );
    let local_ray = vec3<f32>(dot(rd, side), dot(rd, unit_axis), dot(rd, up));
    let local_hit = ray_box(
        local_origin,
        local_ray,
        vec3<f32>(-half_width, 0.0, -half_width),
        vec3<f32>(half_width, segment_length, half_width),
    );
    if !local_hit.found {
        return local_hit;
    }
    let point = ro + rd * local_hit.t;
    let local_point = local_origin + local_ray * local_hit.t;
    let end_grain = abs(local_hit.normal.y) > 0.5;
    let normal = normalize(
        side * local_hit.normal.x + unit_axis * local_hit.normal.y + up * local_hit.normal.z,
    );
    return Hit(
        local_hit.t,
        normal,
        bark_material(segment.style[0u], end_grain),
        select(
            prism_bark_uv(local_point, local_hit.normal, half_width, point, unit_axis),
            prism_end_grain_uv(local_point, half_width),
            end_grain,
        ),
        true,
    );
}

// `texturedRect(p1, p2, image)` from visualizer.js: a texture alpha-tested
// square starts at spineIn, runs to its foliage centre, and is camera-facing
// only because the original 2D canvas has one fixed viewing plane.
fn trace_foliage_connector(ro: vec3<f32>, rd: vec3<f32>, segment: TreeSegment) -> Hit {
    let start = segment.start_radius.xyz;
    let end = segment.end_radius.xyz;
    let axis = end - start;
    let length_on_axis = length(axis);
    if length_on_axis < 0.0001 {
        return empty_hit(1.0e30);
    }
    let unit_axis = axis / length_on_axis;
    let side_seed = cross(unit_axis, u.camera_forward.xyz);
    let backup_seed = cross(unit_axis, u.camera_right.xyz);
    let side = normalize(select(backup_seed, side_seed, length(side_seed) > 0.0001));
    var normal = normalize(cross(unit_axis, side));
    let denominator = dot(rd, normal);
    if abs(denominator) < 0.00001 {
        return empty_hit(1.0e30);
    }
    let t = dot(start - ro, normal) / denominator;
    if t <= 0.0001 {
        return empty_hit(1.0e30);
    }
    let point = ro + rd * t;
    let local = point - start;
    let along = dot(local, unit_axis);
    let across = dot(local, side);
    if along < 0.0 || along > length_on_axis || abs(across) > length_on_axis * 0.5 {
        return empty_hit(1.0e30);
    }
    let uv = vec2<f32>(along / length_on_axis, across / length_on_axis + 0.5);
    let connector_material = FOLIAGE_CONNECTOR_0 + (segment.style[2u] % 6u);
    let alpha = textureSampleLevel(
        tree_texture,
        tree_sampler,
        tree_texture_uv(connector_material, point, normal, uv),
        0.0,
    ).a;
    if alpha < 0.25 {
        return empty_hit(1.0e30);
    }
    if dot(normal, rd) > 0.0 {
        normal = -normal;
    }
    return Hit(t, normal, connector_material, uv, true);
}

fn trace_tree_blas(ro: vec3<f32>, rd: vec3<f32>, root: u32, max_distance: f32) -> Hit {
    var closest = empty_hit(max_distance);
    var stack: array<u32, TREE_BVH_STACK_CAPACITY>;
    var stack_size = 1u;
    stack[0] = root;
    loop {
        if stack_size == 0u {
            break;
        }
        stack_size -= 1u;
        let node = tree_blas_nodes[stack[stack_size]];
        if ray_aabb_entry(ro, rd, node.minimum.xyz, node.maximum.xyz) >= closest.t {
            continue;
        }
        if node.data[2u] == 1u {
            var segment = 0u;
            loop {
                if segment >= node.data[1u] {
                    break;
                }
                let tree_segment = tree_segments[node.data[0u] + segment];
                var wood = trace_hpd_prism(ro, rd, tree_segment);
                if tree_segment.style[1u] == 2u {
                    wood = trace_foliage_connector(ro, rd, tree_segment);
                }
                if wood.found && wood.t < closest.t {
                    closest = wood;
                }
                segment += 1u;
            }
            continue;
        }
        let left = node.data[0u];
        let right = node.data[1u];
        let left_entry = ray_aabb_entry(
            ro, rd, tree_blas_nodes[left].minimum.xyz, tree_blas_nodes[left].maximum.xyz,
        );
        let right_entry = ray_aabb_entry(
            ro, rd, tree_blas_nodes[right].minimum.xyz, tree_blas_nodes[right].maximum.xyz,
        );
        if left_entry < closest.t && right_entry < closest.t {
            if left_entry <= right_entry {
                stack[stack_size] = right;
                stack[stack_size + 1u] = left;
            } else {
                stack[stack_size] = left;
                stack[stack_size + 1u] = right;
            }
            stack_size += 2u;
        } else if left_entry < closest.t {
            stack[stack_size] = left;
            stack_size += 1u;
        } else if right_entry < closest.t {
            stack[stack_size] = right;
            stack_size += 1u;
        }
    }
    return closest;
}

fn trace_trees(ro: vec3<f32>, rd: vec3<f32>, max_distance: f32) -> Hit {
    let root_token = u32(u.simulation.y);
    if root_token == 0u {
        return empty_hit(max_distance);
    }
    var closest = empty_hit(max_distance);
    var stack: array<u32, TREE_BVH_STACK_CAPACITY>;
    var stack_size = 1u;
    stack[0] = root_token - 1u;
    loop {
        if stack_size == 0u {
            break;
        }
        stack_size -= 1u;
        let node = tree_tlas_nodes[stack[stack_size]];
        if ray_aabb_entry(ro, rd, node.minimum.xyz, node.maximum.xyz) >= closest.t {
            continue;
        }
        if node.data[2u] == 1u {
            let ranges = trees[node.data[0u] * 3u + 1u];
            let tree_hit = trace_tree_blas(ro, rd, u32(ranges.z), closest.t);
            if tree_hit.found && tree_hit.t < closest.t {
                closest = tree_hit;
            }
            continue;
        }
        let left = node.data[0u];
        let right = node.data[1u];
        let left_entry = ray_aabb_entry(
            ro, rd, tree_tlas_nodes[left].minimum.xyz, tree_tlas_nodes[left].maximum.xyz,
        );
        let right_entry = ray_aabb_entry(
            ro, rd, tree_tlas_nodes[right].minimum.xyz, tree_tlas_nodes[right].maximum.xyz,
        );
        if left_entry < closest.t && right_entry < closest.t {
            if left_entry <= right_entry {
                stack[stack_size] = right;
                stack[stack_size + 1u] = left;
            } else {
                stack[stack_size] = left;
                stack[stack_size + 1u] = right;
            }
            stack_size += 2u;
        } else if left_entry < closest.t {
            stack[stack_size] = left;
            stack_size += 1u;
        } else if right_entry < closest.t {
            stack[stack_size] = right;
            stack_size += 1u;
        }
    }
    return closest;
}

// Shadows need only visibility, not the nearest intersection, so this path
// does no child sorting and returns immediately after the first exact prism or
// alpha-tested connector hit.  Its primitives and texture test are identical
// to the primary-ray path.
fn tree_blas_any_hit(ro: vec3<f32>, rd: vec3<f32>, root: u32, max_distance: f32) -> bool {
    var stack: array<u32, TREE_BVH_STACK_CAPACITY>;
    var stack_size = 1u;
    stack[0] = root;
    loop {
        if stack_size == 0u {
            break;
        }
        stack_size -= 1u;
        let node = tree_blas_nodes[stack[stack_size]];
        if ray_aabb_entry(ro, rd, node.minimum.xyz, node.maximum.xyz) >= max_distance {
            continue;
        }
        if node.data[2u] == 1u {
            var segment = 0u;
            loop {
                if segment >= node.data[1u] {
                    break;
                }
                let tree_segment = tree_segments[node.data[0u] + segment];
                var candidate = trace_hpd_prism(ro, rd, tree_segment);
                if tree_segment.style[1u] == 2u {
                    candidate = trace_foliage_connector(ro, rd, tree_segment);
                }
                if candidate.found && candidate.t < max_distance {
                    return true;
                }
                segment += 1u;
            }
            continue;
        }
        let left = node.data[0u];
        let right = node.data[1u];
        if ray_aabb_entry(
            ro, rd, tree_blas_nodes[left].minimum.xyz, tree_blas_nodes[left].maximum.xyz,
        ) < max_distance {
            stack[stack_size] = left;
            stack_size += 1u;
        }
        if ray_aabb_entry(
            ro, rd, tree_blas_nodes[right].minimum.xyz, tree_blas_nodes[right].maximum.xyz,
        ) < max_distance {
            stack[stack_size] = right;
            stack_size += 1u;
        }
    }
    return false;
}

fn trees_any_hit(ro: vec3<f32>, rd: vec3<f32>, max_distance: f32) -> bool {
    let root_token = u32(u.simulation.y);
    if root_token == 0u {
        return false;
    }
    var stack: array<u32, TREE_BVH_STACK_CAPACITY>;
    var stack_size = 1u;
    stack[0] = root_token - 1u;
    loop {
        if stack_size == 0u {
            break;
        }
        stack_size -= 1u;
        let node = tree_tlas_nodes[stack[stack_size]];
        if ray_aabb_entry(ro, rd, node.minimum.xyz, node.maximum.xyz) >= max_distance {
            continue;
        }
        if node.data[2u] == 1u {
            let ranges = trees[node.data[0u] * 3u + 1u];
            if tree_blas_any_hit(ro, rd, u32(ranges.z), max_distance) {
                return true;
            }
            continue;
        }
        let left = node.data[0u];
        let right = node.data[1u];
        if ray_aabb_entry(
            ro, rd, tree_tlas_nodes[left].minimum.xyz, tree_tlas_nodes[left].maximum.xyz,
        ) < max_distance {
            stack[stack_size] = left;
            stack_size += 1u;
        }
        if ray_aabb_entry(
            ro, rd, tree_tlas_nodes[right].minimum.xyz, tree_tlas_nodes[right].maximum.xyz,
        ) < max_distance {
            stack[stack_size] = right;
            stack_size += 1u;
        }
    }
    return false;
}

fn trace_world(ro: vec3<f32>, rd: vec3<f32>, max_distance: f32) -> Hit {
    let near_terrain = trace_blocks(ro, rd, min(max_distance, 180.0));
    var terrain = near_terrain;
    // A nearby exact block always wins.  The clipmap is only queried for rays
    // leaving the detailed window, preventing coarse terrain from masking it.
    if !near_terrain.found {
        terrain = trace_lod(ro, rd, max_distance);
    }
    let trees_hit = trace_trees(ro, rd, terrain.t);
    if trees_hit.found {
        return trees_hit;
    }
    return terrain;
}

fn world_any_hit(ro: vec3<f32>, rd: vec3<f32>, max_distance: f32) -> bool {
    let near_terrain = trace_blocks(ro, rd, min(max_distance, 180.0));
    if near_terrain.found {
        return true;
    }
    let lod_terrain = trace_lod(ro, rd, max_distance);
    if lod_terrain.found {
        return true;
    }
    return trees_any_hit(ro, rd, max_distance);
}

fn sky_colour(direction: vec3<f32>) -> vec3<f32> {
    let day = smoothstep(-0.12, 0.18, u.sun_direction.y);
    let horizon = pow(1.0 - max(direction.y, 0.0), 1.8);
    let daytime = mix(vec3<f32>(0.17, 0.37, 0.72), vec3<f32>(0.59, 0.82, 1.0), max(direction.y, 0.0));
    let night = mix(vec3<f32>(0.008, 0.012, 0.035), vec3<f32>(0.025, 0.045, 0.11), max(direction.y, 0.0));
    var colour = mix(night, daytime, day);
    colour = mix(colour, vec3<f32>(0.95, 0.42, 0.22), horizon * (1.0 - day) * 0.55);
    let sun_disc = pow(max(dot(direction, u.sun_direction.xyz), 0.0), 1100.0);
    colour += vec3<f32>(1.0, 0.78, 0.45) * sun_disc * max(day, 0.02) * 8.0;
    return colour;
}

fn primary_direction_from_screen(screen: vec2<f32>) -> vec3<f32> {
    let focal = 1.18;
    return normalize(
        u.camera_forward.xyz * focal +
        u.camera_right.xyz * screen.x * u.lod_world.w +
        u.camera_up.xyz * screen.y,
    );
}

fn primary_direction(input: VertexOut) -> vec3<f32> {
    return primary_direction_from_screen(input.uv * 2.0 - 1.0);
}

fn pixel_coordinate(input: VertexOut) -> vec2<i32> {
    return vec2<i32>(input.position.xy);
}

fn tree_texture_tile(material: u32) -> vec2<f32> {
    if material >= OAK_LEAVES && material <= ACACIA_LEAVES {
        return vec2<f32>(f32(material - OAK_LEAVES + 3u), 0.0);
    }
    if material >= OAK_BARK && material <= ACACIA_BARK {
        return vec2<f32>(f32(material - OAK_BARK), 0.0);
    }
    if material >= OAK_RINGS && material <= ACACIA_RINGS {
        return vec2<f32>(f32(material - OAK_RINGS), 1.0);
    }
    if material >= FOLIAGE_CONNECTOR_0 && material <= FOLIAGE_CONNECTOR_5 {
        let variant = material - FOLIAGE_CONNECTOR_0;
        return vec2<f32>(f32(3u + variant % 3u), f32(1u + variant / 3u));
    }
    return vec2<f32>(0.0);
}

fn tree_texture_uv(
    material: u32, point: vec3<f32>, normal: vec3<f32>, connector_uv: vec2<f32>,
) -> vec2<f32> {
    let tile = tree_texture_tile(material);
    var local = fract(point);
    var uv = local.xz;
    if material >= FOLIAGE_CONNECTOR_0 && material <= FOLIAGE_CONNECTOR_5 {
        uv = connector_uv;
    } else if material >= OAK_BARK && material <= ACACIA_RINGS {
        // Tree prisms supply local unwrapped bark or end-grain coordinates in
        // Hit. World voxels never use tree-wood materials.
        uv = connector_uv;
    } else if abs(normal.x) > 0.5 {
        uv = local.yz;
    } else if abs(normal.z) > 0.5 {
        uv = local.xy;
    }
    return (tile + vec2<f32>(0.015, 0.015) + uv * 0.97) / vec2<f32>(6.0, 3.0);
}

fn material_colour(
    material: u32, point: vec3<f32>, normal: vec3<f32>, connector_uv: vec2<f32>,
) -> vec3<f32> {
    let variation = 0.92 + 0.08 * sin(point.x * 8.0 + point.z * 5.0 + point.y * 3.0);
    if material == GRASS {
        return vec3<f32>(0.24, 0.52, 0.16) * variation;
    }
    if material == DIRT {
        return vec3<f32>(0.38, 0.22, 0.105) * variation;
    }
    if material == STONE {
        return vec3<f32>(0.34, 0.37, 0.40) * variation;
    }
    if material == WATER {
        // Water is deliberately an opaque, saturated blue voxel material at
        // this stage.  It remains unambiguous in shallow LOD ponds and does
        // not borrow terrain colour through transparency or reflections.
        return vec3<f32>(0.025, 0.16, 0.82) * variation;
    }
    if (material >= OAK_LEAVES && material <= ACACIA_LEAVES)
        || (material >= OAK_BARK && material <= ACACIA_RINGS)
        || (material >= FOLIAGE_CONNECTOR_0 && material <= FOLIAGE_CONNECTOR_5)
    {
        let albedo = textureSampleLevel(
            tree_texture,
            tree_sampler,
            tree_texture_uv(material, point, normal, connector_uv),
            0.0,
        ).rgb;
        return albedo * variation;
    }
    if material == ROOTY_SOIL {
        return vec3<f32>(0.25, 0.14, 0.06) * variation;
    }
    return vec3<f32>(0.12, 0.39, 0.105) * variation;
}

// Selection is actual traced geometry, not a post-process mask: each of the
// 12 AABB edges is a short capsule. The spherical caps merge seamlessly at
// corners, so the wireframe has a stable, rounded silhouette at any view.
const SELECTION_EDGE_PIXEL_DIAMETER: f32 = 1.45;
const SELECTION_EDGE_DEPTH_EPSILON: f32 = 0.003;

fn ray_sphere_distance(
    origin: vec3<f32>, direction: vec3<f32>, centre: vec3<f32>, radius: f32,
) -> f32 {
    let offset = origin - centre;
    let half_b = dot(offset, direction);
    let c = dot(offset, offset) - radius * radius;
    let discriminant = half_b * half_b - c;
    if discriminant < 0.0 {
        return 1.0e30;
    }
    let root = sqrt(discriminant);
    let near = -half_b - root;
    let far = -half_b + root;
    return select(far, near, near > 0.0001);
}

fn ray_capsule_distance(
    origin: vec3<f32>, direction: vec3<f32>, start: vec3<f32>, end: vec3<f32>, radius: f32,
) -> f32 {
    let axis = end - start;
    let axis_squared = dot(axis, axis);
    if axis_squared < 0.000001 {
        return ray_sphere_distance(origin, direction, start, radius);
    }

    let from_start = origin - start;
    let direction_axis = dot(direction, axis);
    let origin_axis = dot(from_start, axis);
    let cylinder_a = axis_squared - direction_axis * direction_axis;
    let cylinder_b = axis_squared * dot(from_start, direction) - origin_axis * direction_axis;
    let cylinder_c = axis_squared * dot(from_start, from_start)
        - origin_axis * origin_axis - radius * radius * axis_squared;
    var closest = min(
        ray_sphere_distance(origin, direction, start, radius),
        ray_sphere_distance(origin, direction, end, radius),
    );
    if cylinder_a <= 0.000001 {
        return closest;
    }
    let discriminant = cylinder_b * cylinder_b - cylinder_a * cylinder_c;
    if discriminant < 0.0 {
        return closest;
    }
    let root = sqrt(discriminant);
    let near = (-cylinder_b - root) / cylinder_a;
    let far = (-cylinder_b + root) / cylinder_a;
    let near_axis = origin_axis + near * direction_axis;
    if near > 0.0001 && near_axis >= 0.0 && near_axis <= axis_squared {
        closest = min(closest, near);
    }
    let far_axis = origin_axis + far * direction_axis;
    if far > 0.0001 && far_axis >= 0.0 && far_axis <= axis_squared {
        closest = min(closest, far);
    }
    return closest;
}

fn selection_outline_sample(direction: vec3<f32>) -> f32 {
    if u.selection_min.w < 0.5 {
        return 0.0;
    }
    let minimum = u.selection_min.xyz;
    let maximum = u.selection_max.xyz;
    let origin = u.camera_position.xyz;
    let screen_size = vec2<f32>(textureDimensions(primary_surface));
    let selection_distance = length((minimum + maximum) * 0.5 - origin);
    let radius = min(
        selection_distance * SELECTION_EDGE_PIXEL_DIAMETER / (screen_size.y * 1.18),
        min(maximum.x - minimum.x, min(maximum.y - minimum.y, maximum.z - minimum.z)) * 0.20,
    );
    let broad_phase = ray_box(
        origin,
        direction,
        minimum - vec3<f32>(radius),
        maximum + vec3<f32>(radius),
    );
    if !broad_phase.found {
        return 0.0;
    }

    let x00 = vec3<f32>(minimum.x, minimum.y, minimum.z);
    let x01 = vec3<f32>(minimum.x, minimum.y, maximum.z);
    let x10 = vec3<f32>(minimum.x, maximum.y, minimum.z);
    let x11 = vec3<f32>(minimum.x, maximum.y, maximum.z);
    let y00 = vec3<f32>(maximum.x, minimum.y, minimum.z);
    let y01 = vec3<f32>(maximum.x, minimum.y, maximum.z);
    let y10 = vec3<f32>(maximum.x, maximum.y, minimum.z);
    let y11 = vec3<f32>(maximum.x, maximum.y, maximum.z);
    var closest = 1.0e30;
    // Four x-axis, four y-axis, and four z-axis edges.
    closest = min(closest, ray_capsule_distance(origin, direction, x00, y00, radius));
    closest = min(closest, ray_capsule_distance(origin, direction, x01, y01, radius));
    closest = min(closest, ray_capsule_distance(origin, direction, x10, y10, radius));
    closest = min(closest, ray_capsule_distance(origin, direction, x11, y11, radius));
    closest = min(closest, ray_capsule_distance(origin, direction, x00, x10, radius));
    closest = min(closest, ray_capsule_distance(origin, direction, x01, x11, radius));
    closest = min(closest, ray_capsule_distance(origin, direction, y00, y10, radius));
    closest = min(closest, ray_capsule_distance(origin, direction, y01, y11, radius));
    closest = min(closest, ray_capsule_distance(origin, direction, x00, x01, radius));
    closest = min(closest, ray_capsule_distance(origin, direction, x10, x11, radius));
    closest = min(closest, ray_capsule_distance(origin, direction, y00, y01, radius));
    closest = min(closest, ray_capsule_distance(origin, direction, y10, y11, radius));
    if closest >= 1.0e29 {
        return 0.0;
    }
    // The selection tube is intentionally just outside the selected voxel.
    // A tiny depth allowance prevents its own face from hiding the outline,
    // while an actual foreground block still wins the test.
    let occlusion_distance = max(closest - max(radius * 2.0, SELECTION_EDGE_DEPTH_EPSILON), 0.0001);
    let occluder = trace_world(origin, direction, occlusion_distance);
    return select(1.0, 0.0, occluder.found);
}

// A 4x rotated-grid MSAA pattern. The world itself remains one primary ray per
// pixel; only the extremely small selection-wireframe footprint receives four
// exact sub-pixel rays and depth tests. This is coverage AA, not fwidth blur.
fn selection_outline_alpha(input: VertexOut) -> f32 {
    if u.selection_min.w < 0.5 {
        return 0.0;
    }
    let screen = input.uv * 2.0 - 1.0;
    let screen_size = vec2<f32>(textureDimensions(primary_surface));
    let pixel_span = vec2<f32>(2.0) / screen_size;
    let sample_0 = primary_direction_from_screen(
        screen + pixel_span * vec2<f32>(-0.375, -0.125),
    );
    let sample_1 = primary_direction_from_screen(
        screen + pixel_span * vec2<f32>(0.125, -0.375),
    );
    let sample_2 = primary_direction_from_screen(
        screen + pixel_span * vec2<f32>(0.375, 0.125),
    );
    let sample_3 = primary_direction_from_screen(
        screen + pixel_span * vec2<f32>(-0.125, 0.375),
    );
    return 0.25 * (
        selection_outline_sample(sample_0) +
        selection_outline_sample(sample_1) +
        selection_outline_sample(sample_2) +
        selection_outline_sample(sample_3)
    );
}

@fragment
fn fs_primary(input: VertexOut) -> PrimaryOut {
    let direction = primary_direction(input);
    let hit = trace_world(u.camera_position.xyz, direction, 4600.0);
    if !hit.found {
        return PrimaryOut(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    return PrimaryOut(
        vec4<f32>(hit.t, hit.normal),
        vec4<f32>(hit.texture_uv, f32(hit.material), 1.0),
    );
}

@fragment
fn fs_sun_shadow(input: VertexOut) -> @location(0) u32 {
    let pixel = pixel_coordinate(input);
    let surface = textureLoad(primary_surface, pixel, 0);
    // Ambient-only deliberately avoids every sun/shadow traversal. The empty
    // pass remains so GPU timestamp profiling keeps its stable three-stage
    // layout, but it performs no world intersection work.
    if u.simulation.w > 0.5 || surface.w < 0.5 || max(u.sun_direction.y, 0.0) <= 0.015 {
        return 0u;
    }
    let geometry = textureLoad(primary_geometry, pixel, 0);
    let direction = primary_direction(input);
    let point = u.camera_position.xyz + direction * geometry.x;
    return select(
        0u,
        1u,
        world_any_hit(point + geometry.yzw * 0.025, u.sun_direction.xyz, 85.0),
    );
}

@fragment
fn fs_lighting(input: VertexOut) -> @location(0) vec4<f32> {
    let direction = primary_direction(input);
    let pixel = pixel_coordinate(input);
    let surface = textureLoad(primary_surface, pixel, 0);
    let ambient_only = u.simulation.w > 0.5;
    var colour = vec3<f32>(0.0);
    var scene_distance = 1.0e30;
    if surface.w < 0.5 {
        // A static sky prevents time of day or sun direction from tinting the
        // diagnostic ambient-only view.
        colour = select(
            sky_colour(direction),
            vec3<f32>(0.53, 0.70, 0.85),
            ambient_only,
        );
    } else {
        let geometry = textureLoad(primary_geometry, pixel, 0);
        scene_distance = geometry.x;
        let normal = geometry.yzw;
        let material = u32(surface.z);
        let point = u.camera_position.xyz + direction * scene_distance;
        if ambient_only {
            colour = material_colour(material, point, normal, surface.xy);
        } else {
            let sunlight = max(u.sun_direction.y, 0.0);
            let shadow = select(1.0, 0.24, textureLoad(shadow_mask, pixel, 0).x != 0u);
            let lambert = max(dot(normal, u.sun_direction.xyz), 0.0);
            let light = 0.16 + sunlight * (0.24 + 0.76 * lambert * shadow);
            colour = material_colour(material, point, normal, surface.xy) * light;
            let fog = smoothstep(700.0, 4200.0, scene_distance);
            colour = mix(colour, sky_colour(direction), fog);
        }
    }
    let display_colour = pow(max(colour, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    let outline_alpha = selection_outline_alpha(input);
    return vec4<f32>(mix(display_colour, vec3<f32>(0.0), outline_alpha), 1.0);
}
