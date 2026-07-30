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
const MAX_STEPS: u32 = 160u;
const MAX_LOD_STEPS: u32 = 180u;
const TREE_BVH_STACK_CAPACITY: u32 = 32u;

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
    // Choose the side of an integer boundary that lies inside the AABB.  This
    // prevents a negative-direction entry from testing the just-exited cell.
    let origin_distance = max(detail_interval.entry, 0.0) + 0.0001;
    if origin_distance >= end_distance {
        return empty_hit(max_distance);
    }
    let local_ro = ro + rd * origin_distance;
    let safe_rd = select(vec3<f32>(0.00001), rd, abs(rd) > vec3<f32>(0.00001));
    let delta = abs(1.0 / safe_rd);
    var cell = vec3<i32>(floor(local_ro));
    let step = select(vec3<i32>(-1), vec3<i32>(1), rd >= vec3<f32>(0.0));
    var side = vec3<f32>(0.0);
    if rd.x >= 0.0 {
        side.x = (f32(cell.x + 1) - local_ro.x) * delta.x;
    } else {
        side.x = (local_ro.x - f32(cell.x)) * delta.x;
    }
    if rd.y >= 0.0 {
        side.y = (f32(cell.y + 1) - local_ro.y) * delta.y;
    } else {
        side.y = (local_ro.y - f32(cell.y)) * delta.y;
    }
    if rd.z >= 0.0 {
        side.z = (f32(cell.z + 1) - local_ro.z) * delta.z;
    } else {
        side.z = (local_ro.z - f32(cell.z)) * delta.z;
    }

    var entered = origin_distance;
    var step_count = 0u;
    loop {
        if step_count >= MAX_STEPS {
            break;
        }
        let cell_exit = origin_distance + min(side.x, min(side.y, side.z));
        if entered > end_distance {
            break;
        }
        let packed = block_at(cell);
        if packed != AIR {
            let material = packed & 255u;
            let height = f32((packed >> 8u) & 255u) / 16.0;
            let candidate = ray_box(
                local_ro,
                rd,
                vec3<f32>(f32(cell.x), f32(cell.y), f32(cell.z)),
                vec3<f32>(f32(cell.x) + 1.0, f32(cell.y) + height, f32(cell.z) + 1.0),
            );
            let hit_distance = origin_distance + candidate.t;
            if candidate.found && hit_distance >= entered - 0.002 && hit_distance <= cell_exit + 0.002 && hit_distance < end_distance {
                let point = ro + rd * hit_distance;
                // Alpha-tested foliage must be transparent to both the camera
                // ray and the sun ray. Returning no hit lets DDA advance past
                // the leaf cell instead of creating opaque square canopies.
                if leaf_texture_is_opaque(material, point, candidate.normal) {
                    return Hit(hit_distance, candidate.normal, material, vec2<f32>(0.0), true);
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
    return empty_hit(max_distance);
}

// Traverses the current clipmap cell and then selects a coarser grid as the
// ray leaves each level.  This is the important difference from a single flat
// height ring: work grows with visible LOD transitions, not horizon distance.
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
        return vec3<f32>(0.06, 0.27, 0.47) * variation;
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

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    let screen = input.uv * 2.0 - 1.0;
    let focal = 1.18;
    let direction = normalize(
        u.camera_forward.xyz * focal +
        u.camera_right.xyz * screen.x * u.lod_world.w +
        u.camera_up.xyz * screen.y,
    );
    let hit = trace_world(u.camera_position.xyz, direction, 4600.0);
    if !hit.found {
        let sky = sky_colour(direction);
        return vec4<f32>(pow(sky, vec3<f32>(1.0 / 2.2)), 1.0);
    }

    let point = u.camera_position.xyz + direction * hit.t;
    let sunlight = max(u.sun_direction.y, 0.0);
    var shadow = 1.0;
    if sunlight > 0.015 {
        let shadow_hit = trace_world(point + hit.normal * 0.025, u.sun_direction.xyz, 85.0);
        if shadow_hit.found {
            shadow = 0.24;
        }
    }
    let lambert = max(dot(hit.normal, u.sun_direction.xyz), 0.0);
    let light = 0.16 + sunlight * (0.24 + 0.76 * lambert * shadow);
    var colour = material_colour(hit.material, point, hit.normal, hit.texture_uv) * light;
    if hit.material == WATER {
        colour += vec3<f32>(0.12, 0.22, 0.25) * pow(max(dot(reflect(direction, hit.normal), u.sun_direction.xyz), 0.0), 30.0);
    }
    let fog = smoothstep(700.0, 4200.0, hit.t);
    colour = mix(colour, sky_colour(direction), fog);
    return vec4<f32>(pow(max(colour, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
}
