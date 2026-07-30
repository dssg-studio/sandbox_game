// Full-screen voxel ray tracer.  Chunk data is a packed storage buffer rather
// than a mesh: this lets a block hold an exact 1/16-th height slab.

struct Uniforms {
    camera_position: vec4<f32>,
    camera_forward: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    sun_direction: vec4<f32>,
    detail_world: vec4<f32>,
    lod_world: vec4<f32>,
    simulation: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> detail_blocks: array<u32>;
@group(0) @binding(2) var<storage, read> lod_heights: array<u32>;
// Three vec4s per tree: bounds, branch range, appearance.
@group(0) @binding(3) var<storage, read> trees: array<vec4<f32>>;
// [world origin x, world origin z, cell size, height-buffer offset] per level.
@group(0) @binding(4) var<storage, read> lod_levels: array<vec4<f32>>;
// A Dynamic Trees branch is a square core plus up to six square sleeves.
struct BranchCell {
    // World centre xyz and discrete block radius / 16.
    centre_radius: vec4<f32>,
    // Direction order: down, up, north, south, west, east; two padding cells.
    connections: array<u32, 8>,
};
@group(0) @binding(5) var<storage, read> branches: array<BranchCell>;

const AIR: u32 = 0u;
const GRASS: u32 = 1u;
const DIRT: u32 = 2u;
const STONE: u32 = 3u;
const WATER: u32 = 4u;
const WOOD: u32 = 5u;
const LEAVES: u32 = 6u;
const DIM_LEAVES: u32 = 7u;
const WOOD_RINGS: u32 = 8u;
const ROOTY_SOIL: u32 = 9u;
const MAX_STEPS: u32 = 160u;
const MAX_LOD_STEPS: u32 = 180u;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct Hit {
    t: f32,
    normal: vec3<f32>,
    material: u32,
    found: bool,
};

struct LodColumn {
    minimum: vec2<f32>,
    cell_size: f32,
    height: f32,
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
    return Hit(max_distance, vec3<f32>(0.0), AIR, false);
}

fn opposite_direction(direction: u32) -> u32 {
    if direction == 0u { return 1u; }
    if direction == 1u { return 0u; }
    if direction == 2u { return 3u; }
    if direction == 3u { return 2u; }
    if direction == 4u { return 5u; }
    return 4u;
}

fn block_at(cell: vec3<i32>) -> u32 {
    let detail_x = cell.x - i32(u.detail_world.x);
    let detail_z = cell.z - i32(u.detail_world.y);
    let detail_width = i32(u.detail_world.z);
    let detail_depth = i32(u.detail_world.w);
    if detail_x >= 0 && detail_x < detail_width &&
        detail_z >= 0 && detail_z < detail_depth &&
        cell.y >= 0 && cell.y < i32(u.simulation.z) {
        let index = u32(detail_x) + u32(detail_width) *
            (u32(detail_z) + u32(detail_depth) * u32(cell.y));
        return detail_blocks[index];
    }

    return AIR;
}

fn no_lod_column() -> LodColumn {
    return LodColumn(vec2<f32>(0.0), 0.0, 0.0, false);
}

// The first clipmap that contains the point wins, therefore detail falls off
// in successive 16m, 32m, 64m, 128m and 256m cells up to a 256-chunk radius.
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
            return LodColumn(
                info.xy + vec2<f32>(f32(x) * info.z, f32(z) * info.z),
                info.z,
                f32(lod_heights[index]) / 16.0,
                true,
            );
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
    return Hit(select(exit, entry, entry > 0.0001), normal, AIR, true);
}

fn trace_blocks(ro: vec3<f32>, rd: vec3<f32>, max_distance: f32) -> Hit {
    let safe_rd = select(vec3<f32>(0.00001), rd, abs(rd) > vec3<f32>(0.00001));
    let delta = abs(1.0 / safe_rd);
    var cell = vec3<i32>(floor(ro));
    let step = select(vec3<i32>(-1), vec3<i32>(1), rd >= vec3<f32>(0.0));
    var side = vec3<f32>(0.0);
    if rd.x >= 0.0 {
        side.x = (f32(cell.x + 1) - ro.x) * delta.x;
    } else {
        side.x = (ro.x - f32(cell.x)) * delta.x;
    }
    if rd.y >= 0.0 {
        side.y = (f32(cell.y + 1) - ro.y) * delta.y;
    } else {
        side.y = (ro.y - f32(cell.y)) * delta.y;
    }
    if rd.z >= 0.0 {
        side.z = (f32(cell.z + 1) - ro.z) * delta.z;
    } else {
        side.z = (ro.z - f32(cell.z)) * delta.z;
    }

    var entered = 0.0;
    var step_count = 0u;
    loop {
        if step_count >= MAX_STEPS {
            break;
        }
        let cell_exit = min(side.x, min(side.y, side.z));
        if entered > max_distance {
            break;
        }
        let packed = block_at(cell);
        if packed != AIR {
            let material = packed & 255u;
            let height = f32((packed >> 8u) & 255u) / 16.0;
            let candidate = ray_box(
                ro,
                rd,
                vec3<f32>(f32(cell.x), f32(cell.y), f32(cell.z)),
                vec3<f32>(f32(cell.x) + 1.0, f32(cell.y) + height, f32(cell.z) + 1.0),
            );
            if candidate.found && candidate.t >= entered - 0.002 && candidate.t <= cell_exit + 0.002 && candidate.t < max_distance {
                return Hit(candidate.t, candidate.normal, material, true);
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
            return Hit(candidate.t, candidate.normal, GRASS, true);
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

fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, center: vec3<f32>, radius: f32) -> Hit {
    let offset = ro - center;
    let half_b = dot(offset, rd);
    let discriminant = half_b * half_b - dot(offset, offset) + radius * radius;
    if discriminant < 0.0 {
        return empty_hit(1.0e30);
    }
    let root = sqrt(discriminant);
    var t = -half_b - root;
    if t < 0.0001 {
        t = -half_b + root;
    }
    if t < 0.0001 {
        return empty_hit(1.0e30);
    }
    let point = ro + rd * t;
    return Hit(t, normalize(point - center), LEAVES, true);
}

// This is the shape emitted by BasicBranchBlockBakedModel: a cube from
// 8-radius to 8+radius plus a cuboid sleeve for every connection.  It keeps
// the original mod's thin, angular appearance and discrete 1/16 radii.
fn trace_branch_cell(ro: vec3<f32>, rd: vec3<f32>, branch: BranchCell) -> Hit {
    let centre = branch.centre_radius.xyz;
    let core_radius = branch.centre_radius.w;
    let cell_min = centre - vec3<f32>(0.5);
    let cell_max = centre + vec3<f32>(0.5);

    // ThickBranchBlockBakedModel (radii 9..24) renders a vertical trunk core
    // and its TrunkShellBlock neighbours as one square prism. Ray tracing can
    // intersect the resulting CSG union directly, without materialising the
    // eight proxy shell blocks in the voxel volume.
    if core_radius > 0.5 {
        var thick = ray_box(
            ro,
            rd,
            vec3<f32>(centre.x - core_radius, cell_min.y, centre.z - core_radius),
            vec3<f32>(centre.x + core_radius, cell_max.y, centre.z + core_radius),
        );
        if !thick.found {
            return thick;
        }
        thick.material = WOOD;
        let has_side_branch = branch.connections[2u] + branch.connections[3u]
            + branch.connections[4u] + branch.connections[5u] != 0u;
        let face_direction = select(1u, 0u, thick.normal.y < 0.0);
        if abs(thick.normal.y) > 0.5
            && branch.connections[face_direction] < 1u
            && !has_side_branch
        {
            // The thick renderer selects its rings texture under exactly this
            // condition; all other exposed faces use bark.
            thick.material = WOOD_RINGS;
        }
        return thick;
    }

    var closest = ray_box(
        ro,
        rd,
        centre - vec3<f32>(core_radius),
        centre + vec3<f32>(core_radius),
    );
    var core_was_hit = closest.found;

    var direction = 0u;
    loop {
        if direction >= 6u {
            break;
        }
        let radius = f32(branch.connections[direction]) / 16.0;
        if radius > 0.0 {
            var sleeve_min = centre - vec3<f32>(radius);
            var sleeve_max = centre + vec3<f32>(radius);
            if direction == 0u { // down
                sleeve_min.y = cell_min.y;
                sleeve_max.y = centre.y - radius;
            } else if direction == 1u { // up
                sleeve_min.y = centre.y + radius;
                sleeve_max.y = cell_max.y;
            } else if direction == 2u { // north
                sleeve_min.z = cell_min.z;
                sleeve_max.z = centre.z - radius;
            } else if direction == 3u { // south
                sleeve_min.z = centre.z + radius;
                sleeve_max.z = cell_max.z;
            } else if direction == 4u { // west
                sleeve_min.x = cell_min.x;
                sleeve_max.x = centre.x - radius;
            } else { // east
                sleeve_min.x = centre.x + radius;
                sleeve_max.x = cell_max.x;
            }
            let sleeve = ray_box(ro, rd, sleeve_min, sleeve_max);
            if sleeve.found && sleeve.t < closest.t {
                closest = sleeve;
                core_was_hit = false;
            }
        }
        direction += 1u;
    }
    if !closest.found {
        return closest;
    }

    closest.material = WOOD;
    // BasicBranchBlockBakedModel emits the rings texture only on the side
    // opposite a sole source connection whose radius reaches this core.
    if core_was_hit {
        var number_of_connections = 0u;
        var largest_connection = 0u;
        var source_direction = 0u;
        var side = 0u;
        loop {
            if side >= 6u {
                break;
            }
            let connection = branch.connections[side];
            if connection > 0u {
                number_of_connections += 1u;
            }
            if connection > largest_connection {
                largest_connection = connection;
                source_direction = side;
            }
            side += 1u;
        }
        let core_radius_units = u32(round(core_radius * 16.0));
        if number_of_connections == 1u && largest_connection >= core_radius_units {
            let ring_direction = opposite_direction(source_direction);
            let hit_direction = select(
                select(5u, 4u, closest.normal.x < 0.0),
                select(3u, 2u, closest.normal.z < 0.0),
                abs(closest.normal.z) > 0.5,
            );
            let face_direction = select(
                hit_direction,
                select(1u, 0u, closest.normal.y < 0.0),
                abs(closest.normal.y) > 0.5,
            );
            if face_direction == ring_direction {
                closest.material = WOOD_RINGS;
            }
        }
    }
    return closest;
}

fn trace_trees(ro: vec3<f32>, rd: vec3<f32>, max_distance: f32) -> Hit {
    var closest = empty_hit(max_distance);
    var index = 0u;
    let tree_count = u32(u.simulation.y);
    loop {
        if index >= tree_count {
            break;
        }
        let bounds = trees[index * 3u];
        let ranges = trees[index * 3u + 1u];
        let broad_phase = ray_sphere(ro, rd, bounds.xyz, bounds.w);
        let starts_inside_bounds = length(ro - bounds.xyz) < bounds.w;
        if broad_phase.found && (broad_phase.t < closest.t || starts_inside_bounds) {
            var branch = 0u;
            loop {
                if branch >= u32(ranges.y) {
                    break;
                }
                let branch_cell = branches[u32(ranges.x) + branch];
                let wood = trace_branch_cell(ro, rd, branch_cell);
                if wood.found && wood.t < closest.t {
                    closest = wood;
                }
                branch += 1u;
            }
        }
        index += 1u;
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

fn material_colour(material: u32, point: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
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
    if material == WOOD {
        let bark_lines = 0.84 + 0.16 * sin((point.x + point.z) * 39.0);
        return vec3<f32>(0.31, 0.15, 0.055) * bark_lines * variation;
    }
    if material == WOOD_RINGS {
        let local = fract(point) - vec3<f32>(0.5);
        var ring_plane = vec2<f32>(local.x, local.z);
        if abs(normal.x) > 0.5 {
            ring_plane = vec2<f32>(local.y, local.z);
        } else if abs(normal.z) > 0.5 {
            ring_plane = vec2<f32>(local.x, local.y);
        }
        let rings = 0.72 + 0.28 * sin(length(ring_plane) * 92.0);
        return vec3<f32>(0.42, 0.25, 0.105) * rings * variation;
    }
    if material == DIM_LEAVES {
        return vec3<f32>(0.075, 0.22, 0.055) * variation;
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
    var colour = material_colour(hit.material, point, hit.normal) * light;
    if hit.material == WATER {
        colour += vec3<f32>(0.12, 0.22, 0.25) * pow(max(dot(reflect(direction, hit.normal), u.sun_direction.xyz), 0.0), 30.0);
    }
    let fog = smoothstep(700.0, 4200.0, hit.t);
    colour = mix(colour, sky_colour(direction), fog);
    return vec4<f32>(pow(max(colour, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), 1.0);
}
