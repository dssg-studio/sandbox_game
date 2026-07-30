//! A small ray-traced voxel sandbox.  Geometry is evaluated in the WGSL fragment
//! shader: voxel cells use DDA ray traversal. Dynamic Trees keeps its native
//! growth cells but is rendered through the Eco Machina HPD layout.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet, VecDeque},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex, mpsc},
    thread,
    time::Instant,
};

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec2, Vec3};
use serde::Deserialize;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

// Tracy's convenience macros deliberately panic when no capture client has
// been started. These wrappers keep instrumentation safe in unit tests and in
// a profiling build launched without the viewer, while preserving allocation-
// free static span locations during a real capture.
#[cfg(feature = "profiling")]
macro_rules! profile_span {
    ($name:literal) => {
        tracy_client::Client::running()
            .map(|client| client.span(tracy_client::span_location!($name), 0))
    };
}

#[cfg(not(feature = "profiling"))]
macro_rules! profile_span {
    ($name:literal) => {
        ()
    };
}

#[cfg(feature = "profiling")]
macro_rules! profile_frame_mark {
    () => {
        if let Some(client) = tracy_client::Client::running() {
            client.frame_mark();
        }
    };
}

#[cfg(not(feature = "profiling"))]
macro_rules! profile_frame_mark {
    () => {};
}

#[cfg(feature = "profiling")]
macro_rules! profile_thread_name {
    ($name:literal) => {
        if let Some(client) = tracy_client::Client::running() {
            client.set_thread_name($name);
        }
    };
}

#[cfg(not(feature = "profiling"))]
macro_rules! profile_thread_name {
    ($name:literal) => {};
}

const CHUNK_SIZE: i32 = 16;
const WORLD_HEIGHT: i32 = 48;
const DETAIL_RADIUS: i32 = 4;
/// Distant Horizons-style horizon distance, measured in chunks.
const LOD_RADIUS: i32 = 256;
const DETAIL_DIAMETER: i32 = DETAIL_RADIUS * 2 + 1;
const DETAIL_CHUNK_COUNT: usize = (DETAIL_DIAMETER * DETAIL_DIAMETER) as usize;
const CHUNK_BLOCK_COUNT: usize = (CHUNK_SIZE * CHUNK_SIZE * WORLD_HEIGHT) as usize;
/// Chunks outside the 9×9 ray-traced window are prepared before they are
/// needed.  They never enter the renderer until the complete next stripe is
/// ready, so a stream transition has no hole or synchronous fallback path.
const DETAIL_PREFETCH_RADIUS: i32 = DETAIL_RADIUS + 2;
const MAX_TREES_PER_CHUNK: usize = 2;
const LOD_GRID_SIZE: i32 = 33;
const LOD_LEVEL_FACTORS: [i32; 5] = [1, 2, 4, 8, 16];
const LOD_LEVEL_COUNT: usize = LOD_LEVEL_FACTORS.len();
const LOD_SAMPLE_COUNT: usize = LOD_LEVEL_COUNT * (LOD_GRID_SIZE * LOD_GRID_SIZE) as usize;
/// Dynamic data sources are deliberately smaller than a render grid.  This is
/// the same division of responsibilities as DH's full-data sources: a source
/// owns a compact square of columns and can be generated, cached and merged
/// independently of the visible LOD cut.
const LOD_SECTION_SIDE: i32 = 16;
const LOD_SECTION_COLUMN_COUNT: usize = (LOD_SECTION_SIDE * LOD_SECTION_SIDE) as usize;
const LOD_CACHE_MAGIC: [u8; 4] = *b"RVLH";
const LOD_CACHE_VERSION: u32 = 1;
/// Runtime cache is deliberately relative to the game directory, so a
/// portable copy of the game keeps its distant-world data beside its assets.
const LOD_CACHE_FOLDER: &str = "cache/distant_horizons";
const MAX_TREES: usize = DETAIL_CHUNK_COUNT * MAX_TREES_PER_CHUNK;
/// The Eco Machina layout emits one rectangular spine per wood block plus
/// optional sub-chain and foliage connectors. The limit is deliberately above
/// every bundled JoCode's branch/leaf density, and enforces one renderer-only
/// representation.
const MAX_RENDER_SEGMENTS_PER_TREE: usize = 2048;
const MAX_GPU_TREE_SEGMENTS: usize = MAX_TREES * MAX_RENDER_SEGMENTS_PER_TREE;
/// A BLAS leaf owns at most four exact Eco Machina primitives.  This keeps
/// traversal shallow without allocating a binary node for every segment.
const TREE_BVH_LEAF_SEGMENTS: usize = 4;
/// A forest of small trees has a few more leaves than `segments / 2`, hence
/// the per-tree allowance in addition to the wide-BVH bound.
const MAX_GPU_TREE_BLAS_NODES: usize =
    MAX_GPU_TREE_SEGMENTS / 2 + MAX_TREES * TREE_BVH_LEAF_SEGMENTS;
const MAX_GPU_TREE_TLAS_NODES: usize = MAX_TREES * 2;
const TREE_ATLAS_SEGMENTS_PER_CHUNK: usize = MAX_TREES_PER_CHUNK * MAX_RENDER_SEGMENTS_PER_TREE;
/// A binary BLAS with leaves of four primitives has fewer than half as many
/// nodes as primitives; this rounded per-chunk slot retains room for both
/// Dynamic Trees roots a generated chunk can contain.
const TREE_ATLAS_BLAS_NODES_PER_CHUNK: usize =
    MAX_TREES_PER_CHUNK * (MAX_RENDER_SEGMENTS_PER_TREE / 2);
const DETAIL_STREAM_WORKERS: usize = 4;
/// Host cadence for one call to Dynamic Trees' Species#grow. The source
/// routine itself applies each species' growth rate as a probability.
const TREE_GROWTH_TICK_SECONDS: f32 = 1.0;
/// Dynamic Trees' default `maxBranchRotRadius` server setting. A radius of
/// eight is a full block and therefore survives ordinary unsupported rot.
const MAX_BRANCH_ROT_RADIUS: u8 = 7;
const JO_CODE_ALPHABET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
// Unmodified JoCode registries for the three species used by this prototype.
// Each line is `radius:base64-encoded 3-bit instructions`, loaded from the
// user's Dynamic Trees checkout and embedded so the game remains standalone.
const OAK_JO_CODES: &str = include_str!("../assets/dynamictrees/jo_codes/oak.txt");
const SPRUCE_JO_CODES: &str = include_str!("../assets/dynamictrees/jo_codes/spruce.txt");
const ACACIA_JO_CODES: &str = include_str!("../assets/dynamictrees/jo_codes/acacia.txt");

type Block = u32;

const AIR: Block = 0;
const GRASS: u32 = 1;
const DIRT: u32 = 2;
const STONE: u32 = 3;
const WATER: u32 = 4;
const OAK_LEAVES: u32 = 6;
const SPRUCE_LEAVES: u32 = 7;
const ACACIA_LEAVES: u32 = 8;
const ROOTY_SOIL: u32 = 9;
/// Authored tree materials use a single native 64×64 pixel-art tile. Keeping
/// this size exact avoids a resample between the source PNG and GPU atlas.
const TREE_TEXTURE_TILE_SIZE: u32 = 64;
const TREE_TEXTURE_COLUMNS: u32 = 6;
const TREE_TEXTURE_ROWS: u32 = 3;
const TREE_TEXTURE_CONFIG_PATH: &str = "assets/tree/tree_textures.json";

/// A block packs its material in the low byte and height in 1/16th-block units
/// in the next byte.  A height of 16 is a regular block; all values 1..=15 are
/// real slabs, not a visual approximation.
fn block(material: u32, sixteenths: u32) -> Block {
    material | (sixteenths.clamp(1, 16) << 8)
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ChunkPos {
    x: i32,
    z: i32,
}

impl ChunkPos {
    fn from_world(position: Vec3) -> Self {
        Self {
            x: (position.x.floor() as i32).div_euclid(CHUNK_SIZE),
            z: (position.z.floor() as i32).div_euclid(CHUNK_SIZE),
        }
    }

    fn distance(self, other: Self) -> i32 {
        (self.x - other.x).abs().max((self.z - other.z).abs())
    }
}

#[derive(Clone, Copy)]
enum TreeForm {
    Deciduous,
    Conifer,
    Acacia,
}

impl TreeForm {
    fn from_seed(seed: u32) -> Self {
        match seed % 9 {
            0 | 1 => Self::Conifer,
            2 => Self::Acacia,
            _ => Self::Deciduous,
        }
    }

    fn render_id(self) -> u32 {
        match self {
            Self::Deciduous => 0,
            Self::Conifer => 1,
            Self::Acacia => 2,
        }
    }

    fn leaf_material(self) -> u32 {
        match self {
            Self::Deciduous => OAK_LEAVES,
            Self::Conifer => SPRUCE_LEAVES,
            Self::Acacia => ACACIA_LEAVES,
        }
    }
}

#[derive(Clone)]
struct Chunk {
    // x + 16 * (z + 16 * y)
    // Terrain is immutable after generation. Sharing it makes growth snapshots
    // cheap: only the dynamic tree graphs are copied to a worker.
    blocks: Arc<[Block]>,
    trees: Vec<Tree>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTree {
    // Sphere used for a cheap per-tree broad phase.
    bounds: [f32; 4],
    // Eco Machina HPD segment offset/count, unused, unused.
    layout: [f32; 4],
    // Tree form, soil fertility, age in pulses, unused.
    appearance: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTreeSegment {
    // End-point world positions and half-width in blocks. The two widths are
    // intentionally equal: each Dynamic Trees wood cell owns one uniform,
    // rectangular Eco Machina strip at its direct 1/16-based DT radius.
    start_radius: [f32; 4],
    end_radius: [f32; 4],
    // [species form, kind, foliage texture variant, unused].
    style: [u32; 4],
}

/// One node is shared by the forest TLAS and each tree's segment BLAS. Leaves
/// use `data = [first, count, 1, 0]`; inner nodes use child indices in the
/// first two entries and a zero leaf flag. Bounds conservatively include the
/// camera-facing foliage rectangles as well as rectangular wood prisms.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTreeBvhNode {
    minimum: [f32; 4],
    maximum: [f32; 4],
    data: [u32; 4],
}

/// Renderer-ready data for one generated chunk. Segment and BLAS offsets are
/// local to the chunk and are relocated only when it occupies a physical ring
/// slot, so worker threads never touch the live GPU atlas.
#[derive(Clone)]
struct ChunkTreeGeometry {
    trees: [GpuTree; MAX_TREES_PER_CHUNK],
    tree_count: usize,
    segments: Vec<GpuTreeSegment>,
    blas_nodes: Vec<GpuTreeBvhNode>,
}

#[derive(Clone)]
struct ChunkBuild {
    position: ChunkPos,
    ticket: u64,
    chunk: Chunk,
    tree_geometry: ChunkTreeGeometry,
}

struct ActiveChunk {
    chunk: Chunk,
    tree_geometry: ChunkTreeGeometry,
    growth_revision: u64,
}

#[derive(Clone)]
struct GrowthSnapshotChunk {
    position: ChunkPos,
    revision: u64,
    chunk: Chunk,
}

struct GrowthRequest {
    ticks: u32,
    chunks: Vec<GrowthSnapshotChunk>,
}

struct GrowthChunkUpdate {
    position: ChunkPos,
    revision: u64,
    chunk: Chunk,
    tree_geometry: ChunkTreeGeometry,
}

struct GrowthResult {
    updates: Vec<GrowthChunkUpdate>,
}

struct ChunkBuildRequest {
    position: ChunkPos,
    ticket: u64,
    priority: i32,
}

impl Ord for ChunkBuildRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max heap. Lower distance / higher travel bias is
        // therefore ordered first, with stable coordinates as tie breakers.
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.position.z.cmp(&self.position.z))
            .then_with(|| other.position.x.cmp(&self.position.x))
            .then_with(|| other.ticket.cmp(&self.ticket))
    }
}

impl PartialOrd for ChunkBuildRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ChunkBuildRequest {
    fn eq(&self, other: &Self) -> bool {
        self.ticket == other.ticket
    }
}

impl Eq for ChunkBuildRequest {}

struct ChunkBuildQueue {
    pending: Mutex<BinaryHeap<ChunkBuildRequest>>,
    ready: Condvar,
}

impl ChunkBuildQueue {
    fn new() -> Self {
        Self {
            pending: Mutex::new(BinaryHeap::new()),
            ready: Condvar::new(),
        }
    }

    fn push(&self, request: ChunkBuildRequest) {
        let mut pending = self.pending.lock().expect("chunk build queue poisoned");
        pending.push(request);
        self.ready.notify_one();
    }

    fn pop(&self) -> ChunkBuildRequest {
        let mut pending = self.pending.lock().expect("chunk build queue poisoned");
        loop {
            if let Some(request) = pending.pop() {
                return request;
            }
            pending = self
                .ready
                .wait(pending)
                .expect("chunk build queue poisoned");
        }
    }
}

#[derive(Default)]
struct WorldChanges {
    detail_slots: Vec<usize>,
    tree_slots: Vec<usize>,
    tree_tlas_changed: bool,
    lod_layout_changed: bool,
}

impl WorldChanges {
    fn has_tree_updates(&self) -> bool {
        !self.tree_slots.is_empty()
    }

    fn merge(&mut self, mut other: Self) {
        self.detail_slots.append(&mut other.detail_slots);
        self.tree_slots.append(&mut other.tree_slots);
        self.tree_tlas_changed |= other.tree_tlas_changed;
        self.lod_layout_changed |= other.lod_layout_changed;
        self.detail_slots.sort_unstable();
        self.detail_slots.dedup();
        self.tree_slots.sort_unstable();
        self.tree_slots.dedup();
    }
}

struct TreeSegmentPrimitive {
    segment: GpuTreeSegment,
    minimum: Vec3,
    maximum: Vec3,
    centroid: Vec3,
}

struct TreeTlasPrimitive {
    tree_index: usize,
    minimum: Vec3,
    maximum: Vec3,
    centroid: Vec3,
}

fn bvh_node(minimum: Vec3, maximum: Vec3, data: [u32; 4]) -> GpuTreeBvhNode {
    GpuTreeBvhNode {
        minimum: minimum.extend(0.0).to_array(),
        maximum: maximum.extend(0.0).to_array(),
        data,
    }
}

fn bvh_bounds<'a>(
    items: impl IntoIterator<Item = (&'a Vec3, &'a Vec3, &'a Vec3)>,
) -> (Vec3, Vec3, Vec3, Vec3) {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    let mut centroid_minimum = Vec3::splat(f32::INFINITY);
    let mut centroid_maximum = Vec3::splat(f32::NEG_INFINITY);
    for (item_minimum, item_maximum, centroid) in items {
        minimum = minimum.min(*item_minimum);
        maximum = maximum.max(*item_maximum);
        centroid_minimum = centroid_minimum.min(*centroid);
        centroid_maximum = centroid_maximum.max(*centroid);
    }
    (minimum, maximum, centroid_minimum, centroid_maximum)
}

fn largest_axis(extent: Vec3) -> usize {
    if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    }
}

fn segment_primitive(segment: GpuTreeSegment) -> TreeSegmentPrimitive {
    let start = Vec3::from_array([
        segment.start_radius[0],
        segment.start_radius[1],
        segment.start_radius[2],
    ]);
    let end = Vec3::from_array([
        segment.end_radius[0],
        segment.end_radius[1],
        segment.end_radius[2],
    ]);
    let wood_half_width = segment.start_radius[3].max(segment.end_radius[3]);
    // A foliage connector is a camera-facing rectangle whose half-width is
    // half its length. Wood instead remains a square prism at the direct DT
    // radius. This bound is conservative in every camera orientation.
    let half_width = if segment.style[1] == SEGMENT_FOLIAGE_CONNECTOR {
        wood_half_width.max((end - start).length() * 0.5)
    } else {
        wood_half_width
    };
    let extent = Vec3::splat(half_width);
    let minimum = start.min(end) - extent;
    let maximum = start.max(end) + extent;
    TreeSegmentPrimitive {
        segment,
        minimum,
        maximum,
        centroid: (minimum + maximum) * 0.5,
    }
}

fn build_tree_blas(
    items: &mut [TreeSegmentPrimitive],
    segments: &mut [GpuTreeSegment],
    segment_cursor: &mut usize,
    nodes: &mut [GpuTreeBvhNode],
    node_cursor: &mut usize,
) -> usize {
    assert!(
        !items.is_empty(),
        "a BLAS cannot be built from zero segments"
    );
    let node_index = *node_cursor;
    *node_cursor += 1;
    assert!(
        *node_cursor <= nodes.len(),
        "Eco Machina BLAS node budget exhausted"
    );
    let (minimum, maximum, centroid_minimum, centroid_maximum) = bvh_bounds(
        items
            .iter()
            .map(|item| (&item.minimum, &item.maximum, &item.centroid)),
    );
    if items.len() <= TREE_BVH_LEAF_SEGMENTS {
        let item_count = items.len();
        let first_segment = *segment_cursor;
        assert!(
            first_segment + item_count <= segments.len(),
            "Eco Machina segment budget exhausted while building BLAS"
        );
        for item in items {
            segments[*segment_cursor] = item.segment;
            *segment_cursor += 1;
        }
        nodes[node_index] = bvh_node(
            minimum,
            maximum,
            [first_segment as u32, item_count as u32, 1, 0],
        );
        return node_index;
    }
    let axis = largest_axis(centroid_maximum - centroid_minimum);
    items.sort_by(|left, right| left.centroid[axis].total_cmp(&right.centroid[axis]));
    let split = items.len() / 2;
    let (left, right) = items.split_at_mut(split);
    let left_child = build_tree_blas(left, segments, segment_cursor, nodes, node_cursor);
    let right_child = build_tree_blas(right, segments, segment_cursor, nodes, node_cursor);
    nodes[node_index] = bvh_node(
        minimum,
        maximum,
        [left_child as u32, right_child as u32, 0, 0],
    );
    node_index
}

fn build_tree_tlas(
    items: &mut [TreeTlasPrimitive],
    nodes: &mut [GpuTreeBvhNode],
    node_cursor: &mut usize,
) -> usize {
    assert!(!items.is_empty(), "a TLAS cannot be built from zero trees");
    let node_index = *node_cursor;
    *node_cursor += 1;
    assert!(
        *node_cursor <= nodes.len(),
        "tree TLAS node budget exhausted"
    );
    let (minimum, maximum, centroid_minimum, centroid_maximum) = bvh_bounds(
        items
            .iter()
            .map(|item| (&item.minimum, &item.maximum, &item.centroid)),
    );
    if items.len() == 1 {
        nodes[node_index] = bvh_node(minimum, maximum, [items[0].tree_index as u32, 1, 1, 0]);
        return node_index;
    }
    let axis = largest_axis(centroid_maximum - centroid_minimum);
    items.sort_by(|left, right| left.centroid[axis].total_cmp(&right.centroid[axis]));
    let split = items.len() / 2;
    let (left, right) = items.split_at_mut(split);
    let left_child = build_tree_tlas(left, nodes, node_cursor);
    let right_child = build_tree_tlas(right, nodes, node_cursor);
    nodes[node_index] = bvh_node(
        minimum,
        maximum,
        [left_child as u32, right_child as u32, 0, 0],
    );
    node_index
}

// Dynamic Trees stores a tree as a sparse collection of branch and leaf
// blocks. The renderer reads that topology only to build an HPD hierarchy and
// traces constant-width Eco Machina rectangular segments directly.
const DIRECTIONS: [IVec3; 6] = [
    IVec3::NEG_Y,
    IVec3::Y,
    IVec3::NEG_Z,
    IVec3::Z,
    IVec3::NEG_X,
    IVec3::X,
];
const DOWN: usize = 0;
const UP: usize = 1;

const SEGMENT_WOOD_SPINE: u32 = 0;
const SEGMENT_WOOD_CONNECTOR: u32 = 1;
const SEGMENT_FOLIAGE_CONNECTOR: u32 = 2;

/// Direct 3D counterpart of the visualizer's per-wood-block HPD metadata.
/// Dynamic Trees remains the living source graph; this structure only turns
/// it into Eco Machina spine meshes and connectors.
struct HpdNode {
    position: IVec3,
    parent: Option<usize>,
    children: Vec<usize>,
    subtree_weight: u32,
    // The Dynamic Trees branch value is the authoritative render half-width:
    // radius 1..=8 maps directly to 1/16..=8/16 of a block.
    dt_radius: u8,
    chain_id: u16,
    position_in_chain: u16,
    chain_depth: u16,
    heavy_child: Option<usize>,
}

fn opposite(direction: usize) -> usize {
    [UP, DOWN, 3, 2, 5, 4][direction]
}

#[derive(Clone, Copy)]
struct GrowSignal {
    energy: f32,
    direction: usize,
    default_direction: usize,
    num_turns: u8,
    num_steps: u8,
    root: IVec3,
    delta: IVec3,
    radius: f32,
    success: bool,
    choked: bool,
}

impl GrowSignal {
    fn new(root: IVec3, energy: f32) -> Self {
        Self {
            energy,
            direction: UP,
            default_direction: UP,
            num_turns: 0,
            num_steps: 0,
            root,
            delta: IVec3::ZERO,
            radius: 0.0,
            success: true,
            choked: false,
        }
    }

    fn step(&mut self) -> bool {
        self.num_steps = self.num_steps.saturating_add(1);
        self.delta += DIRECTIONS[self.direction];
        self.energy -= 1.0;
        if self.energy <= 0.0 {
            self.success = false;
        }
        self.success
    }

    fn turn(&mut self, target: usize) {
        if self.direction != target {
            self.direction = target;
            self.num_turns = self.num_turns.saturating_add(1);
        }
    }

    fn is_in_trunk(self) -> bool {
        self.num_turns == 0
    }
}

#[derive(Clone)]
struct Tree {
    // Position of rooty soil. Branches start at root + UP, just like DT.
    root: IVec3,
    form: TreeForm,
    fertility: u8,
    random_state: u32,
    // Radius 1..=8 for each Dynamic Trees branch block.
    branches: HashMap<IVec3, u8>,
    // Hydration 1..=7 for dynamic leaf blocks; missing cells are air.
    leaves: HashMap<IVec3, u8>,
}

/// Read-only world view for one growth tick. Dynamic Trees asks Minecraft for
/// a TreePart and block state at every candidate location; this is the small
/// equivalent needed by the standalone voxel world.
#[derive(Default)]
struct GrowthEnvironment {
    terrain: HashSet<IVec3>,
    tree_part_owner: HashMap<IVec3, IVec3>,
    tree_parts: HashMap<IVec3, EnvironmentTreePart>,
}

#[derive(Clone, Copy)]
enum EnvironmentTreePart {
    Branch(u8),
    Leaf,
}

impl GrowthEnvironment {
    fn is_clear_for(&self, position: IVec3, root: IVec3) -> bool {
        !self.terrain.contains(&position) && !self.has_foreign_tree_part(position, root)
    }

    fn has_foreign_tree_part(&self, position: IVec3, root: IVec3) -> bool {
        self.tree_part_owner
            .get(&position)
            .is_some_and(|owner| *owner != root)
    }

    fn can_place_leaf(&self, position: IVec3, root: IVec3) -> bool {
        self.is_clear_for(position, root) && !self.terrain.contains(&(position + IVec3::NEG_Y))
    }

    fn has_tree_part(&self, position: IVec3) -> bool {
        self.tree_part_owner.contains_key(&position)
    }

    fn has_clear_sky(&self, position: IVec3) -> bool {
        ((position.y + 1)..WORLD_HEIGHT).all(|y| {
            let above = IVec3::new(position.x, y, position.z);
            !self.terrain.contains(&above) && !self.has_tree_part(above)
        })
    }
}

impl Tree {
    fn new(root: IVec3, seed: u32) -> Self {
        let mut tree = Self {
            root,
            form: TreeForm::from_seed(seed),
            fertility: 15,
            random_state: seed ^ 0x6d2b_79f5,
            branches: HashMap::new(),
            leaves: HashMap::new(),
        };
        // Dynamic Trees worldgen selects a JoCode for a requested radius, then
        // emits its branch instructions, inflates the network and ages the
        // leaf volume. It does not replay ordinary random growth.
        assert!(
            tree.generate_jocode_worldgen(seed),
            "the bundled Dynamic Trees JoCode registry must generate a rooted tree"
        );
        tree
    }

    fn jocode_registry(&self) -> &'static str {
        match self.form {
            TreeForm::Deciduous => OAK_JO_CODES,
            TreeForm::Conifer => SPRUCE_JO_CODES,
            TreeForm::Acacia => ACACIA_JO_CODES,
        }
    }

    fn generate_jocode_worldgen(&mut self, seed: u32) -> bool {
        // DynamicTreeFeature supplies an integer generation radius. The
        // standalone terrain has no biome Poisson circle, so derive the same
        // 2..=8 JoCode bucket deterministically from its placement seed.
        let requested_radius = 2 + (hash_u32(seed ^ 0x71c3_4e2d) % 7) as u8;
        let codes: Vec<&str> = self
            .jocode_registry()
            .lines()
            .filter_map(|line| line.split_once(':'))
            .filter_map(|(radius, code)| {
                (radius.parse::<u8>().ok() == Some(requested_radius)).then_some(code)
            })
            .collect();
        let Some(code) = codes
            .get((hash_u32(seed ^ 0xb529_7a4d) as usize) % codes.len().max(1))
            .copied()
        else {
            return false;
        };

        self.branches.clear();
        self.leaves.clear();
        let instructions = Self::decode_jocode(code);
        self.emit_jocode_fork(&instructions, 0, self.root, false);
        if !self.branches.contains_key(&(self.root + IVec3::Y)) {
            self.branches.clear();
            return false;
        }
        self.inflate_jocode_network();
        true
    }

    fn decode_jocode(code: &str) -> Vec<u8> {
        let mut instructions = Vec::with_capacity(code.len() * 2);
        for character in code.bytes() {
            if let Some(value) = JO_CODE_ALPHABET.bytes().position(|c| c == character) {
                instructions.push((value >> 3) as u8);
                instructions.push((value & 7) as u8);
            }
        }
        instructions
    }

    /// Direct counterpart of JoCode#generateFork. The source uses six
    /// directions plus 6=fork and 7=return, and disables only the current
    /// failed fork when it hits an occupied location.
    fn emit_jocode_fork(
        &mut self,
        instructions: &[u8],
        mut code_position: usize,
        mut position: IVec3,
        mut disabled: bool,
    ) -> usize {
        while code_position < instructions.len() {
            match instructions[code_position] {
                6 => {
                    code_position =
                        self.emit_jocode_fork(instructions, code_position + 1, position, disabled);
                }
                7 => return code_position + 1,
                direction => {
                    position += DIRECTIONS[direction as usize];
                    if !disabled {
                        if self.branches.contains_key(&position) || position == self.root {
                            disabled = true;
                        } else {
                            self.branches.insert(position, 1);
                        }
                    }
                    code_position += 1;
                }
            }
        }
        code_position
    }

    /// Counterpart of the JoCode post-pass made by InflatorNode: leaf clusters
    /// are blitted at every twig, while parent radius is the square root of
    /// child areas plus the 1.5× worldgen tapering factor.
    fn inflate_jocode_network(&mut self) {
        let start = self.root + IVec3::Y;
        let mut visited = HashSet::new();
        let mut twigs = Vec::new();
        self.inflate_jocode_branch(start, self.root, &mut visited, &mut twigs);
        let environment = GrowthEnvironment::default();
        for twig in twigs {
            self.stamp_leaf_cluster_in(twig, &environment);
        }
        // Species#getWorldGenAgeIterations returns three. This is the local
        // CellKit equivalent of TreeHelper#ageVolume during JoCode generation.
        for _ in 0..3 {
            self.update_leaves_once();
        }
    }

    fn inflate_jocode_branch(
        &mut self,
        position: IVec3,
        parent: IVec3,
        visited: &mut HashSet<IVec3>,
        twigs: &mut Vec<IVec3>,
    ) -> f32 {
        if !visited.insert(position) {
            return f32::from(*self.branches.get(&position).unwrap_or(&1));
        }
        let children: Vec<IVec3> = DIRECTIONS
            .iter()
            .map(|offset| position + *offset)
            .filter(|candidate| *candidate != parent && self.branches.contains_key(candidate))
            .collect();
        if children.is_empty() {
            twigs.push(position);
            self.branches.insert(position, 1);
            return 1.0;
        }

        let mut area = 0.0_f32;
        for child in children {
            if !visited.contains(&child) {
                area += self
                    .inflate_jocode_branch(child, position, visited, twigs)
                    .powi(2);
            }
        }
        // A connected branch with no unvisited child belongs to a merged
        // network; keep its existing twig-sized contribution rather than
        // allowing a zero-radius cell.
        if area == 0.0 {
            area = 1.0;
        }
        let radius = (area.sqrt() + self.tapering() * 1.5).clamp(2.0, 8.0);
        self.branches.insert(position, radius.floor() as u8);
        radius
    }

    fn random(&mut self) -> f32 {
        self.random_state = hash_u32(self.random_state.wrapping_add(0x9e37_79b9));
        (self.random_state & 0x00ff_ffff) as f32 / 16_777_215.0
    }

    fn signal_energy(&self) -> f32 {
        match self.form {
            // trees/dynamictrees/species/oak.json
            TreeForm::Deciduous => 12.0,
            // spruce.json selects ConiferLogic, whose default height
            // variation is [0, 4] in addition to the species energy.
            TreeForm::Conifer => 16.0 + (hash_2d(self.root.x, self.root.z, 2) % 5) as f32,
            // acacia.json
            TreeForm::Acacia => 12.0,
        }
    }

    fn tapering(&self) -> f32 {
        match self.form {
            TreeForm::Deciduous => 0.30,
            TreeForm::Conifer => 0.25,
            TreeForm::Acacia => 0.15,
        }
    }

    fn up_probability(&self) -> i32 {
        match self.form {
            TreeForm::Deciduous => 2,
            TreeForm::Conifer => 3,
            TreeForm::Acacia => 0,
        }
    }

    fn lowest_branch_height(&self) -> u8 {
        match self.form {
            TreeForm::Deciduous => 3,
            TreeForm::Conifer => 3,
            TreeForm::Acacia => 3,
        }
    }

    fn growth_rate(&self) -> f32 {
        match self.form {
            TreeForm::Deciduous => 0.8,
            TreeForm::Conifer => 0.9,
            TreeForm::Acacia => 0.7,
        }
    }

    fn leaf_smother_limit(&self) -> u8 {
        match self.form {
            // oak.json uses LeavesProperties' default of zero: it relies on
            // sky light alone instead of the special vertical canopy rule.
            TreeForm::Deciduous => 0,
            // leaves_properties/{spruce,acacia}.json
            TreeForm::Conifer => 3,
            TreeForm::Acacia => 2,
        }
    }

    fn is_tree_part_at(&self, position: IVec3, environment: &GrowthEnvironment) -> bool {
        self.branches.contains_key(&position)
            || self.leaves.contains_key(&position)
            || environment.has_tree_part(position)
    }

    /// `DynamicLeavesBlock#isBottom`: a leaf stack above another leaf or a
    /// twig is not its own bottom, while one sitting over a stocky branch (or
    /// a non-tree block) is. The environment carries foreign tree parts so a
    /// neighbouring tree can also shade this one.
    fn is_leaf_stack_bottom(&self, position: IVec3, environment: &GrowthEnvironment) -> bool {
        let below = position + IVec3::NEG_Y;
        if let Some(radius) = self.branches.get(&below) {
            return *radius > 1;
        }
        if self.leaves.contains_key(&below) {
            return false;
        }
        match environment.tree_parts.get(&below) {
            Some(EnvironmentTreePart::Branch(radius)) => *radius > 1,
            Some(EnvironmentTreePart::Leaf) => false,
            // This covers owner-only entries used by lightweight tests and
            // keeps them conservative tree-part blockers.
            None => !environment.has_tree_part(below),
        }
    }

    /// Standalone counterpart of `DynamicLeavesBlock#hasAdequateLight`. Our
    /// voxel world has either an unobstructed skylight column or no skylight;
    /// the source smother values and bottom-of-stack rule are preserved.
    fn has_adequate_leaf_light(&self, position: IVec3, environment: &GrowthEnvironment) -> bool {
        if environment.has_clear_sky(position) {
            return true;
        }
        let smother = self.leaf_smother_limit();
        if smother != 0 && self.is_leaf_stack_bottom(position, environment) {
            let smothered = (1..=smother)
                .filter(|offset| {
                    self.is_tree_part_at(position + IVec3::Y * i32::from(*offset), environment)
                })
                .count();
            if smothered >= usize::from(smother) {
                return false;
            }
        }
        // Dynamic Trees defaults to a sky-light requirement of 14 for a new
        // leaf. With no propagated skylight implementation yet, an occluded
        // column has level zero and cannot create a new leaf.
        false
    }

    /// Counterpart of `Species#update`: analyze tips, prune unsupported
    /// branches, then send normal growth signals only if the network remains.
    fn update_in(&mut self, environment: &GrowthEnvironment) -> bool {
        let mut changed = self.handle_rot_in(environment);
        if !self.branches.is_empty() {
            changed |= self.grow_in(environment);
        }
        changed
    }

    fn branch_endpoints(&self) -> Vec<IVec3> {
        let trunk_base = self.root + IVec3::Y;
        self.branches
            .keys()
            .copied()
            .filter(|position| {
                let neighbours = DIRECTIONS
                    .iter()
                    .filter(|offset| self.branches.contains_key(&(*position + **offset)))
                    .count();
                // FindEndsNode walks away from rooty soil, so a connected
                // trunk base is not an endpoint simply because it has one
                // branch neighbour. An isolated base is still an endpoint.
                neighbours == 0 || (*position != trunk_base && neighbours == 1)
            })
            .collect()
    }

    fn is_branch_reinforced(&self, position: IVec3, radius: u8) -> bool {
        let mut branch_support = 0_u8;
        let mut leaf_support = 0_u8;
        for offset in DIRECTIONS {
            let neighbour = position + offset;
            if self.branches.contains_key(&neighbour) || neighbour == self.root {
                branch_support = branch_support.saturating_add(1);
                leaf_support = leaf_support.saturating_add(1);
            } else if radius == 1 && self.leaves.contains_key(&neighbour) {
                // DynamicLeavesBlock supports only primary-thickness twigs.
                leaf_support = leaf_support.saturating_add(1);
            }
            if branch_support >= 1 && leaf_support >= 2 {
                return true;
            }
        }
        false
    }

    /// Port of `BasicBranchBlock#checkForRot` plus `Species#rot` for the
    /// branch and leaf data this standalone renderer owns. Natural rot checks
    /// only endpoints; a bare twig with fertile soil tries UP/N/S/E/W leaves
    /// before it can disappear.
    fn handle_rot_in(&mut self, environment: &GrowthEnvironment) -> bool {
        let mut changed = false;
        for endpoint in self.branch_endpoints() {
            let Some(radius) = self.branches.get(&endpoint).copied() else {
                continue;
            };
            let chance = 0.3 + 1.0 / f32::from(radius);
            if self.random() > chance || self.is_branch_reinforced(endpoint, radius) {
                continue;
            }
            if radius == 1 && self.fertility > 0 {
                let grew_leaf = [UP, 2, 3, 5, 4].into_iter().any(|direction| {
                    self.try_place_leaf_in(endpoint + DIRECTIONS[direction], Some(0), environment)
                        != 0
                });
                if grew_leaf {
                    changed = true;
                    continue;
                }
            }
            if radius <= MAX_BRANCH_ROT_RADIUS {
                self.branches.remove(&endpoint);
                changed = true;
            }
        }
        if changed {
            // Removing a branch schedules neighbouring dynamic leaves in the
            // mod. Resolve the same local CellKit response synchronously.
            self.update_leaves_in(environment);
        }
        changed
    }

    /// One host call to Species#grow. Its source do/while loop is preserved:
    /// a fractional rate gives one Bernoulli attempt, and a rate above one can
    /// make additional attempts.
    fn grow_in(&mut self, environment: &GrowthEnvironment) -> bool {
        if self.fertility == 0 {
            return false;
        }
        let mut changed = false;
        let mut rate = self.growth_rate();
        loop {
            if self.fertility > 0 && rate > self.random() {
                changed |= self.grow_pulse_in(environment);
            }
            rate -= 1.0;
            if rate <= 0.0 {
                break;
            }
        }
        changed
    }

    #[cfg(test)]
    fn grow_pulse(&mut self) -> bool {
        self.grow_pulse_in(&GrowthEnvironment::default())
    }

    fn grow_pulse_in(&mut self, environment: &GrowthEnvironment) -> bool {
        if self.fertility == 0 {
            return false;
        }
        let start = self.root + IVec3::Y;
        if self.branches.is_empty() {
            self.branches.insert(start, 1);
            self.stamp_leaf_cluster_in(start, environment);
            self.update_leaves_in(environment);
            return true;
        }
        let mut signal = GrowSignal::new(self.root, self.signal_energy());
        self.grow_branch(signal.root + IVec3::Y, &mut signal, 0, environment);
        // Species.grow draws nutrients with a species-specific longevity. The
        // values below correspond to one depletion in roughly 8 successes and
        // make failed signals sixteen times less expensive, as in the source.
        let longevity = if signal.success { 8 } else { 128 };
        if self.random() < 1.0 / longevity as f32 {
            self.fertility = self.fertility.saturating_sub(1);
        }
        signal.success || signal.choked
    }

    fn grow_branch(
        &mut self,
        position: IVec3,
        signal: &mut GrowSignal,
        depth: u8,
        environment: &GrowthEnvironment,
    ) {
        if depth >= 31 || !signal.step() {
            return;
        }
        let Some(current_radius) = self.branches.get(&position).copied() else {
            signal.success = false;
            return;
        };
        // BasicBranchBlock takes originDir before `doTurn`; the return path
        // must exclude that incoming branch rather than the new direction.
        let origin_direction = opposite(signal.direction);
        let target_direction = self.select_direction(position, current_radius, signal, environment);
        signal.turn(target_direction);
        let next = position + DIRECTIONS[signal.direction];
        if self.branches.contains_key(&next) {
            self.grow_branch(next, signal, depth + 1, environment);
        } else if self.leaves.contains_key(&next) {
            // DynamicLeavesBlock is itself a TreePart. Its growSignal starts
            // with a second GrowSignal#step before it attempts branchOut.
            if signal.step() {
                self.branch_out(next, signal, environment);
            }
        } else if environment.is_clear_for(next, self.root) {
            self.grow_into_air(next, current_radius, signal, environment);
        }

        // This is the GrowSignal return path from BasicBranchBlock: conserved
        // cross-sectional area of child branches plus species tapering.
        let mut area = signal.radius * signal.radius;
        for (direction, offset) in DIRECTIONS.iter().enumerate() {
            if direction != origin_direction
                && direction != signal.direction
                && let Some(radius) = self.branches.get(&(position + *offset))
            {
                area += f32::from(*radius).powi(2);
            }
        }
        if !signal.choked {
            let target_radius =
                (area.sqrt() + self.tapering()).clamp(f32::from(current_radius), 8.0);
            let radius = target_radius.floor() as u8;
            self.branches.insert(position, radius);
            signal.radius = target_radius;
        }
    }

    fn select_direction(
        &mut self,
        position: IVec3,
        radius: u8,
        signal: &mut GrowSignal,
        environment: &GrowthEnvironment,
    ) -> usize {
        if signal.num_steps.saturating_add(1) <= self.lowest_branch_height() {
            return signal.default_direction;
        }
        let origin = opposite(signal.direction);
        let mut probability = [0_i32; 6];
        if signal.direction != opposite(signal.default_direction) {
            probability[signal.default_direction] = self.up_probability();
        }
        probability[signal.direction] += 1; // getProbabilityForCurrentDir()
        for (direction, offset) in DIRECTIONS.iter().enumerate() {
            if direction == origin {
                continue;
            }
            let candidate = position + *offset;
            probability[direction] += if let Some(other_radius) = self.branches.get(&candidate) {
                i32::from(*other_radius) + 2
            } else if self.leaves.contains_key(&candidate) {
                2
            } else if environment.is_clear_for(candidate, self.root) {
                1 // NullTreePart returns 1 for air.
            } else {
                0
            };
        }

        // The source's ConiferLogic replaces the generic distribution: no down
        // growth, a vertical trunk, and controlled side turns/whorls.
        if matches!(self.form, TreeForm::Conifer) {
            probability[DOWN] = 0;
            probability[UP] = if signal.is_in_trunk() {
                self.up_probability()
            } else {
                0
            };
            let side_turn = if !signal.is_in_trunk()
                || (signal.is_in_trunk() && signal.num_steps % 2 == 1 && radius > 1)
            {
                2
            } else {
                0
            };
            probability[2] = side_turn;
            probability[3] = side_turn;
            probability[4] = side_turn;
            probability[5] = side_turn;
            probability[origin] = 0;
            probability[signal.direction] += if signal.is_in_trunk() {
                0
            } else if signal.num_turns == 1 {
                2
            } else {
                1
            };
        }
        let sum: i32 = probability.iter().sum();
        if sum <= 0 {
            return UP; // GrowthLogicKit falls back to Direction.UP.
        }
        let mut choice = (self.random() * sum as f32) as i32;
        for (direction, weight) in probability.into_iter().enumerate() {
            choice -= weight;
            if choice < 0 {
                // ConiferLogic changes the energy before BasicBranchBlock
                // applies GrowSignal#doTurn, while the signal is still trunk.
                if matches!(self.form, TreeForm::Conifer) && signal.is_in_trunk() && direction != UP
                {
                    signal.energy = (signal.energy / 3.0).min(16.0);
                }
                return direction;
            }
        }
        UP
    }

    fn grow_into_air(
        &mut self,
        position: IVec3,
        from_radius: u8,
        signal: &mut GrowSignal,
        environment: &GrowthEnvironment,
    ) {
        if from_radius == 1 {
            // BasicBranchBlock#growIntoAir sends zero to
            // growLeavesIfLocationIsSuitable, which substitutes the CellKit
            // default hydration (four for these three species).
            signal.success = self.try_place_leaf_in(position, Some(4), environment) != 0;
        } else {
            // BasicBranchBlock delegates a non-twig growth attempt to
            // DynamicLeavesBlock#branchOut. It may only become a branch after
            // proving there is (or can be) a viable leaf canopy around it.
            self.branch_out(position, signal, environment);
        }
    }

    fn branch_out(
        &mut self,
        position: IVec3,
        signal: &mut GrowSignal,
        environment: &GrowthEnvironment,
    ) {
        let origin = opposite(signal.direction);
        // DynamicLeavesBlock#branchOut first guarantees that its current
        // position could support leaves. Existing compatible leaves satisfy
        // this test without being replaced.
        if !self.need_leaf_in(position, environment) {
            signal.success = false;
            return;
        }
        if (0..6).any(|direction| {
            direction != origin
                && self
                    .branches
                    .contains_key(&(position + DIRECTIONS[direction]))
        }) {
            signal.success = false;
            return;
        }
        // Test the canopy before replacing this leaf with a branch. `needLeaf`
        // uses the default CellKit hydration for fresh neighbours, exactly as
        // DynamicLeavesBlock#needLeaves does.
        let has_leaves = DIRECTIONS
            .iter()
            .any(|offset| self.need_leaf_in(position + *offset, environment));
        self.update_all_leaves_in(position, environment);
        if !has_leaves {
            signal.success = false;
            return;
        }
        self.leaves.remove(&position);
        self.branches.insert(position, 1);
        signal.radius = 2.0; // Family.secondaryThickness
        signal.success = true;
    }

    fn stamp_leaf_cluster_in(&mut self, centre: IVec3, environment: &GrowthEnvironment) {
        let (shape, size_x, size_y, anchor) = match self.form {
            TreeForm::Conifer => (&CONIFER_LEAF_CLUSTER[..], 5, 2, IVec3::new(2, 0, 2)),
            TreeForm::Acacia => (&ACACIA_LEAF_CLUSTER[..], 7, 2, IVec3::new(3, 0, 3)),
            TreeForm::Deciduous => (&DECIDUOUS_LEAF_CLUSTER[..], 5, 4, IVec3::new(2, 1, 2)),
        };
        let base = centre - anchor;
        for y in 0..size_y {
            for z in 0..size_x {
                for x in 0..size_x {
                    let hydration = shape[(x + size_x * (z + size_x * y)) as usize];
                    if hydration > 0 {
                        let position = base + IVec3::new(x, y, z);
                        if self.is_leaf_location_suitable_in(position, environment) {
                            self.leaves
                                .entry(position)
                                .and_modify(|current| *current = (*current).max(hydration))
                                .or_insert(hydration);
                        }
                    }
                }
            }
        }
    }

    fn update_leaves_in(&mut self, _environment: &GrowthEnvironment) {
        // DynamicLeavesBlock asks each adjacent TreePart for a Cell, counts
        // the values it receives through that side, then runs CellKit's
        // BasicSolver. Two synchronous passes settle the local update before
        // the next 150 ms simulation pulse.
        for _ in 0..2 {
            self.update_leaves_once();
        }
    }

    fn update_leaves_once(&mut self) {
        let previous = self.leaves.clone();
        let mut next = HashMap::with_capacity(previous.len());
        for position in previous.keys() {
            let hydration = self.hydration_from_neighbours(*position, &previous);
            if hydration > 0 {
                next.insert(*position, hydration);
            }
        }
        self.leaves = next;
    }

    /// Direct port of DynamicLeavesBlock#updateAllLeaves for the parts that
    /// do not depend on Minecraft's block registry. It deliberately updates a
    /// bounded breadth-first canopy, rather than re-growing a prefab crown.
    fn update_all_leaves_in(&mut self, start: IVec3, environment: &GrowthEnvironment) -> bool {
        let first_hydration = self.update_leaf_hydration_in(start);
        if first_hydration == 0 {
            return false;
        }
        let mut queue = VecDeque::from([(start, first_hydration)]);
        let mut processed = HashSet::new();
        while let Some((position, hydration)) = queue.pop_front() {
            if processed.len() > 256 {
                break; // LeavesProperties#maxLeavesRecursion
            }
            processed.insert(position);
            for offset in DIRECTIONS {
                if hydration > 1 || self.random() < 0.25 {
                    let side = position + offset;
                    if processed.contains(&side) {
                        continue;
                    }
                    let side_hydration = if self.leaves.contains_key(&side) {
                        self.update_leaf_hydration_in(side)
                    } else {
                        self.try_place_leaf_in(side, None, environment)
                    };
                    // The Java queue permits both a dying cell (zero) and a
                    // cell no stronger than its parent to continue the wave.
                    if side_hydration == 0 || side_hydration <= hydration {
                        queue.push_back((side, side_hydration));
                    }
                }
            }
        }
        true
    }

    fn update_leaf_hydration_in(&mut self, position: IVec3) -> u8 {
        if !self.leaves.contains_key(&position) {
            return 0;
        }
        let hydration = self.hydration_from_neighbours(position, &self.leaves);
        if hydration == 0 {
            self.leaves.remove(&position);
        } else {
            self.leaves.insert(position, hydration);
        }
        hydration
    }

    fn is_leaf_location_suitable_in(
        &self,
        position: IVec3,
        environment: &GrowthEnvironment,
    ) -> bool {
        // This reproduces the relevant DynamicLeavesBlock placement checks in
        // our terrain model: a leaf needs replaceable air, cannot overwrite a
        // branch, and cannot grow on rooty/solid ground.
        position.y > self.root.y
            && position.y < WORLD_HEIGHT
            && !self.branches.contains_key(&position)
            && !self.leaves.contains_key(&position)
            && environment.can_place_leaf(position, self.root)
            && self.has_adequate_leaf_light(position, environment)
    }

    /// DynamicLeavesBlock#needLeaves. Existing leaves count as suitable;
    /// fresh ones receive the CellKit's default hydration of four.
    fn need_leaf_in(&mut self, position: IVec3, environment: &GrowthEnvironment) -> bool {
        self.leaves.contains_key(&position)
            || self.try_place_leaf_in(position, Some(4), environment) != 0
    }

    fn try_place_leaf_in(
        &mut self,
        position: IVec3,
        requested_hydration: Option<u8>,
        environment: &GrowthEnvironment,
    ) -> u8 {
        if !self.is_leaf_location_suitable_in(position, environment) {
            return 0;
        }
        let hydration = match requested_hydration {
            // `growLeavesIfLocationIsSuitable(..., 0)` means CellKit's
            // default hydration, not an empty leaf. The primary CellKits in
            // this prototype all use the source default of four.
            Some(0) => 4,
            Some(hydration) => hydration,
            None => self.hydration_from_neighbours(position, &self.leaves),
        };
        if hydration == 0 {
            return 0;
        }
        self.leaves.insert(position, hydration);
        hydration
    }

    fn hydration_from_neighbours(&self, position: IVec3, leaves: &HashMap<IVec3, u8>) -> u8 {
        let mut counts = [0_u8; 8];
        for (direction, offset) in DIRECTIONS.iter().enumerate() {
            let neighbour = position + *offset;
            // DynamicLeavesBlock requests the neighbouring cell from the
            // opposite side of the direction it is scanning.
            let received = if let Some(radius) = self.branches.get(&neighbour) {
                self.branch_cell_value(neighbour, *radius, opposite(direction))
            } else if let Some(hydration) = leaves.get(&neighbour) {
                self.leaf_cell_value(*hydration, opposite(direction))
            } else {
                0
            };
            counts[received as usize] = counts[received as usize].saturating_add(1);
        }
        self.solve_leaf_cell(&counts)
    }

    /// `Cell#getValueFromSide` for the three CellKits used by the bundled
    /// oak, spruce and acacia definitions.
    fn leaf_cell_value(&self, hydration: u8, side: usize) -> u8 {
        match self.form {
            TreeForm::Deciduous => hydration, // NormalCell
            TreeForm::Conifer => CONIFER_LEAF_CELL_VALUES[side][hydration as usize],
            TreeForm::Acacia => ACACIA_LEAF_CELL_VALUES[side][hydration as usize],
        }
    }

    /// `CellKit#getCellForBranch`, including the TOP_BRANCH metadata emitted
    /// by Family#getRadiusForCellKit for a twig above another branch.
    fn branch_cell_value(&self, position: IVec3, radius: u8, side: usize) -> u8 {
        if radius != 1 {
            return 0;
        }
        match self.form {
            TreeForm::Deciduous => 5, // NormalCell(5)
            TreeForm::Conifer => {
                let is_top_branch = self.branches.contains_key(&(position + IVec3::NEG_Y));
                if is_top_branch {
                    CONIFER_TOP_BRANCH_VALUES[side]
                } else {
                    CONIFER_BRANCH_VALUES[side]
                }
            }
            TreeForm::Acacia => ACACIA_BRANCH_VALUES[side],
        }
    }

    /// Direct decoding of CellKits.BasicSolver codes. Each triple is
    /// `(neighbour value, required count, result hydration)`.
    fn solve_leaf_cell(&self, counts: &[u8; 8]) -> u8 {
        let rules = match self.form {
            TreeForm::Deciduous => &DECIDUOUS_CELL_SOLVER[..],
            TreeForm::Conifer => &CONIFER_CELL_SOLVER[..],
            TreeForm::Acacia => &ACACIA_CELL_SOLVER[..],
        };
        rules
            .iter()
            .find_map(|&(value, minimum, result)| {
                (counts[value as usize] >= minimum).then_some(result)
            })
            .unwrap_or(0)
    }

    /// Builds the same breadth-first parent map used by the public Eco
    /// Machina visualizer. The Dynamic Trees graph remains authoritative for
    /// growth; only its connected wood cells enter this render hierarchy.
    fn build_hpd_nodes(&self) -> Vec<HpdNode> {
        let trunk = self.root + IVec3::Y;
        if !self.branches.contains_key(&trunk) {
            return Vec::new();
        }

        let mut nodes = vec![HpdNode {
            position: trunk,
            parent: None,
            children: Vec::new(),
            subtree_weight: 0,
            dt_radius: *self
                .branches
                .get(&trunk)
                .expect("a rooted HPD node was checked above"),
            chain_id: 0,
            position_in_chain: 0,
            chain_depth: 0,
            heavy_child: None,
        }];
        let mut lookup = HashMap::with_capacity(self.branches.len());
        lookup.insert(trunk, 0_usize);
        let mut queue = VecDeque::from([0_usize]);
        while let Some(parent) = queue.pop_front() {
            let parent_position = nodes[parent].position;
            for offset in DIRECTIONS {
                let child_position = parent_position + offset;
                if !self.branches.contains_key(&child_position)
                    || lookup.contains_key(&child_position)
                {
                    continue;
                }
                let child = nodes.len();
                nodes.push(HpdNode {
                    position: child_position,
                    parent: Some(parent),
                    children: Vec::new(),
                    subtree_weight: 0,
                    dt_radius: *self
                        .branches
                        .get(&child_position)
                        .expect("HPD children are Dynamic Trees branch cells"),
                    chain_id: 0,
                    position_in_chain: 0,
                    chain_depth: 0,
                    heavy_child: None,
                });
                lookup.insert(child_position, child);
                nodes[parent].children.push(child);
                queue.push_back(child);
            }
        }
        nodes
    }

    /// `computeSubtreeWeights` and `pickPrimaryChild` from visualizer.js.
    /// Every wood block contributes exactly one topology weight; its Dynamic
    /// Trees radius independently remains the rendering width.
    fn compute_subtree_weights(nodes: &mut [HpdNode], index: usize) -> u32 {
        let children = nodes[index].children.clone();
        let mut weight = 1_u32;
        let mut heavy_child = None;
        let mut best_weight = 0_u32;
        for child in children {
            let child_weight = Self::compute_subtree_weights(nodes, child);
            weight = weight.saturating_add(child_weight);
            // Strict comparison retains the source's first-neighbour tie
            // behaviour and its stable, angular branch choice.
            if child_weight > best_weight {
                best_weight = child_weight;
                heavy_child = Some(child);
            }
        }
        nodes[index].subtree_weight = weight;
        nodes[index].heavy_child = heavy_child;
        weight
    }

    /// `assignChainsWithMeta` from visualizer.js. A primary child continues
    /// its parent's chain; every other child begins a sub-chain one depth down.
    fn assign_hpd_chains(
        nodes: &mut [HpdNode],
        index: usize,
        chain_id: u16,
        position_in_chain: u16,
        chain_depth: u16,
        next_chain_id: &mut u16,
    ) {
        nodes[index].chain_id = chain_id;
        nodes[index].position_in_chain = position_in_chain;
        nodes[index].chain_depth = chain_depth;
        let heavy_child = nodes[index].heavy_child;
        let children = nodes[index].children.clone();
        for child in children {
            if Some(child) == heavy_child {
                Self::assign_hpd_chains(
                    nodes,
                    child,
                    chain_id,
                    position_in_chain.saturating_add(1),
                    chain_depth,
                    next_chain_id,
                );
            } else {
                let child_chain = *next_chain_id;
                *next_chain_id = next_chain_id.saturating_add(1);
                Self::assign_hpd_chains(
                    nodes,
                    child,
                    child_chain,
                    0,
                    chain_depth.saturating_add(1),
                    next_chain_id,
                );
            }
        }
    }

    fn node_spine(nodes: &[HpdNode], index: usize) -> (Vec3, Vec3) {
        let node = &nodes[index];
        let centre = node.position.as_vec3() + Vec3::splat(0.5);
        let spine_in = node.parent.map_or(centre - Vec3::Y * 0.5, |parent| {
            (centre + nodes[parent].position.as_vec3() + Vec3::splat(0.5)) * 0.5
        });
        let spine_out = node
            .heavy_child
            .map_or(centre + (centre - spine_in), |child| {
                (centre + nodes[child].position.as_vec3() + Vec3::splat(0.5)) * 0.5
            });
        (spine_in, spine_out)
    }

    fn push_hpd_segment(
        segments: &mut Vec<GpuTreeSegment>,
        start: Vec3,
        end: Vec3,
        half_width: f32,
        form: u32,
        kind: u32,
        variant: u32,
    ) {
        segments.push(GpuTreeSegment {
            start_radius: [start.x, start.y, start.z, half_width],
            end_radius: [end.x, end.y, end.z, half_width],
            style: [form, kind, variant, 0],
        });
    }

    /// Direct 3D lift of Eco Machina's public visualizer. Its 2D face and
    /// diagonal leaf search becomes all 26 immediate 3D neighbours here;
    /// Dynamic Trees itself still uses only its six face-connected wood graph.
    fn foliage_neighbour_offsets() -> Vec<IVec3> {
        let mut offsets = DIRECTIONS.to_vec();
        for y in -1..=1 {
            for z in -1..=1 {
                for x in -1..=1 {
                    let offset = IVec3::new(x, y, z);
                    let distance = x.abs() + y.abs() + z.abs();
                    if distance > 1 {
                        offsets.push(offset);
                    }
                }
            }
        }
        offsets
    }

    /// Produces the visualizer's single rectangular wood strip per block,
    /// sub-chain connectors from a parent's `spineIn`, and leaf-owned textured
    /// square connectors. Strip widths come directly from the Dynamic Trees
    /// branch radius; no branch-cell/sleeve renderer is retained.
    fn eco_machina_segments(&self) -> Vec<GpuTreeSegment> {
        let mut nodes = self.build_hpd_nodes();
        if nodes.is_empty() {
            return Vec::new();
        }
        Self::compute_subtree_weights(&mut nodes, 0);
        let mut chain_count = 1_u16;
        Self::assign_hpd_chains(&mut nodes, 0, 0, 0, 0, &mut chain_count);
        debug_assert!(nodes.iter().all(|node| node.chain_id < chain_count));

        let mut node_lookup = HashMap::with_capacity(nodes.len());
        for (index, node) in nodes.iter().enumerate() {
            node_lookup.insert(node.position, index);
        }
        let spines: Vec<(Vec3, Vec3)> = (0..nodes.len())
            .map(|index| Self::node_spine(&nodes, index))
            .collect();
        let form = self.form.render_id();
        let mut segments = Vec::with_capacity(nodes.len() * 3 + self.leaves.len());

        for (index, node) in nodes.iter().enumerate() {
            let (spine_in, spine_out) = spines[index];
            Self::push_hpd_segment(
                &mut segments,
                spine_in,
                spine_out,
                f32::from(node.dt_radius) / 16.0,
                form,
                SEGMENT_WOOD_SPINE,
                0,
            );
            for &child in &node.children {
                if Some(child) == node.heavy_child {
                    continue;
                }
                let child_centre = nodes[child].position.as_vec3() + Vec3::splat(0.5);
                let connector_end =
                    (node.position.as_vec3() + Vec3::splat(0.5) + child_centre) * 0.5;
                // Keep the connector entirely inside both direct DT branch
                // radii. This is the square-prism counterpart of joining two
                // Dynamic Trees branch cells, not a synthetic HPD taper.
                let connector_half_width =
                    f32::from(node.dt_radius.min(nodes[child].dt_radius)) / 16.0;
                Self::push_hpd_segment(
                    &mut segments,
                    spine_in,
                    connector_end,
                    connector_half_width,
                    form,
                    SEGMENT_WOOD_CONNECTOR,
                    0,
                );
            }
        }

        let offsets = Self::foliage_neighbour_offsets();
        let mut leaves: Vec<IVec3> = self.leaves.keys().copied().collect();
        leaves.sort_by_key(|position| (position.x, position.y, position.z));
        for leaf in leaves {
            let leaf_centre = leaf.as_vec3() + Vec3::splat(0.5);
            let mut face_candidates = Vec::new();
            let mut diagonal_candidates = Vec::new();
            for offset in &offsets {
                let Some(&index) = node_lookup.get(&(leaf + *offset)) else {
                    continue;
                };
                let wood_centre = nodes[index].position.as_vec3() + Vec3::splat(0.5);
                let foliage_anchor = (wood_centre + leaf_centre) * 0.5;
                if spines[index].1.distance(foliage_anchor) < 0.5 {
                    continue;
                }
                if offset.abs().element_sum() == 1 {
                    face_candidates.push(index);
                } else {
                    diagonal_candidates.push(index);
                }
            }
            let candidates = if face_candidates.is_empty() {
                diagonal_candidates
            } else {
                face_candidates
            };
            let Some(index) = candidates
                .iter()
                .copied()
                .find(|index| nodes[*index].heavy_child.is_none())
                .or_else(|| candidates.first().copied())
            else {
                continue;
            };
            // The public code uses (row * 31 + column * 17) % 6. This is the
            // same deterministic six-way selection with the third 3D axis.
            let texture_variant = (leaf.x * 31 + leaf.y * 17 + leaf.z * 13).rem_euclid(6) as u32;
            Self::push_hpd_segment(
                &mut segments,
                spines[index].0,
                leaf_centre,
                f32::from(nodes[index].dt_radius) / 16.0,
                form,
                SEGMENT_FOLIAGE_CONNECTOR,
                texture_variant,
            );
        }
        segments
    }
}

const DECIDUOUS_LEAF_CLUSTER: [u8; 100] = [
    0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 3,
    4, 3, 1, 1, 4, 0, 4, 1, 1, 3, 4, 3, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 2, 3, 2, 1, 1, 3, 4, 3,
    1, 1, 2, 3, 2, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0,
    0, 0, 0, 0,
];

const CONIFER_LEAF_CLUSTER: [u8; 50] = [
    0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 1, 2, 0, 2, 1, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
];

// LeafClusters.ACACIA, centred at (3, 0, 3). The source model deliberately
// uses a low, broad, flat canopy rather than the deciduous 5×4×5 cluster.
const ACACIA_LEAF_CLUSTER: [u8; 98] = [
    // Layer 0
    0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 1, 2, 4, 0, 4, 2, 1, 1, 2, 3, 4,
    3, 2, 1, 0, 1, 2, 2, 2, 1, 0, 0, 0, 1, 1, 1, 0, 0, // Layer 1
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 2, 2, 1, 0, 0, 1, 2, 2, 2, 1, 0, 0, 1, 2, 2,
    2, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

// MatrixCell maps, in Direction order DOWN/UP/NORTH/SOUTH/WEST/EAST. These
// are copied from ConiferLeafCell and AcaciaLeafCell in Dynamic Trees.
const CONIFER_LEAF_CELL_VALUES: [[u8; 8]; 6] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 2, 4, 0, 0, 0],
    [0, 1, 2, 0, 2, 0, 0, 0],
    [0, 1, 2, 0, 2, 0, 0, 0],
    [0, 1, 2, 0, 2, 0, 0, 0],
    [0, 1, 2, 0, 2, 0, 0, 0],
];
const ACACIA_LEAF_CELL_VALUES: [[u8; 8]; 6] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 3, 3, 0, 0, 0],
    [0, 1, 2, 3, 4, 0, 0, 0],
    [0, 1, 2, 3, 4, 0, 0, 0],
    [0, 1, 2, 3, 4, 0, 0, 0],
    [0, 1, 2, 3, 4, 0, 0, 0],
];
const CONIFER_BRANCH_VALUES: [u8; 6] = [2, 2, 3, 3, 3, 3];
const CONIFER_TOP_BRANCH_VALUES: [u8; 6] = [2, 5, 3, 3, 3, 3];
const ACACIA_BRANCH_VALUES: [u8; 6] = [0, 3, 5, 5, 5, 5];

const DECIDUOUS_CELL_SOLVER: [(u8, u8, u8); 6] = [
    (5, 1, 4),
    (4, 2, 3),
    (3, 2, 2),
    (4, 1, 1),
    (3, 1, 1),
    (2, 1, 1),
];
const CONIFER_CELL_SOLVER: [(u8, u8, u8); 4] = [(5, 1, 4), (4, 1, 3), (3, 1, 2), (2, 1, 1)];
const ACACIA_CELL_SOLVER: [(u8, u8, u8); 5] =
    [(5, 1, 4), (4, 2, 3), (4, 1, 2), (3, 1, 2), (2, 1, 1)];

/// A square LOD clipmap. `origin` is the world-space lower-left corner,
/// `cell_size` is its current LOD resolution and `sample_offset` indexes the
/// flattened height buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLodLevel {
    data: [f32; 4],
}

/// Address of a 16×16 full-data source.  `detail` identifies the quadtree
/// depth: each parent is exactly a 2×2 merge of sources at the preceding
/// detail level.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct LodSectionKey {
    detail: u8,
    x: i32,
    z: i32,
}

impl LodSectionKey {
    fn parent(self) -> Option<Self> {
        let detail = self.detail.checked_add(1)?;
        (usize::from(detail) < LOD_LEVEL_COUNT).then_some(Self {
            detail,
            x: self.x.div_euclid(2),
            z: self.z.div_euclid(2),
        })
    }

    fn factor(self) -> i32 {
        LOD_LEVEL_FACTORS[self.detail as usize]
    }

    fn world_minimum(self) -> (i32, i32) {
        let side = LOD_SECTION_SIDE * CHUNK_SIZE * self.factor();
        (self.x * side, self.z * side)
    }
}

/// One Distant-Horizons-like full-data source.  A packed value keeps the
/// top height in 1/16ths plus separate top and cliff materials, so a distant
/// hill does not degenerate into a uniformly green cuboid.
#[derive(Clone)]
struct LodSection {
    key: LodSectionKey,
    columns: Vec<u32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LodGenerationRequest {
    key: LodSectionKey,
    priority: i32,
}

impl Ord for LodGenerationRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max heap; invert distance so sections nearest the
        // player are generated first.  The coordinates make ties deterministic.
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.key.detail.cmp(&self.key.detail))
            .then_with(|| other.key.z.cmp(&self.key.z))
            .then_with(|| other.key.x.cmp(&self.key.x))
    }
}

impl PartialOrd for LodGenerationRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Sparse quadtree cut plus the background full-data provider.  The renderer
/// consumes its current cut as compact clipmap metadata, while source data
/// itself remains independent, asynchronously built and persisted on disk.
struct LodQuadTree {
    center: Option<ChunkPos>,
    active: HashSet<LodSectionKey>,
    cached: HashMap<LodSectionKey, LodSection>,
    queued: HashSet<LodSectionKey>,
    requests: mpsc::Sender<LodGenerationRequest>,
    completed: mpsc::Receiver<LodSection>,
}

impl LodQuadTree {
    fn new() -> Self {
        let (request_sender, request_receiver) = mpsc::channel();
        let (completed_sender, completed_receiver) = mpsc::channel();
        thread::Builder::new()
            .name("distant-horizons-data".into())
            .spawn(move || lod_generation_worker(request_receiver, completed_sender))
            .expect("could not start Distant Horizons data worker");
        Self {
            center: None,
            active: HashSet::new(),
            cached: HashMap::new(),
            queued: HashSet::new(),
            requests: request_sender,
            completed: completed_receiver,
        }
    }

    /// Rebuilds the visible *cut* of the sparse quadtree.  Each requested
    /// source is 16×16 cells; the five nested levels form non-overlapping
    /// distance rings in the ray tracer, with an unloaded child naturally
    /// resolved by its available parent level.
    fn center_on(
        &mut self,
        center: ChunkPos,
        lod_samples: &mut [u32],
        lod_levels: &mut [GpuLodLevel; LOD_LEVEL_COUNT],
    ) {
        let _profile_span = profile_span!("LOD::rebuild visible quadtree cut");
        self.center = Some(center);
        self.active.clear();
        let samples_per_level = (LOD_GRID_SIZE * LOD_GRID_SIZE) as usize;

        for (detail, factor) in LOD_LEVEL_FACTORS.iter().copied().enumerate() {
            let half_width = LOD_GRID_SIZE / 2;
            let origin_chunk = ChunkPos {
                x: center.x - half_width * factor,
                z: center.z - half_width * factor,
            };
            let sample_offset = detail * samples_per_level;
            lod_levels[detail] = GpuLodLevel {
                data: [
                    (origin_chunk.x * CHUNK_SIZE) as f32,
                    (origin_chunk.z * CHUNK_SIZE) as f32,
                    (CHUNK_SIZE * factor) as f32,
                    sample_offset as f32,
                ],
            };

            for z in 0..LOD_GRID_SIZE {
                for x in 0..LOD_GRID_SIZE {
                    let global_chunk_x = origin_chunk.x + x * factor;
                    let global_chunk_z = origin_chunk.z + z * factor;
                    let key = LodSectionKey {
                        detail: detail as u8,
                        x: global_chunk_x.div_euclid(LOD_SECTION_SIDE * factor),
                        z: global_chunk_z.div_euclid(LOD_SECTION_SIDE * factor),
                    };
                    self.active.insert(key);
                }
            }
        }

        for key in self.active.iter().copied() {
            if let Some(parent) = key.parent() {
                debug_assert_eq!(parent.factor(), key.factor() * 2);
            }
            if self.cached.contains_key(&key) || !self.queued.insert(key) {
                continue;
            }
            let (minimum_x, minimum_z) = key.world_minimum();
            let request = LodGenerationRequest {
                key,
                priority: ((minimum_x.div_euclid(CHUNK_SIZE) - center.x).abs()
                    + (minimum_z.div_euclid(CHUNK_SIZE) - center.z).abs()),
            };
            // The worker cannot disappear while this tree owns the receiver;
            // a disconnect would be a programming error, not a recoverable
            // missing terrain condition.
            self.requests
                .send(request)
                .expect("Distant Horizons data worker stopped unexpectedly");
        }
        self.pack_visible_sources(lod_samples);
        self.prune_memory_cache(center);
    }

    /// Moves completed I/O/generation work into the data source cache.  It is
    /// intentionally non-blocking: walking across a chunk never waits for a
    /// far terrain build.
    fn collect_completed(&mut self, lod_samples: &mut [u32]) -> bool {
        let _profile_span = profile_span!("LOD::integrate completed sections");
        let mut changed = false;
        while let Ok(section) = self.completed.try_recv() {
            self.queued.remove(&section.key);
            if self.active.contains(&section.key) {
                changed |= self
                    .cached
                    .get(&section.key)
                    .is_none_or(|previous| previous.columns != section.columns);
            }
            self.cached.insert(section.key, section);
        }
        if changed {
            self.pack_visible_sources(lod_samples);
        }
        changed
    }

    fn pack_visible_sources(&self, lod_samples: &mut [u32]) {
        let _profile_span = profile_span!("LOD::pack GPU columns");
        let Some(center) = self.center else {
            return;
        };
        let samples_per_level = (LOD_GRID_SIZE * LOD_GRID_SIZE) as usize;
        lod_samples.fill(0);
        for (detail, factor) in LOD_LEVEL_FACTORS.iter().copied().enumerate() {
            let half_width = LOD_GRID_SIZE / 2;
            let origin_chunk = ChunkPos {
                x: center.x - half_width * factor,
                z: center.z - half_width * factor,
            };
            let sample_offset = detail * samples_per_level;
            for z in 0..LOD_GRID_SIZE {
                for x in 0..LOD_GRID_SIZE {
                    let global_chunk_x = origin_chunk.x + x * factor;
                    let global_chunk_z = origin_chunk.z + z * factor;
                    let key = LodSectionKey {
                        detail: detail as u8,
                        x: global_chunk_x.div_euclid(LOD_SECTION_SIDE * factor),
                        z: global_chunk_z.div_euclid(LOD_SECTION_SIDE * factor),
                    };
                    let local_x = global_chunk_x
                        .rem_euclid(LOD_SECTION_SIDE * factor)
                        .div_euclid(factor) as usize;
                    let local_z = global_chunk_z
                        .rem_euclid(LOD_SECTION_SIDE * factor)
                        .div_euclid(factor) as usize;
                    if let Some(section) = self.cached.get(&key) {
                        lod_samples[sample_offset + (x + LOD_GRID_SIZE * z) as usize] =
                            section.columns[local_x + LOD_SECTION_SIDE as usize * local_z];
                    }
                }
            }
        }
    }

    fn prune_memory_cache(&mut self, center: ChunkPos) {
        self.cached.retain(|key, _| {
            let (minimum_x, minimum_z) = key.world_minimum();
            let section_radius = LOD_SECTION_SIDE * key.factor() / CHUNK_SIZE;
            (minimum_x.div_euclid(CHUNK_SIZE) - center.x).abs() <= LOD_RADIUS + section_radius * 3
                && (minimum_z.div_euclid(CHUNK_SIZE) - center.z).abs()
                    <= LOD_RADIUS + section_radius * 3
        });
    }
}

fn lod_cache_root() -> PathBuf {
    PathBuf::from(LOD_CACHE_FOLDER)
}

fn lod_cache_path(key: LodSectionKey) -> PathBuf {
    lod_cache_root()
        .join(format!("v{LOD_CACHE_VERSION}"))
        .join(format!("detail-{}", key.detail))
        .join(format!("{}_{}.lod", key.x, key.z))
}

fn read_i32(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("four cache bytes"),
    )
}

fn load_lod_section(key: LodSectionKey) -> Option<LodSection> {
    let _profile_span = profile_span!("LOD worker::load disk cache");
    let bytes = fs::read(lod_cache_path(key)).ok()?;
    let header_size = 20;
    if bytes.len() != header_size + LOD_SECTION_COLUMN_COUNT * std::mem::size_of::<u32>()
        || bytes[0..4] != LOD_CACHE_MAGIC
        || u32::from_le_bytes(bytes[4..8].try_into().ok()?) != LOD_CACHE_VERSION
        || bytes[8] != key.detail
        || read_i32(&bytes, 12) != key.x
        || read_i32(&bytes, 16) != key.z
    {
        return None;
    }
    let columns = bytes[header_size..]
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four column bytes")))
        .collect();
    Some(LodSection { key, columns })
}

fn save_lod_section(section: &LodSection) {
    let _profile_span = profile_span!("LOD worker::save disk cache");
    let path = lod_cache_path(section.key);
    let Some(parent) = path.parent() else {
        return;
    };
    if fs::create_dir_all(parent).is_err() {
        return;
    }
    let mut bytes = Vec::with_capacity(20 + section.columns.len() * std::mem::size_of::<u32>());
    bytes.extend_from_slice(&LOD_CACHE_MAGIC);
    bytes.extend_from_slice(&LOD_CACHE_VERSION.to_le_bytes());
    bytes.push(section.key.detail);
    bytes.extend_from_slice(&[0; 3]);
    bytes.extend_from_slice(&section.key.x.to_le_bytes());
    bytes.extend_from_slice(&section.key.z.to_le_bytes());
    for column in &section.columns {
        bytes.extend_from_slice(&column.to_le_bytes());
    }
    // A cache is an optimisation.  A denied or full disk must never prevent
    // the procedural source from appearing in this session.
    let _ = fs::write(path, bytes);
}

fn pack_lod_column(height: u16, top_material: u32, side_material: u32) -> u32 {
    u32::from(height) | (top_material << 16) | (side_material << 24)
}

fn generate_lod_section(key: LodSectionKey) -> LodSection {
    let _profile_span = profile_span!("LOD worker::generate full-data section");
    let factor = key.factor();
    let cell_size = CHUNK_SIZE * factor;
    let (minimum_x, minimum_z) = key.world_minimum();
    let mut columns = Vec::with_capacity(LOD_SECTION_COLUMN_COUNT);
    for z in 0..LOD_SECTION_SIDE {
        for x in 0..LOD_SECTION_SIDE {
            let center_x = minimum_x + x * cell_size + cell_size / 2;
            let center_z = minimum_z + z * cell_size + cell_size / 2;
            // Full data records a compact vertical column.  Sampling the
            // centre and four in-cell extrema keeps a coarse source tied to
            // the detailed height field instead of a separately generated
            // far-world noise function.
            let sample_offset = (cell_size / 3).max(1);
            let samples = [
                terrain_height_units(center_x, center_z),
                terrain_height_units(center_x - sample_offset, center_z),
                terrain_height_units(center_x + sample_offset, center_z),
                terrain_height_units(center_x, center_z - sample_offset),
                terrain_height_units(center_x, center_z + sample_offset),
            ];
            let lowest = *samples.iter().min().expect("LOD samples are nonempty");
            let highest = *samples.iter().max().expect("LOD samples are nonempty");
            let average =
                samples.iter().map(|height| u32::from(*height)).sum::<u32>() / samples.len() as u32;
            // The tallest sample preserves ridges at a distance, while a
            // single valley does not punch a visible hole into a coarse cell.
            let height = u16::try_from((average * 2 + u32::from(highest)) / 3)
                .expect("terrain height fits u16");
            let slope = highest - lowest;
            let top_material = if height <= 10 * 16 {
                WATER
            } else if height > 19 * 16 && slope > 12 {
                STONE
            } else {
                GRASS
            };
            let side_material = if top_material == WATER {
                WATER
            } else if slope > (8 * factor) as u16 || height > 21 * 16 {
                STONE
            } else {
                DIRT
            };
            columns.push(pack_lod_column(height, top_material, side_material));
        }
    }
    LodSection { key, columns }
}

fn lod_generation_worker(
    requests: mpsc::Receiver<LodGenerationRequest>,
    completed: mpsc::Sender<LodSection>,
) {
    profile_thread_name!("Distant Horizons worker");
    let mut pending = BinaryHeap::new();
    loop {
        if pending.is_empty() {
            let Ok(request) = requests.recv() else {
                return;
            };
            pending.push(request);
        }
        while let Ok(request) = requests.try_recv() {
            pending.push(request);
        }
        let request = pending.pop().expect("nonempty LOD priority queue");
        let section = load_lod_section(request.key).unwrap_or_else(|| {
            let section = generate_lod_section(request.key);
            save_lod_section(&section);
            section
        });
        if completed.send(section).is_err() {
            return;
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    camera_position: [f32; 4],
    camera_forward: [f32; 4],
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    sun_direction: [f32; 4],
    // detailed origin X/Z and detailed dimensions X/Z
    detail_world: [f32; 4],
    // physical chunk offset for logical detailed (0, 0), chunk ring side, unused
    detail_ring: [f32; 4],
    // LOD grid side, LOD count, unused, camera aspect ratio
    lod_world: [f32; 4],
    // elapsed world time, forest-TLAS root plus one (zero = empty), detailed world height, unused
    simulation: [f32; 4],
}

impl Uniforms {
    fn new() -> Self {
        Self {
            camera_position: [0.0; 4],
            camera_forward: [0.0; 4],
            camera_right: [0.0; 4],
            camera_up: [0.0; 4],
            sun_direction: [0.0; 4],
            detail_world: [0.0; 4],
            detail_ring: [0.0; 4],
            lod_world: [0.0; 4],
            simulation: [0.0; 4],
        }
    }
}

struct World {
    /// Chunks currently visible through the detailed 9×9 ray-tracing ring.
    detailed: HashMap<ChunkPos, ActiveChunk>,
    /// Finished immutable chunk builds just outside the visible ring. A chunk
    /// moves from here into the renderer only as part of a complete stripe.
    prepared: HashMap<ChunkPos, ChunkBuild>,
    requested: HashMap<ChunkPos, u64>,
    slot_positions: Vec<Option<ChunkPos>>,
    build_queue: Arc<ChunkBuildQueue>,
    build_completed: mpsc::Receiver<ChunkBuild>,
    next_build_ticket: u64,
    center: Option<ChunkPos>,
    wanted_center: Option<ChunkPos>,
    detail_origin: ChunkPos,
    /// Physical chunk coordinates that back logical (0, 0) in the shader.
    detail_ring: (i32, i32),
    /// Chunk-major physical storage. A whole chunk is contiguous, so a stream
    /// update writes one range rather than repacking the full 9×9 window.
    detailed_blocks: Vec<Block>,
    lod_samples: Vec<u32>,
    lod_levels: [GpuLodLevel; LOD_LEVEL_COUNT],
    lod_tree: LodQuadTree,
    gpu_trees: Vec<GpuTree>,
    gpu_tree_segments: Vec<GpuTreeSegment>,
    gpu_tree_blas_nodes: Vec<GpuTreeBvhNode>,
    gpu_tree_tlas_nodes: Vec<GpuTreeBvhNode>,
    tree_slot_segments: Vec<usize>,
    tree_slot_blas_nodes: Vec<usize>,
    tree_tlas_root: Option<usize>,
    tree_count: usize,
    growth_time: f32,
    last_tree_tick: f32,
    growth_requests: mpsc::Sender<GrowthRequest>,
    growth_completed: mpsc::Receiver<GrowthResult>,
    growth_in_flight: bool,
}

impl World {
    fn new() -> Self {
        let build_queue = Arc::new(ChunkBuildQueue::new());
        let (build_completed_sender, build_completed) = mpsc::channel();
        let (growth_requests, growth_request_receiver) = mpsc::channel();
        let (growth_completed_sender, growth_completed) = mpsc::channel();
        for worker_index in 0..DETAIL_STREAM_WORKERS {
            let queue = build_queue.clone();
            let completed = build_completed_sender.clone();
            thread::Builder::new()
                .name(format!("detailed-chunk-build-{worker_index}"))
                .spawn(move || chunk_build_worker(queue, completed))
                .expect("could not start detailed chunk worker");
        }
        thread::Builder::new()
            .name("dynamic-tree-growth".into())
            .spawn(move || {
                dynamic_tree_growth_worker(growth_request_receiver, growth_completed_sender)
            })
            .expect("could not start Dynamic Trees growth worker");
        Self {
            detailed: HashMap::new(),
            prepared: HashMap::new(),
            requested: HashMap::new(),
            slot_positions: vec![None; DETAIL_CHUNK_COUNT],
            build_queue,
            build_completed,
            next_build_ticket: 1,
            center: None,
            wanted_center: None,
            detail_origin: ChunkPos { x: 0, z: 0 },
            detail_ring: (0, 0),
            detailed_blocks: vec![AIR; DETAIL_CHUNK_COUNT * CHUNK_BLOCK_COUNT],
            lod_samples: vec![0; LOD_SAMPLE_COUNT],
            lod_levels: [GpuLodLevel::zeroed(); LOD_LEVEL_COUNT],
            lod_tree: LodQuadTree::new(),
            gpu_trees: vec![GpuTree::zeroed(); MAX_TREES],
            gpu_tree_segments: vec![GpuTreeSegment::zeroed(); MAX_GPU_TREE_SEGMENTS],
            gpu_tree_blas_nodes: vec![GpuTreeBvhNode::zeroed(); MAX_GPU_TREE_BLAS_NODES],
            gpu_tree_tlas_nodes: vec![GpuTreeBvhNode::zeroed(); MAX_GPU_TREE_TLAS_NODES],
            tree_slot_segments: vec![0; DETAIL_CHUNK_COUNT],
            tree_slot_blas_nodes: vec![0; DETAIL_CHUNK_COUNT],
            tree_tlas_root: None,
            tree_count: 0,
            growth_time: 0.0,
            last_tree_tick: 0.0,
            growth_requests,
            growth_completed,
            growth_in_flight: false,
        }
    }

    /// Builds the first detailed ring before the game loop begins. From this
    /// point on, all generation and BLAS work stays on background workers.
    fn bootstrap(&mut self, position: Vec3) {
        let center = ChunkPos::from_world(position);
        self.center = Some(center);
        self.wanted_center = Some(center);
        self.detail_origin = Self::detail_origin_for(center);
        self.detail_ring = (0, 0);
        self.rebuild_lod_buffer(center);
        self.schedule_prefetch(center, Vec3::ZERO);
        while !self.visible_window_is_prepared(center) {
            let build = self
                .build_completed
                .recv()
                .expect("detailed chunk workers stopped unexpectedly");
            self.accept_chunk_build(build);
        }

        let mut changes = WorldChanges::default();
        for z in 0..DETAIL_DIAMETER {
            for x in 0..DETAIL_DIAMETER {
                let position = ChunkPos {
                    x: self.detail_origin.x + x,
                    z: self.detail_origin.z + z,
                };
                let slot = self.slot_for(position);
                self.install_prepared_chunk(position, slot, &mut changes);
            }
        }
        let initial_slots = changes.detail_slots.clone();
        self.repaint_tree_voxels(&initial_slots, &mut changes);
        self.rebuild_tree_tlas();
        self.schedule_prefetch(center, Vec3::ZERO);
    }

    /// Schedules work, integrates a bounded number of worker results and
    /// advances by at most one complete stripe. It has no synchronous terrain
    /// generation, tree packing or full-buffer upload path.
    fn stream_around(&mut self, position: Vec3, heading: Vec3) -> WorldChanges {
        let _profile_span = profile_span!("World::integrate async chunks");
        let wanted = ChunkPos::from_world(position);
        let mut changes = WorldChanges::default();
        if self.wanted_center != Some(wanted) {
            self.wanted_center = Some(wanted);
            self.rebuild_lod_buffer(wanted);
            changes.lod_layout_changed = true;
        }
        self.schedule_prefetch(wanted, heading);
        if let Some(render_center) = self.center
            && render_center != wanted
        {
            let next_center = ChunkPos {
                x: render_center.x + (wanted.x - render_center.x).signum(),
                z: render_center.z + (wanted.z - render_center.z).signum(),
            };
            // A long camera jump still advances through complete prepared
            // windows rather than exposing an uninitialised ring slot.
            self.schedule_prefetch(next_center, heading);
        }
        self.integrate_completed_chunk_builds();
        self.prune_prepared(wanted);
        self.try_advance_visible_window(wanted, &mut changes);
        changes
    }

    fn advance(&mut self, seconds: f32) -> WorldChanges {
        let _profile_span = profile_span!("Dynamic Trees::integrate growth");
        let changes = self.integrate_growth_results();
        self.growth_time += seconds;
        let ticks =
            ((self.growth_time - self.last_tree_tick) / TREE_GROWTH_TICK_SECONDS).floor() as u32;
        if ticks == 0 || self.growth_in_flight {
            return changes;
        }

        // The worker shares immutable terrain through Arc and receives a
        // snapshot of the Dynamic Trees graph revisions. It performs the
        // source-style growth simulation and local BLAS work off the frame
        // thread, then publishes only changed chunks.
        let chunks = self
            .detailed
            .iter()
            .map(|(&position, active)| GrowthSnapshotChunk {
                position,
                revision: active.growth_revision,
                chunk: active.chunk.clone(),
            })
            .collect::<Vec<_>>();
        self.growth_requests
            .send(GrowthRequest { ticks, chunks })
            .expect("Dynamic Trees growth worker stopped unexpectedly");
        self.last_tree_tick += ticks as f32 * TREE_GROWTH_TICK_SECONDS;
        self.growth_in_flight = true;
        changes
    }

    /// Integrates completed distant-data jobs without stalling the frame.
    /// The caller uploads the compact buffer only when this returns true.
    fn collect_lod_updates(&mut self) -> bool {
        self.lod_tree.collect_completed(&mut self.lod_samples)
    }

    fn active_tree_count(&self) -> usize {
        self.tree_count
    }

    fn rebuild_lod_buffer(&mut self, center: ChunkPos) {
        let _profile_span = profile_span!("LOD::schedule visible sources");
        self.lod_tree
            .center_on(center, &mut self.lod_samples, &mut self.lod_levels);
    }

    fn detail_origin_for(center: ChunkPos) -> ChunkPos {
        ChunkPos {
            x: center.x - DETAIL_RADIUS,
            z: center.z - DETAIL_RADIUS,
        }
    }

    fn slot_for(&self, position: ChunkPos) -> usize {
        let local_x = position.x - self.detail_origin.x;
        let local_z = position.z - self.detail_origin.z;
        assert!(
            (0..DETAIL_DIAMETER).contains(&local_x) && (0..DETAIL_DIAMETER).contains(&local_z),
            "chunk is outside the detailed ring"
        );
        let physical_x = (self.detail_ring.0 + local_x).rem_euclid(DETAIL_DIAMETER);
        let physical_z = (self.detail_ring.1 + local_z).rem_euclid(DETAIL_DIAMETER);
        (physical_x + DETAIL_DIAMETER * physical_z) as usize
    }

    fn is_in_visible_window(&self, position: ChunkPos) -> bool {
        (self.detail_origin.x..self.detail_origin.x + DETAIL_DIAMETER).contains(&position.x)
            && (self.detail_origin.z..self.detail_origin.z + DETAIL_DIAMETER).contains(&position.z)
    }

    fn is_in_prefetch_window(center: ChunkPos, position: ChunkPos) -> bool {
        position.distance(center) <= DETAIL_PREFETCH_RADIUS
    }

    fn visible_window_is_prepared(&self, center: ChunkPos) -> bool {
        let origin = Self::detail_origin_for(center);
        (0..DETAIL_DIAMETER).all(|z| {
            (0..DETAIL_DIAMETER).all(|x| {
                let position = ChunkPos {
                    x: origin.x + x,
                    z: origin.z + z,
                };
                self.detailed.contains_key(&position) || self.prepared.contains_key(&position)
            })
        })
    }

    fn schedule_prefetch(&mut self, center: ChunkPos, heading: Vec3) {
        let heading = Vec2::new(heading.x, heading.z).normalize_or_zero();
        for z in -DETAIL_PREFETCH_RADIUS..=DETAIL_PREFETCH_RADIUS {
            for x in -DETAIL_PREFETCH_RADIUS..=DETAIL_PREFETCH_RADIUS {
                let position = ChunkPos {
                    x: center.x + x,
                    z: center.z + z,
                };
                if self.detailed.contains_key(&position)
                    || self.prepared.contains_key(&position)
                    || self.requested.contains_key(&position)
                {
                    continue;
                }
                let direction_bias = heading.dot(Vec2::new(x as f32, z as f32));
                let ticket = self.next_build_ticket;
                self.next_build_ticket += 1;
                self.requested.insert(position, ticket);
                self.build_queue.push(ChunkBuildRequest {
                    position,
                    ticket,
                    priority: x.abs().max(z.abs()) * 32 - (direction_bias * 12.0) as i32,
                });
            }
        }
    }

    fn accept_chunk_build(&mut self, build: ChunkBuild) {
        if self.requested.get(&build.position).copied() != Some(build.ticket) {
            return;
        }
        self.requested.remove(&build.position);
        let Some(center) = self.wanted_center else {
            return;
        };
        let near_render_window = self.center.is_some_and(|render_center| {
            Self::is_in_prefetch_window(render_center, build.position)
        });
        if (Self::is_in_prefetch_window(center, build.position) || near_render_window)
            && !self.detailed.contains_key(&build.position)
        {
            self.prepared.insert(build.position, build);
        }
    }

    fn integrate_completed_chunk_builds(&mut self) {
        // Moving the result into a hash map is bounded, allocation-free work
        // in the frame loop. Chunk generation and BVH construction completed
        // before it reached this point.
        for _ in 0..8 {
            let Ok(build) = self.build_completed.try_recv() else {
                break;
            };
            self.accept_chunk_build(build);
        }
    }

    fn prune_prepared(&mut self, center: ChunkPos) {
        let render_center = self.center;
        let keep = |position: ChunkPos| {
            Self::is_in_prefetch_window(center, position)
                || render_center
                    .is_some_and(|current| Self::is_in_prefetch_window(current, position))
        };
        self.prepared.retain(|position, _| keep(*position));
        self.requested.retain(|position, _| keep(*position));
    }

    fn install_prepared_chunk(
        &mut self,
        position: ChunkPos,
        slot: usize,
        changes: &mut WorldChanges,
    ) {
        let build = self
            .prepared
            .remove(&position)
            .expect("a complete stripe is installed only from prepared chunks");
        if let Some(previous_position) = self.slot_positions[slot]
            && previous_position != position
            && let Some(previous) = self.detailed.remove(&previous_position)
            && self
                .wanted_center
                .is_some_and(|center| Self::is_in_prefetch_window(center, previous_position))
        {
            self.prepared.insert(
                previous_position,
                ChunkBuild {
                    position: previous_position,
                    ticket: 0,
                    chunk: previous.chunk,
                    tree_geometry: previous.tree_geometry,
                },
            );
        }

        let block_start = slot * CHUNK_BLOCK_COUNT;
        self.detailed_blocks[block_start..block_start + CHUNK_BLOCK_COUNT]
            .copy_from_slice(&build.chunk.blocks);
        self.materialize_tree_slot(slot, &build.tree_geometry);
        self.slot_positions[slot] = Some(position);
        self.detailed.insert(
            position,
            ActiveChunk {
                chunk: build.chunk,
                tree_geometry: build.tree_geometry,
                growth_revision: 0,
            },
        );
        changes.detail_slots.push(slot);
        changes.tree_slots.push(slot);
    }

    fn materialize_tree_slot(&mut self, slot: usize, geometry: &ChunkTreeGeometry) {
        assert!(geometry.segments.len() <= TREE_ATLAS_SEGMENTS_PER_CHUNK);
        assert!(geometry.blas_nodes.len() <= TREE_ATLAS_BLAS_NODES_PER_CHUNK);
        let tree_start = slot * MAX_TREES_PER_CHUNK;
        let segment_start = slot * TREE_ATLAS_SEGMENTS_PER_CHUNK;
        let blas_start = slot * TREE_ATLAS_BLAS_NODES_PER_CHUNK;
        for local_tree in 0..MAX_TREES_PER_CHUNK {
            let mut tree = geometry.trees[local_tree];
            if local_tree < geometry.tree_count {
                tree.layout[0] += segment_start as f32;
                tree.layout[2] += blas_start as f32;
            } else {
                tree = GpuTree::zeroed();
            }
            self.gpu_trees[tree_start + local_tree] = tree;
        }
        self.gpu_tree_segments[segment_start..segment_start + geometry.segments.len()]
            .copy_from_slice(&geometry.segments);
        // BLAS nodes are built in worker-local coordinates. Relocate every
        // stored index into this physical chunk slot before the shader sees
        // it: inner children address the BLAS atlas, leaves address the
        // segment atlas. The root offset alone is not sufficient.
        for (local_index, local_node) in geometry.blas_nodes.iter().copied().enumerate() {
            let mut node = local_node;
            if node.data[2] == 1 {
                node.data[0] += segment_start as u32;
            } else {
                node.data[0] += blas_start as u32;
                node.data[1] += blas_start as u32;
            }
            self.gpu_tree_blas_nodes[blas_start + local_index] = node;
        }
        self.tree_slot_segments[slot] = geometry.segments.len();
        self.tree_slot_blas_nodes[slot] = geometry.blas_nodes.len();
    }

    fn mark_tree_slots(&self, chunk: &ActiveChunk, slots: &mut HashSet<usize>) {
        for tree in &chunk.chunk.trees {
            if let Some(slot) = self.slot_for_world_cell(tree.root) {
                slots.insert(slot);
            }
            for &leaf in tree.leaves.keys() {
                if let Some(slot) = self.slot_for_world_cell(leaf) {
                    slots.insert(slot);
                }
            }
        }
    }

    fn slot_for_world_cell(&self, cell: IVec3) -> Option<usize> {
        if !(0..WORLD_HEIGHT).contains(&cell.y) {
            return None;
        }
        let position = ChunkPos {
            x: cell.x.div_euclid(CHUNK_SIZE),
            z: cell.z.div_euclid(CHUNK_SIZE),
        };
        self.is_in_visible_window(position)
            .then(|| self.slot_for(position))
    }

    fn block_index_in_slot(slot: usize, cell: IVec3) -> usize {
        let local_x = cell.x.rem_euclid(CHUNK_SIZE) as usize;
        let local_z = cell.z.rem_euclid(CHUNK_SIZE) as usize;
        slot * CHUNK_BLOCK_COUNT
            + local_x
            + CHUNK_SIZE as usize * (local_z + CHUNK_SIZE as usize * cell.y as usize)
    }

    fn reset_slot_to_terrain(&mut self, slot: usize) {
        let position = self.slot_positions[slot].expect("active ring slot has a position");
        let chunk = &self
            .detailed
            .get(&position)
            .expect("active ring slot has a chunk")
            .chunk;
        let start = slot * CHUNK_BLOCK_COUNT;
        self.detailed_blocks[start..start + CHUNK_BLOCK_COUNT].copy_from_slice(&chunk.blocks);
    }

    fn repaint_tree_voxels(&mut self, initial_slots: &[usize], changes: &mut WorldChanges) {
        let mut slots = initial_slots.iter().copied().collect::<HashSet<_>>();
        for &slot in &slots {
            self.reset_slot_to_terrain(slot);
        }
        let mut roots = Vec::new();
        let mut leaves = Vec::new();
        for chunk in self.detailed.values() {
            for tree in &chunk.chunk.trees {
                roots.push(tree.root);
                leaves.extend(
                    tree.leaves
                        .keys()
                        .copied()
                        .map(|position| (position, tree.form.leaf_material())),
                );
            }
        }
        for root in roots {
            if let Some(slot) = self.slot_for_world_cell(root)
                && slots.contains(&slot)
            {
                let index = Self::block_index_in_slot(slot, root);
                let height = (self.detailed_blocks[index] >> 8) & 255;
                self.detailed_blocks[index] = block(ROOTY_SOIL, height);
            }
        }
        for (leaf, material) in leaves {
            if let Some(slot) = self.slot_for_world_cell(leaf)
                && slots.contains(&slot)
            {
                let index = Self::block_index_in_slot(slot, leaf);
                if self.detailed_blocks[index] == AIR {
                    self.detailed_blocks[index] = block(material, 16);
                }
            }
        }
        changes.detail_slots.extend(slots.drain());
        changes.detail_slots.sort_unstable();
        changes.detail_slots.dedup();
    }

    fn try_advance_visible_window(&mut self, wanted: ChunkPos, changes: &mut WorldChanges) {
        let Some(center) = self.center else {
            return;
        };
        let delta_x = (wanted.x - center.x).signum();
        let delta_z = (wanted.z - center.z).signum();
        if delta_x == 0 && delta_z == 0 {
            return;
        }
        let next_center = ChunkPos {
            x: center.x + delta_x,
            z: center.z + delta_z,
        };
        if !self.visible_window_is_prepared(next_center) {
            return;
        }

        let old_positions = self.detailed.keys().copied().collect::<HashSet<_>>();
        let next_origin = Self::detail_origin_for(next_center);
        let additions = (0..DETAIL_DIAMETER)
            .flat_map(|z| {
                (0..DETAIL_DIAMETER).map(move |x| ChunkPos {
                    x: next_origin.x + x,
                    z: next_origin.z + z,
                })
            })
            .filter(|position| !old_positions.contains(position))
            .collect::<Vec<_>>();

        self.center = Some(next_center);
        self.detail_origin = next_origin;
        self.detail_ring = (
            (self.detail_ring.0 + delta_x).rem_euclid(DETAIL_DIAMETER),
            (self.detail_ring.1 + delta_z).rem_euclid(DETAIL_DIAMETER),
        );
        let mut repaint_slots = HashSet::new();
        for position in additions {
            let slot = self.slot_for(position);
            if let Some(previous_position) = self.slot_positions[slot]
                && let Some(previous) = self.detailed.get(&previous_position)
            {
                self.mark_tree_slots(previous, &mut repaint_slots);
            }
            self.install_prepared_chunk(position, slot, changes);
            let installed = self
                .detailed
                .get(&position)
                .expect("installed chunk is active");
            self.mark_tree_slots(installed, &mut repaint_slots);
            repaint_slots.insert(slot);
        }
        self.repaint_tree_voxels(&repaint_slots.into_iter().collect::<Vec<_>>(), changes);
        if changes.has_tree_updates() {
            changes.tree_slots.sort_unstable();
            changes.tree_slots.dedup();
            self.rebuild_tree_tlas();
            changes.tree_tlas_changed = true;
        }
    }

    fn rebuild_tree_tlas(&mut self) {
        let _profile_span = profile_span!("Eco Machina::build forest TLAS");
        self.gpu_tree_tlas_nodes.fill(GpuTreeBvhNode::zeroed());
        self.tree_tlas_root = None;
        let mut primitives = Vec::new();
        for (tree_index, tree) in self.gpu_trees.iter().copied().enumerate() {
            if tree.layout[1] <= 0.0 {
                continue;
            }
            let centre = Vec3::from_array([tree.bounds[0], tree.bounds[1], tree.bounds[2]]);
            let radius = tree.bounds[3];
            let extent = Vec3::splat(radius);
            primitives.push(TreeTlasPrimitive {
                tree_index,
                minimum: centre - extent,
                maximum: centre + extent,
                centroid: centre,
            });
        }
        self.tree_count = primitives.len();
        if !primitives.is_empty() {
            let mut node_cursor = 0;
            self.tree_tlas_root = Some(build_tree_tlas(
                &mut primitives,
                &mut self.gpu_tree_tlas_nodes,
                &mut node_cursor,
            ));
        }
    }

    fn integrate_growth_results(&mut self) -> WorldChanges {
        let mut changes = WorldChanges::default();
        let mut repaint_slots = HashSet::new();
        while let Ok(result) = self.growth_completed.try_recv() {
            self.growth_in_flight = false;
            for update in result.updates {
                let applies = self
                    .detailed
                    .get(&update.position)
                    .is_some_and(|active| active.growth_revision == update.revision);
                if !applies {
                    continue;
                }
                let slot = self.slot_for(update.position);
                if let Some(previous) = self.detailed.get(&update.position) {
                    self.mark_tree_slots(previous, &mut repaint_slots);
                }
                self.materialize_tree_slot(slot, &update.tree_geometry);
                self.detailed.insert(
                    update.position,
                    ActiveChunk {
                        chunk: update.chunk,
                        tree_geometry: update.tree_geometry,
                        growth_revision: update.revision + 1,
                    },
                );
                let current = self
                    .detailed
                    .get(&update.position)
                    .expect("growth update remains active");
                self.mark_tree_slots(current, &mut repaint_slots);
                repaint_slots.insert(slot);
                changes.tree_slots.push(slot);
            }
        }
        if changes.has_tree_updates() {
            self.repaint_tree_voxels(&repaint_slots.into_iter().collect::<Vec<_>>(), &mut changes);
            changes.tree_slots.sort_unstable();
            changes.tree_slots.dedup();
            self.rebuild_tree_tlas();
            changes.tree_tlas_changed = true;
        }
        changes
    }
}

#[derive(Deserialize)]
struct TreeTextureSet {
    bark: String,
    end_grain: String,
    leaves: String,
}

#[derive(Deserialize)]
struct TreeTextureConfig {
    tile_size: u32,
    oak: TreeTextureSet,
    spruce: TreeTextureSet,
    acacia: TreeTextureSet,
    connectors: [String; 6],
}

fn load_tree_texture_config() -> (PathBuf, TreeTextureConfig) {
    let config_path = PathBuf::from(TREE_TEXTURE_CONFIG_PATH);
    let text = fs::read_to_string(&config_path).unwrap_or_else(|error| {
        panic!(
            "could not read required tree texture config {}: {error}",
            config_path.display()
        )
    });
    let config = serde_json::from_str::<TreeTextureConfig>(&text).unwrap_or_else(|error| {
        panic!(
            "could not parse required tree texture config {}: {error}",
            config_path.display()
        )
    });
    assert_eq!(
        config.tile_size, TREE_TEXTURE_TILE_SIZE,
        "tree texture config must declare a {TREE_TEXTURE_TILE_SIZE}px tile"
    );
    let asset_dir = config_path
        .parent()
        .expect("tree texture config path has a parent directory")
        .to_path_buf();
    (asset_dir, config)
}

fn decode_tree_png(path: &Path, label: &str) -> Vec<u8> {
    let image = image::open(path)
        .unwrap_or_else(|error| {
            panic!(
                "could not decode required {label} texture {}: {error}",
                path.display()
            )
        })
        .to_rgba8();
    assert_eq!(
        image.dimensions(),
        (TREE_TEXTURE_TILE_SIZE, TREE_TEXTURE_TILE_SIZE),
        "required {label} texture must be exactly {TREE_TEXTURE_TILE_SIZE}x{TREE_TEXTURE_TILE_SIZE}"
    );
    image.into_raw()
}

fn decode_alpha_tree_png(path: &Path, label: &str) -> Vec<u8> {
    let pixels = decode_tree_png(path, label);
    assert!(
        pixels.chunks_exact(4).any(|pixel| pixel[3] < 255),
        "required {label} texture must preserve transparent alpha pixels"
    );
    pixels
}

fn configured_tree_asset(asset_dir: &Path, relative_path: &str, label: &str) -> PathBuf {
    let relative_path = Path::new(relative_path);
    assert!(
        !relative_path.is_absolute(),
        "{label} texture path must be relative to {TREE_TEXTURE_CONFIG_PATH}"
    );
    let path = asset_dir.join(relative_path);
    assert!(
        path.is_file(),
        "required {label} texture is missing: {}",
        path.display()
    );
    path
}

fn blit_tree_tile(atlas: &mut [u8], atlas_width: u32, tile_x: u32, tile_y: u32, source: &[u8]) {
    let row_bytes = (TREE_TEXTURE_TILE_SIZE * 4) as usize;
    for source_y in 0..TREE_TEXTURE_TILE_SIZE as usize {
        let source_start = source_y * row_bytes;
        let target_x = tile_x * TREE_TEXTURE_TILE_SIZE;
        let target_y = tile_y * TREE_TEXTURE_TILE_SIZE + source_y as u32;
        let target_start = ((target_x + target_y * atlas_width) * 4) as usize;
        atlas[target_start..target_start + row_bytes]
            .copy_from_slice(&source[source_start..source_start + row_bytes]);
    }
}

/// Produces the immutable tree-material atlas consumed by the ray tracer.
/// The supplied oak/spruce bark, end-grain, alpha leaf textures, and six
/// foliage connectors are mandatory embedded assets. Acacia keeps its
/// deterministic authored atlas tiles until matching source PNGs are supplied.
fn generate_tree_texture_atlas() -> (Vec<u8>, u32, u32) {
    let width = TREE_TEXTURE_TILE_SIZE * TREE_TEXTURE_COLUMNS;
    let height = TREE_TEXTURE_TILE_SIZE * TREE_TEXTURE_ROWS;
    let mut pixels = vec![0_u8; (width * height * 4) as usize];

    // All occupied atlas tiles are external runtime assets. The zeroed pixels
    // in unused slots never participate in a material lookup.
    let (asset_dir, config) = load_tree_texture_config();
    let species = [
        ("oak", &config.oak, 0_u32),
        ("spruce", &config.spruce, 1_u32),
        ("acacia", &config.acacia, 2_u32),
    ];
    for (name, textures, column) in species {
        let bark_label = format!("{name} bark");
        let bark_path = configured_tree_asset(&asset_dir, &textures.bark, &bark_label);
        let end_grain_label = format!("{name} end grain");
        let end_grain_path =
            configured_tree_asset(&asset_dir, &textures.end_grain, &end_grain_label);
        let leaves_label = format!("{name} leaves");
        let leaves_path = configured_tree_asset(&asset_dir, &textures.leaves, &leaves_label);
        let bark = decode_tree_png(&bark_path, &bark_label);
        let end_grain = decode_tree_png(&end_grain_path, &end_grain_label);
        let leaves = decode_alpha_tree_png(&leaves_path, &leaves_label);
        blit_tree_tile(&mut pixels, width, column, 0, &bark);
        blit_tree_tile(&mut pixels, width, column, 1, &end_grain);
        blit_tree_tile(&mut pixels, width, column + 3, 0, &leaves);
    }

    let connector_tiles = [(3_u32, 1_u32), (4, 1), (5, 1), (0, 2), (1, 2), (2, 2)];
    for (index, ((tile_x, tile_y), relative_path)) in connector_tiles
        .into_iter()
        .zip(config.connectors.iter())
        .enumerate()
    {
        let label = format!("connector-{}", index + 1);
        let path = configured_tree_asset(&asset_dir, relative_path, &label);
        let connector = decode_alpha_tree_png(&path, &label);
        blit_tree_tile(&mut pixels, width, tile_x, tile_y, &connector);
    }
    (pixels, width, height)
}

fn create_tree_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::Sampler) {
    let (pixels, width, height) = generate_tree_texture_atlas();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("externally configured tree material atlas"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        texture.size(),
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("tree texture atlas sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        ..Default::default()
    });
    (texture, view, sampler)
}

fn hash_u32(mut value: u32) -> u32 {
    value ^= value >> 16;
    value = value.wrapping_mul(0x7feb_352d);
    value ^= value >> 15;
    value = value.wrapping_mul(0x846c_a68b);
    value ^ (value >> 16)
}

fn hash_2d(x: i32, z: i32, salt: u32) -> u32 {
    hash_u32((x as u32).wrapping_mul(0x1f12_3bb5) ^ (z as u32).wrapping_mul(0x9e37_79b9) ^ salt)
}

fn smooth_noise(x: i32, z: i32, spacing: i32, salt: u32) -> f32 {
    let cell_x = x.div_euclid(spacing);
    let cell_z = z.div_euclid(spacing);
    let fraction_x = x.rem_euclid(spacing) as f32 / spacing as f32;
    let fraction_z = z.rem_euclid(spacing) as f32 / spacing as f32;
    let fade = |value: f32| value * value * (3.0 - 2.0 * value);
    let value = |ix: i32, iz: i32| -> f32 { (hash_2d(ix, iz, salt) & 0xffff) as f32 / 65535.0 };
    let a = value(cell_x, cell_z);
    let b = value(cell_x + 1, cell_z);
    let c = value(cell_x, cell_z + 1);
    let d = value(cell_x + 1, cell_z + 1);
    let x0 = a + (b - a) * fade(fraction_x);
    let x1 = c + (d - c) * fade(fraction_x);
    x0 + (x1 - x0) * fade(fraction_z)
}

/// Ground height in 1/16th block units.  This is shared by detail chunks and
/// Distant-Horizons-style low-detail chunks, so the two representations meet.
fn terrain_height_units(x: i32, z: i32) -> u16 {
    let continent = smooth_noise(x, z, 56, 0x91a7);
    let hills = smooth_noise(x, z, 15, 0x6f21);
    let ridges = smooth_noise(x, z, 7, 0x031d);
    let whole = 8 + (continent * 8.0 + hills * 4.0 + ridges * 2.0) as i32;
    let slab = 1 + (hash_2d(x, z, 0xa81f) % 16) as i32;
    (whole * 16 + slab).clamp(2, (WORLD_HEIGHT - 4) * 16) as u16
}

fn generate_chunk(position: ChunkPos) -> Chunk {
    let side = CHUNK_SIZE as usize;
    let mut blocks = vec![AIR; side * side * WORLD_HEIGHT as usize];
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let world_x = position.x * CHUNK_SIZE + x;
            let world_z = position.z * CHUNK_SIZE + z;
            let ground = terrain_height_units(world_x, world_z) as i32;
            let full = ground / 16;
            let partial = ground % 16;
            for y in 0..WORLD_HEIGHT {
                let index = (x + CHUNK_SIZE * (z + CHUNK_SIZE * y)) as usize;
                blocks[index] = if y < full {
                    if y < full - 3 {
                        block(STONE, 16)
                    } else {
                        block(DIRT, 16)
                    }
                } else if y == full && partial > 0 {
                    block(GRASS, partial as u32)
                } else if y <= 10 && y > full {
                    // A few valley ponds make the shallow top slabs easy to see.
                    block(WATER, 16)
                } else {
                    AIR
                };
            }
        }
    }

    let mut trees = Vec::new();
    // Each chunk can grow one or two independent Dynamic Trees graphs. The
    // renderer later decomposes them into Eco Machina HPD chains.
    for tree_slot in 0..2 {
        let h = hash_2d(position.x, position.z, 0x4d3b_1f01 + tree_slot);
        if h % 100 >= 72 {
            continue;
        }
        let local_x = 2 + ((h >> 8) % 12) as i32;
        let local_z = 2 + ((h >> 16) % 12) as i32;
        let world_x = position.x * CHUNK_SIZE + local_x;
        let world_z = position.z * CHUNK_SIZE + local_z;
        let ground_units = terrain_height_units(world_x, world_z);
        if ground_units < 160 {
            continue;
        }
        trees.push(Tree::new(
            IVec3::new(world_x, i32::from(ground_units / 16), world_z),
            h,
        ));
    }
    Chunk {
        blocks: blocks.into(),
        trees,
    }
}

fn build_chunk_tree_geometry(chunk: &Chunk) -> ChunkTreeGeometry {
    let _profile_span = profile_span!("Eco Machina worker::build chunk BLAS");
    let mut trees = [GpuTree::zeroed(); MAX_TREES_PER_CHUNK];
    let mut packed_segments = vec![GpuTreeSegment::zeroed(); TREE_ATLAS_SEGMENTS_PER_CHUNK];
    let mut packed_nodes = vec![GpuTreeBvhNode::zeroed(); TREE_ATLAS_BLAS_NODES_PER_CHUNK];
    let mut tree_count = 0;
    let mut segment_cursor = 0;
    let mut node_cursor = 0;

    for tree in &chunk.trees {
        if tree_count >= MAX_TREES_PER_CHUNK {
            break;
        }
        let segments = tree.eco_machina_segments();
        if segments.is_empty() {
            continue;
        }
        assert!(
            segments.len() <= MAX_RENDER_SEGMENTS_PER_TREE,
            "Eco Machina segment budget exceeded for one tree"
        );
        assert!(
            segment_cursor + segments.len() <= TREE_ATLAS_SEGMENTS_PER_CHUNK,
            "Eco Machina chunk segment atlas exhausted"
        );
        let mut primitives = segments
            .into_iter()
            .map(segment_primitive)
            .collect::<Vec<_>>();
        let segment_start = segment_cursor;
        let node_start = node_cursor;
        let blas_root = build_tree_blas(
            &mut primitives,
            &mut packed_segments,
            &mut segment_cursor,
            &mut packed_nodes,
            &mut node_cursor,
        );
        let segment_count = segment_cursor - segment_start;
        if segment_count == 0 {
            continue;
        }
        let bounds = packed_nodes[blas_root];
        let minimum = Vec3::from_array([bounds.minimum[0], bounds.minimum[1], bounds.minimum[2]]);
        let maximum = Vec3::from_array([bounds.maximum[0], bounds.maximum[1], bounds.maximum[2]]);
        let centre = (minimum + maximum) * 0.5;
        trees[tree_count] = GpuTree {
            bounds: [centre.x, centre.y, centre.z, (maximum - centre).length()],
            // Local offsets are relocated to the ring slot by World.
            layout: [
                segment_start as f32,
                segment_count as f32,
                blas_root as f32,
                (node_cursor - node_start) as f32,
            ],
            appearance: [
                tree.form.render_id() as f32,
                f32::from(tree.fertility),
                0.0,
                0.0,
            ],
        };
        tree_count += 1;
    }
    packed_segments.truncate(segment_cursor);
    packed_nodes.truncate(node_cursor);
    ChunkTreeGeometry {
        trees,
        tree_count,
        segments: packed_segments,
        blas_nodes: packed_nodes,
    }
}

fn chunk_build_worker(queue: Arc<ChunkBuildQueue>, completed: mpsc::Sender<ChunkBuild>) {
    profile_thread_name!("Detailed chunk worker");
    loop {
        let request = queue.pop();
        let _profile_span = profile_span!("Detailed chunk worker::build");
        let chunk = generate_chunk(request.position);
        let tree_geometry = build_chunk_tree_geometry(&chunk);
        if completed
            .send(ChunkBuild {
                position: request.position,
                ticket: request.ticket,
                chunk,
                tree_geometry,
            })
            .is_err()
        {
            return;
        }
    }
}

fn growth_environment_from_snapshot(chunks: &[GrowthSnapshotChunk]) -> GrowthEnvironment {
    let _profile_span = profile_span!("Dynamic Trees worker::build environment");
    let mut environment = GrowthEnvironment::default();
    for snapshot in chunks {
        let chunk = &snapshot.chunk;
        for y in 0..WORLD_HEIGHT {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let index = (x + CHUNK_SIZE * (z + CHUNK_SIZE * y)) as usize;
                    if chunk.blocks[index] != AIR {
                        environment.terrain.insert(IVec3::new(
                            snapshot.position.x * CHUNK_SIZE + x,
                            y,
                            snapshot.position.z * CHUNK_SIZE + z,
                        ));
                    }
                }
            }
        }
        for tree in &chunk.trees {
            for (&part, &radius) in &tree.branches {
                environment.tree_part_owner.insert(part, tree.root);
                environment
                    .tree_parts
                    .insert(part, EnvironmentTreePart::Branch(radius));
            }
            for &part in tree.leaves.keys() {
                environment.tree_part_owner.insert(part, tree.root);
                environment
                    .tree_parts
                    .insert(part, EnvironmentTreePart::Leaf);
            }
        }
    }
    environment
}

fn dynamic_tree_growth_worker(
    requests: mpsc::Receiver<GrowthRequest>,
    completed: mpsc::Sender<GrowthResult>,
) {
    profile_thread_name!("Dynamic Trees growth worker");
    while let Ok(request) = requests.recv() {
        let _profile_span = profile_span!("Dynamic Trees worker::grow and build");
        let environment = growth_environment_from_snapshot(&request.chunks);
        let mut updates = Vec::new();
        for snapshot in request.chunks {
            let mut chunk = snapshot.chunk;
            let mut chunk_changed = false;
            chunk.trees.retain_mut(|tree| {
                for _ in 0..request.ticks {
                    chunk_changed |= tree.update_in(&environment);
                }
                // Match Dynamic Trees root cleanup after terminal rot.
                !tree.branches.is_empty()
            });
            if chunk_changed {
                let tree_geometry = build_chunk_tree_geometry(&chunk);
                updates.push(GrowthChunkUpdate {
                    position: snapshot.position,
                    revision: snapshot.revision,
                    chunk,
                    tree_geometry,
                });
            }
        }
        if completed.send(GrowthResult { updates }).is_err() {
            return;
        }
    }
}

struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
}

impl Camera {
    fn new() -> Self {
        Self {
            position: Vec3::new(5.5, 24.0, 25.5),
            yaw: -2.52,
            pitch: -0.16,
        }
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    fn move_with_keys(&mut self, keys: &HashSet<KeyCode>, delta: f32) {
        let mut movement = Vec3::ZERO;
        let forward = self.forward();
        let flat_forward = Vec3::new(forward.x, 0.0, forward.z).normalize();
        if keys.contains(&KeyCode::KeyW) {
            movement += flat_forward;
        }
        if keys.contains(&KeyCode::KeyS) {
            movement -= flat_forward;
        }
        if keys.contains(&KeyCode::KeyD) {
            movement += self.right();
        }
        if keys.contains(&KeyCode::KeyA) {
            movement -= self.right();
        }
        if keys.contains(&KeyCode::Space) {
            movement += Vec3::Y;
        }
        if keys.contains(&KeyCode::ShiftLeft) || keys.contains(&KeyCode::ShiftRight) {
            movement -= Vec3::Y;
        }
        if movement.length_squared() > 0.0 {
            let speed = if keys.contains(&KeyCode::ControlLeft) {
                24.0
            } else {
                8.0
            };
            self.position += movement.normalize() * speed * delta;
        }
        let turn_speed = 1.55 * delta;
        if keys.contains(&KeyCode::ArrowLeft) {
            self.yaw -= turn_speed;
        }
        if keys.contains(&KeyCode::ArrowRight) {
            self.yaw += turn_speed;
        }
        if keys.contains(&KeyCode::ArrowUp) {
            self.pitch += turn_speed;
        }
        if keys.contains(&KeyCode::ArrowDown) {
            self.pitch -= turn_speed;
        }
        self.pitch = self.pitch.clamp(-1.45, 1.45);
    }
}

// `wgpu` only exposes portable timing around command-encoder/render-pass
// boundaries.  A ray tracer implemented as one fragment pass therefore has
// one honest in-engine GPU scope; shader-instruction analysis remains the job
// of PIX/Nsight, where these pass labels are visible.
#[cfg(feature = "profiling")]
const GPU_PROFILER_TIMESTAMP_RING_SIZE: usize = 6;
#[cfg(feature = "profiling")]
const GPU_PROFILER_QUERIES_PER_FRAME: u32 = 2;
#[cfg(feature = "profiling")]
const GPU_PROFILER_HISTORY_SIZE: usize = 180;
#[cfg(feature = "profiling")]
const GPU_PROFILER_RESOLVE_STRIDE: u64 = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT;

#[cfg(feature = "profiling")]
#[derive(Clone, Copy)]
struct GpuTimestampFrame {
    readback_slot: usize,
    first_query: u32,
}

#[cfg(feature = "profiling")]
struct GpuTimestampReadback {
    buffer: wgpu::Buffer,
    in_flight: bool,
}

#[cfg(feature = "profiling")]
struct GpuTimestampCompletion {
    readback_slot: usize,
    succeeded: bool,
}

#[cfg(feature = "profiling")]
#[derive(Default)]
struct RollingTimings {
    samples_ms: VecDeque<f64>,
}

#[cfg(feature = "profiling")]
impl RollingTimings {
    fn record(&mut self, milliseconds: f64) {
        if self.samples_ms.len() == GPU_PROFILER_HISTORY_SIZE {
            self.samples_ms.pop_front();
        }
        self.samples_ms.push_back(milliseconds);
    }

    fn average(&self) -> Option<f64> {
        (!self.samples_ms.is_empty())
            .then(|| self.samples_ms.iter().sum::<f64>() / self.samples_ms.len() as f64)
    }

    fn percentile(&self, percentile: f64) -> Option<f64> {
        let mut samples = self.samples_ms.iter().copied().collect::<Vec<_>>();
        if samples.is_empty() {
            return None;
        }
        samples.sort_by(f64::total_cmp);
        let index = ((samples.len() - 1) as f64 * percentile.clamp(0.0, 1.0)).ceil() as usize;
        Some(samples[index])
    }
}

/// Timestamp-query manager for the full GPU ray-tracing pass. Query results
/// are copied into a six-frame ring and mapped only after the GPU completes;
/// neither profiling nor a slow capture client is allowed to block rendering.
#[cfg(feature = "profiling")]
struct GpuPassProfiler {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    readbacks: Vec<GpuTimestampReadback>,
    completed_sender: mpsc::Sender<GpuTimestampCompletion>,
    completed_receiver: mpsc::Receiver<GpuTimestampCompletion>,
    timestamp_period_ns: f64,
    next_readback_slot: usize,
    timings: RollingTimings,
    discarded_samples: u64,
}

#[cfg(feature = "profiling")]
impl GpuPassProfiler {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let query_count = GPU_PROFILER_TIMESTAMP_RING_SIZE as u32 * GPU_PROFILER_QUERIES_PER_FRAME;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("RayVoxel::GPU timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RayVoxel::GPU timestamp resolve"),
            size: GPU_PROFILER_TIMESTAMP_RING_SIZE as u64 * GPU_PROFILER_RESOLVE_STRIDE,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readbacks = (0..GPU_PROFILER_TIMESTAMP_RING_SIZE)
            .map(|index| GpuTimestampReadback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("RayVoxel::GPU timestamp readback {index}")),
                    size: u64::from(GPU_PROFILER_QUERIES_PER_FRAME)
                        * std::mem::size_of::<u64>() as u64,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                in_flight: false,
            })
            .collect();
        let (completed_sender, completed_receiver) = mpsc::channel();
        Self {
            query_set,
            resolve_buffer,
            readbacks,
            completed_sender,
            completed_receiver,
            timestamp_period_ns: f64::from(queue.get_timestamp_period()),
            next_readback_slot: 0,
            timings: RollingTimings::default(),
            discarded_samples: 0,
        }
    }

    fn reserve_frame(&mut self) -> Option<GpuTimestampFrame> {
        for _ in 0..self.readbacks.len() {
            let readback_slot = self.next_readback_slot;
            self.next_readback_slot = (self.next_readback_slot + 1) % self.readbacks.len();
            let readback = &mut self.readbacks[readback_slot];
            if !readback.in_flight {
                readback.in_flight = true;
                return Some(GpuTimestampFrame {
                    readback_slot,
                    first_query: readback_slot as u32 * GPU_PROFILER_QUERIES_PER_FRAME,
                });
            }
        }
        // The render thread never waits for profiling readback.  A capture
        // overload costs one sample rather than one frame hitch.
        self.discarded_samples += 1;
        None
    }

    fn timestamp_writes(&self, frame: GpuTimestampFrame) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(frame.first_query),
            end_of_pass_write_index: Some(frame.first_query + 1),
        }
    }

    fn encode_resolve(&self, encoder: &mut wgpu::CommandEncoder, frame: GpuTimestampFrame) {
        let byte_offset = frame.readback_slot as u64 * GPU_PROFILER_RESOLVE_STRIDE;
        let byte_count =
            u64::from(GPU_PROFILER_QUERIES_PER_FRAME) * std::mem::size_of::<u64>() as u64;
        encoder.resolve_query_set(
            &self.query_set,
            frame.first_query..frame.first_query + GPU_PROFILER_QUERIES_PER_FRAME,
            &self.resolve_buffer,
            byte_offset,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            byte_offset,
            &self.readbacks[frame.readback_slot].buffer,
            0,
            byte_count,
        );
    }

    fn map_after_submit(&mut self, frame: GpuTimestampFrame) {
        let sender = self.completed_sender.clone();
        self.readbacks[frame.readback_slot].buffer.map_async(
            wgpu::MapMode::Read,
            ..,
            move |result| {
                let _ = sender.send(GpuTimestampCompletion {
                    readback_slot: frame.readback_slot,
                    succeeded: result.is_ok(),
                });
            },
        );
    }

    fn poll(&mut self, device: &wgpu::Device) {
        // Poll, never Wait: the following channel only contains mappings the
        // GPU has already completed.
        let _ = device.poll(wgpu::PollType::Poll);
        while let Ok(completion) = self.completed_receiver.try_recv() {
            let readback = &mut self.readbacks[completion.readback_slot];
            readback.in_flight = false;
            if !completion.succeeded {
                self.discarded_samples += 1;
                continue;
            }
            let timestamps = match readback.buffer.get_mapped_range(..) {
                Ok(bytes) if bytes.len() == 16 => {
                    let begin = u64::from_le_bytes(bytes[0..8].try_into().expect("timestamp size"));
                    let end = u64::from_le_bytes(bytes[8..16].try_into().expect("timestamp size"));
                    drop(bytes);
                    readback.buffer.unmap();
                    Some((begin, end))
                }
                Ok(bytes) => {
                    drop(bytes);
                    readback.buffer.unmap();
                    None
                }
                Err(_) => None,
            };
            let Some((begin, end)) = timestamps else {
                self.discarded_samples += 1;
                continue;
            };
            // Timestamp absolute values may wrap. A negative interval is not
            // a useful performance sample and is deliberately discarded.
            if end < begin {
                self.discarded_samples += 1;
                continue;
            }
            let milliseconds = (end - begin) as f64 * self.timestamp_period_ns / 1_000_000.0;
            if !milliseconds.is_finite() || !(0.0..=1_000.0).contains(&milliseconds) {
                self.discarded_samples += 1;
                continue;
            }
            self.timings.record(milliseconds);
            tracy_client::plot!("GPU Raytrace (ms)", milliseconds);
        }
    }

    fn summary(&self) -> String {
        match (self.timings.average(), self.timings.percentile(0.95)) {
            (Some(average), Some(p95)) => format!("GPU ray {average:.2} ms · p95 {p95:.2} ms"),
            _ => "GPU ray collecting…".to_owned(),
        }
    }
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    detail_buffer: wgpu::Buffer,
    lod_buffer: wgpu::Buffer,
    lod_info_buffer: wgpu::Buffer,
    tree_buffer: wgpu::Buffer,
    tree_segment_buffer: wgpu::Buffer,
    tree_blas_buffer: wgpu::Buffer,
    tree_tlas_buffer: wgpu::Buffer,
    camera: Camera,
    world: World,
    world_time: f32,
    #[cfg(feature = "profiling")]
    gpu_profiler: Option<GpuPassProfiler>,
}

impl State {
    async fn new(window: Arc<Window>) -> Result<Self, String> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window)
            .map_err(|error| format!("could not create the rendering surface: {error}"))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|error| format!("no suitable graphics adapter: {error}"))?;
        let profiling_timestamps_supported = cfg!(feature = "profiling")
            && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ray voxel device"),
                required_features: if profiling_timestamps_supported {
                    wgpu::Features::TIMESTAMP_QUERY
                } else {
                    wgpu::Features::empty()
                },
                ..Default::default()
            })
            .await
            .map_err(|error| format!("could not create graphics device: {error}"))?;
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or("no compatible surface format")?;
        surface.configure(&device, &config);

        let mut world = World::new();
        let camera = Camera::new();
        world.bootstrap(camera.position);

        let uniforms = Uniforms::new();
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ray world uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let detail_buffer = storage_buffer(
            &device,
            "streamed detailed voxel chunks",
            &world.detailed_blocks,
        );
        let lod_buffer = storage_buffer(
            &device,
            "Distant Horizons full-data columns",
            &world.lod_samples,
        );
        let lod_info_buffer = storage_buffer(
            &device,
            "distant horizons clipmap metadata",
            &world.lod_levels,
        );
        let tree_buffer = storage_buffer(&device, "Dynamic Trees bounds", &world.gpu_trees);
        let tree_segment_buffer = storage_buffer(
            &device,
            "eco machina hpd tree segments",
            &world.gpu_tree_segments,
        );
        let tree_blas_buffer = storage_buffer(
            &device,
            "Eco Machina tree BLAS nodes",
            &world.gpu_tree_blas_nodes,
        );
        let tree_tlas_buffer = storage_buffer(
            &device,
            "Eco Machina forest TLAS nodes",
            &world.gpu_tree_tlas_nodes,
        );
        #[cfg(feature = "profiling")]
        let gpu_profiler =
            profiling_timestamps_supported.then(|| GpuPassProfiler::new(&device, &queue));
        let (_tree_texture, tree_texture_view, tree_texture_sampler) =
            create_tree_texture(&device, &queue);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ray world bind group layout"),
            entries: &[
                buffer_layout_entry(0, wgpu::BufferBindingType::Uniform),
                buffer_layout_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_layout_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_layout_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_layout_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_layout_entry(5, wgpu::BufferBindingType::Storage { read_only: true }),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                buffer_layout_entry(8, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_layout_entry(9, wgpu::BufferBindingType::Storage { read_only: true }),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ray world bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: detail_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lod_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tree_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: lod_info_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tree_segment_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&tree_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&tree_texture_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: tree_blas_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: tree_tlas_buffer.as_entire_binding(),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DDA ray tracing shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ray world pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ray-traced voxel pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            pipeline,
            bind_group,
            uniform_buffer,
            detail_buffer,
            lod_buffer,
            lod_info_buffer,
            tree_buffer,
            tree_segment_buffer,
            tree_blas_buffer,
            tree_tlas_buffer,
            camera,
            world,
            world_time: 18.0,
            #[cfg(feature = "profiling")]
            gpu_profiler,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }

    fn update(&mut self, keys: &HashSet<KeyCode>, seconds: f32) {
        let _profile_span = profile_span!("Frame::update world and uploads");
        self.camera.move_with_keys(keys, seconds);
        self.world_time += seconds;
        let mut stream_changes = self
            .world
            .stream_around(self.camera.position, self.camera.forward());
        stream_changes.merge(self.world.advance(seconds));
        let lod_changed = self.world.collect_lod_updates();

        for slot in &stream_changes.detail_slots {
            let block_start = *slot * CHUNK_BLOCK_COUNT;
            self.queue.write_buffer(
                &self.detail_buffer,
                (block_start * std::mem::size_of::<Block>()) as u64,
                bytemuck::cast_slice(
                    &self.world.detailed_blocks[block_start..block_start + CHUNK_BLOCK_COUNT],
                ),
            );
        }

        for slot in &stream_changes.tree_slots {
            let tree_start = *slot * MAX_TREES_PER_CHUNK;
            self.queue.write_buffer(
                &self.tree_buffer,
                (tree_start * std::mem::size_of::<GpuTree>()) as u64,
                bytemuck::cast_slice(
                    &self.world.gpu_trees[tree_start..tree_start + MAX_TREES_PER_CHUNK],
                ),
            );
            let segment_count = self.world.tree_slot_segments[*slot];
            if segment_count > 0 {
                let segment_start = *slot * TREE_ATLAS_SEGMENTS_PER_CHUNK;
                self.queue.write_buffer(
                    &self.tree_segment_buffer,
                    (segment_start * std::mem::size_of::<GpuTreeSegment>()) as u64,
                    bytemuck::cast_slice(
                        &self.world.gpu_tree_segments[segment_start..segment_start + segment_count],
                    ),
                );
            }
            let node_count = self.world.tree_slot_blas_nodes[*slot];
            if node_count > 0 {
                let node_start = *slot * TREE_ATLAS_BLAS_NODES_PER_CHUNK;
                self.queue.write_buffer(
                    &self.tree_blas_buffer,
                    (node_start * std::mem::size_of::<GpuTreeBvhNode>()) as u64,
                    bytemuck::cast_slice(
                        &self.world.gpu_tree_blas_nodes[node_start..node_start + node_count],
                    ),
                );
            }
        }
        if stream_changes.tree_tlas_changed {
            self.queue.write_buffer(
                &self.tree_tlas_buffer,
                0,
                bytemuck::cast_slice(&self.world.gpu_tree_tlas_nodes),
            );
        }
        if stream_changes.lod_layout_changed || lod_changed {
            self.queue.write_buffer(
                &self.lod_buffer,
                0,
                bytemuck::cast_slice(&self.world.lod_samples),
            );
        }
        if stream_changes.lod_layout_changed {
            self.queue.write_buffer(
                &self.lod_info_buffer,
                0,
                bytemuck::cast_slice(&self.world.lod_levels),
            );
        }

        let day_phase = self.world_time * std::f32::consts::TAU / 150.0;
        let sun = Vec3::new(
            day_phase.cos() * 0.55,
            day_phase.sin() * 0.9,
            day_phase.sin() * 0.32,
        )
        .normalize();
        let forward = self.camera.forward();
        let right = self.camera.right();
        let up = right.cross(forward).normalize();
        let mut uniforms = Uniforms::new();
        uniforms.camera_position = self.camera.position.extend(1.0).to_array();
        uniforms.camera_forward = forward.extend(0.0).to_array();
        uniforms.camera_right = right.extend(0.0).to_array();
        uniforms.camera_up = up.extend(0.0).to_array();
        uniforms.sun_direction = sun.extend(0.0).to_array();
        uniforms.detail_world = [
            (self.world.detail_origin.x * CHUNK_SIZE) as f32,
            (self.world.detail_origin.z * CHUNK_SIZE) as f32,
            (DETAIL_DIAMETER * CHUNK_SIZE) as f32,
            (DETAIL_DIAMETER * CHUNK_SIZE) as f32,
        ];
        uniforms.detail_ring = [
            self.world.detail_ring.0 as f32,
            self.world.detail_ring.1 as f32,
            DETAIL_DIAMETER as f32,
            0.0,
        ];
        uniforms.lod_world = [
            LOD_GRID_SIZE as f32,
            LOD_LEVEL_COUNT as f32,
            0.0,
            self.size.width as f32 / self.size.height.max(1) as f32,
        ];
        uniforms.simulation = [
            self.world_time,
            self.world
                .tree_tlas_root
                .map_or(0.0, |root| root as f32 + 1.0),
            WORLD_HEIGHT as f32,
            0.0,
        ];
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn render(&mut self) {
        let _profile_span = profile_span!("Renderer::encode, submit, present");
        #[cfg(feature = "profiling")]
        if let Some(profiler) = &mut self.gpu_profiler {
            profiler.poll(&self.device);
        }
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };
        #[cfg(feature = "profiling")]
        let gpu_timestamp_frame = self
            .gpu_profiler
            .as_mut()
            .and_then(GpuPassProfiler::reserve_frame);
        #[cfg(feature = "profiling")]
        let gpu_timestamp_writes = gpu_timestamp_frame.map(|frame| {
            self.gpu_profiler
                .as_ref()
                .expect("timestamp frame has an owning profiler")
                .timestamp_writes(frame)
        });
        #[cfg(not(feature = "profiling"))]
        let gpu_timestamp_writes = None;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RayVoxel::Frame"),
            });
        encoder.push_debug_group("RayVoxel::Frame");
        encoder.insert_debug_marker("RayVoxel::Raytrace");
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("RayVoxel::Raytrace"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: gpu_timestamp_writes,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.push_debug_group("RayVoxel::Raytrace");
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
            pass.pop_debug_group();
        }
        #[cfg(feature = "profiling")]
        if let Some(frame) = gpu_timestamp_frame {
            self.gpu_profiler
                .as_ref()
                .expect("timestamp frame has an owning profiler")
                .encode_resolve(&mut encoder, frame);
        }
        encoder.pop_debug_group();
        self.queue.submit(Some(encoder.finish()));
        #[cfg(feature = "profiling")]
        if let Some(frame) = gpu_timestamp_frame {
            self.gpu_profiler
                .as_mut()
                .expect("timestamp frame has an owning profiler")
                .map_after_submit(frame);
        }
        self.queue.present(output);
    }

    fn status(&self) -> String {
        let status = format!(
            "RayVoxel — {} detailed chunks · 256-chunk LOD horizon · {} Dynamic Trees (Eco Machina)",
            self.world.detailed.len(),
            self.world.active_tree_count()
        );
        #[cfg(feature = "profiling")]
        {
            let mut profiled_status = status;
            let profiler_status = self.gpu_profiler.as_ref().map_or(
                "GPU timestamps unavailable".to_owned(),
                GpuPassProfiler::summary,
            );
            profiled_status.push_str(" · ");
            profiled_status.push_str(&profiler_status);
            profiled_status
        }
        #[cfg(not(feature = "profiling"))]
        status
    }
}

fn storage_buffer<T: Pod>(
    device: &wgpu::Device,
    label: &'static str,
    values: &[T],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    })
}

fn buffer_layout_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
    pressed_keys: HashSet<KeyCode>,
    last_frame: Instant,
    last_title_update: f32,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            state: None,
            pressed_keys: HashSet::new(),
            last_frame: Instant::now(),
            last_title_update: 0.0,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        profile_thread_name!("RayVoxel main");
        let _profile_span = profile_span!("App::initialise renderer");
        let attributes = Window::default_attributes()
            .with_title("RayVoxel — starting renderer")
            .with_inner_size(PhysicalSize::new(1280, 720));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("could not create game window"),
        );
        let state = pollster::block_on(State::new(window.clone()))
            .unwrap_or_else(|error| panic!("failed to initialise the voxel renderer: {error}"));
        self.last_frame = Instant::now();
        self.window = Some(window);
        self.state = Some(state);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(state) = &mut self.state {
                    state.resize(size);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if code == KeyCode::Escape {
                                event_loop.exit();
                            }
                            self.pressed_keys.insert(code);
                        }
                        ElementState::Released => {
                            self.pressed_keys.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let _profile_span = profile_span!("Frame");
                let now = Instant::now();
                let delta = (now - self.last_frame).as_secs_f32().min(0.05);
                self.last_frame = now;
                if let (Some(state), Some(window)) = (&mut self.state, &self.window) {
                    state.update(&self.pressed_keys, delta);
                    if state.world_time - self.last_title_update > 1.0 {
                        window.set_title(&state.status());
                        self.last_title_update = state.world_time;
                    }
                    state.render();
                    profile_frame_mark!();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

fn main() -> Result<(), winit::error::EventLoopError> {
    env_logger::init();
    #[cfg(feature = "profiling")]
    let _tracy_client = tracy_client::Client::start();
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut App::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "profiling")]
    #[test]
    fn gpu_profiler_rolling_p95_uses_completed_samples_only() {
        let mut timings = RollingTimings::default();
        for sample in [1.0, 2.0, 3.0, 10.0, 4.0] {
            timings.record(sample);
        }
        assert_eq!(timings.average(), Some(4.0));
        assert_eq!(timings.percentile(0.95), Some(10.0));
    }

    #[test]
    fn generated_terrain_has_valid_sixteenth_heights() {
        for x in -32..32 {
            for z in -32..32 {
                let height = terrain_height_units(x, z);
                assert!((2..=(WORLD_HEIGHT - 4) as u16 * 16).contains(&height));
                let fractional = height % 16;
                assert!(fractional <= 15);
            }
        }
    }

    #[test]
    fn streamed_windows_have_fixed_gpu_sizes() {
        let mut world = World::new();
        world.bootstrap(Vec3::new(0.0, 20.0, 0.0));
        assert_eq!(
            world.detailed.len(),
            (DETAIL_DIAMETER * DETAIL_DIAMETER) as usize
        );
        assert_eq!(
            world.detailed_blocks.len(),
            (DETAIL_DIAMETER * CHUNK_SIZE * DETAIL_DIAMETER * CHUNK_SIZE * WORLD_HEIGHT) as usize
        );
        assert_eq!(world.lod_samples.len(), LOD_SAMPLE_COUNT);
        assert_eq!(world.lod_levels.len(), LOD_LEVEL_COUNT);
        assert!(world.tree_count <= MAX_TREES);
    }

    #[test]
    fn detailed_ring_keeps_retained_chunks_in_their_physical_slots() {
        let mut world = World::new();
        world.bootstrap(Vec3::new(0.0, 20.0, 0.0));
        let retained = ChunkPos { x: 0, z: 0 };
        let original_slot = world.slot_for(retained);
        let next_center = ChunkPos { x: 1, z: 0 };

        while !world.visible_window_is_prepared(next_center) {
            let build = world
                .build_completed
                .recv()
                .expect("detailed chunk workers stopped unexpectedly");
            world.accept_chunk_build(build);
        }
        let changes = world.stream_around(Vec3::new(16.1, 20.0, 0.0), Vec3::X);

        assert_eq!(world.center, Some(next_center));
        assert_eq!(world.slot_for(retained), original_slot);
        assert!(changes.detail_slots.len() >= DETAIL_DIAMETER as usize);
        assert!(changes.detail_slots.len() < DETAIL_CHUNK_COUNT);
    }

    #[test]
    fn chunk_tree_blas_indices_relocate_into_the_gpu_atlas() {
        let mut world = World::new();
        let geometry = ChunkTreeGeometry {
            trees: [
                GpuTree {
                    bounds: [0.0, 1.0, 0.0, 1.0],
                    layout: [0.0, 2.0, 0.0, 3.0],
                    appearance: [0.0; 4],
                },
                GpuTree::zeroed(),
            ],
            tree_count: 1,
            segments: vec![GpuTreeSegment::zeroed(), GpuTreeSegment::zeroed()],
            blas_nodes: vec![
                GpuTreeBvhNode {
                    minimum: [0.0; 4],
                    maximum: [1.0; 4],
                    data: [1, 2, 0, 0],
                },
                GpuTreeBvhNode {
                    minimum: [0.0; 4],
                    maximum: [1.0; 4],
                    data: [0, 1, 1, 0],
                },
                GpuTreeBvhNode {
                    minimum: [0.0; 4],
                    maximum: [1.0; 4],
                    data: [1, 1, 1, 0],
                },
            ],
        };
        let slot = 7;
        world.materialize_tree_slot(slot, &geometry);
        let segment_base = slot * TREE_ATLAS_SEGMENTS_PER_CHUNK;
        let node_base = slot * TREE_ATLAS_BLAS_NODES_PER_CHUNK;

        assert_eq!(
            world.gpu_trees[slot * MAX_TREES_PER_CHUNK].layout[0],
            segment_base as f32
        );
        assert_eq!(
            world.gpu_trees[slot * MAX_TREES_PER_CHUNK].layout[2],
            node_base as f32
        );
        assert_eq!(
            world.gpu_tree_blas_nodes[node_base].data[0],
            (node_base + 1) as u32
        );
        assert_eq!(
            world.gpu_tree_blas_nodes[node_base].data[1],
            (node_base + 2) as u32
        );
        assert_eq!(
            world.gpu_tree_blas_nodes[node_base + 1].data[0],
            segment_base as u32
        );
        assert_eq!(
            world.gpu_tree_blas_nodes[node_base + 2].data[0],
            (segment_base + 1) as u32
        );
    }

    #[test]
    fn lod_sources_follow_quadtree_parent_addresses() {
        let child = LodSectionKey {
            detail: 2,
            x: -3,
            z: 5,
        };
        assert_eq!(
            child.parent(),
            Some(LodSectionKey {
                detail: 3,
                x: -2,
                z: 2,
            })
        );
        let outermost = LodSectionKey {
            detail: LOD_LEVEL_COUNT as u8 - 1,
            x: 0,
            z: 0,
        };
        assert!(outermost.parent().is_none());
    }

    #[test]
    fn lod_full_data_keeps_surface_and_cliff_materials() {
        let section = generate_lod_section(LodSectionKey {
            detail: 1,
            x: 0,
            z: 0,
        });
        assert_eq!(section.columns.len(), LOD_SECTION_COLUMN_COUNT);
        assert!(section.columns.iter().all(|packed| {
            let height = packed & 65535;
            let top = (packed >> 16) & 255;
            let side = (packed >> 24) & 255;
            height > 0
                && matches!(top, GRASS | STONE | WATER)
                && matches!(side, DIRT | STONE | WATER)
        }));
    }

    #[test]
    fn dynamic_tree_uses_branch_and_leaf_cells() {
        let mut tree = Tree::new(IVec3::new(0, 12, 0), 0x1234_5678);
        let initial_branches = tree.branches.len();
        for _ in 0..18 {
            tree.grow_pulse();
        }
        assert!(tree.branches.len() >= initial_branches);
        assert!(tree.branches.contains_key(&(tree.root + IVec3::Y)));
        assert!(
            tree.branches
                .values()
                .all(|radius| (1..=8).contains(radius))
        );
        assert!(!tree.leaves.is_empty());
        assert!(
            tree.leaves
                .values()
                .all(|hydration| (1..=7).contains(hydration))
        );
    }

    #[test]
    fn worldgen_uses_embedded_dynamic_trees_jocodes() {
        // "J" is index 9 in Dynamic Trees' six-bit alphabet: UP, UP.
        assert_eq!(Tree::decode_jocode("J"), vec![UP as u8, UP as u8]);

        for (seed, expected_form) in [
            (0_u32, TreeForm::Conifer),
            (2, TreeForm::Acacia),
            (3, TreeForm::Deciduous),
        ] {
            let tree = Tree::new(IVec3::new(16, 12, -16), seed);
            assert!(matches!(
                (tree.form, expected_form),
                (TreeForm::Conifer, TreeForm::Conifer)
                    | (TreeForm::Acacia, TreeForm::Acacia)
                    | (TreeForm::Deciduous, TreeForm::Deciduous)
            ));
            assert!(tree.branches.len() > 4);
            assert!(tree.branches.contains_key(&(tree.root + IVec3::Y)));
            assert!(
                tree.branches
                    .values()
                    .all(|radius| (1..=8).contains(radius))
            );
            assert!(!tree.leaves.is_empty());
        }
    }

    #[test]
    fn source_species_growth_parameters_are_preserved() {
        let root = IVec3::new(19, 12, -7);
        let mut tree = Tree::new(root, 0);

        tree.form = TreeForm::Deciduous;
        assert_eq!(tree.signal_energy(), 12.0);
        assert_eq!(tree.tapering(), 0.30);
        assert_eq!(tree.up_probability(), 2);
        assert_eq!(tree.lowest_branch_height(), 3);
        assert_eq!(tree.growth_rate(), 0.8);

        tree.form = TreeForm::Conifer;
        assert!((16.0..=20.0).contains(&tree.signal_energy()));
        assert_eq!(tree.tapering(), 0.25);
        assert_eq!(tree.up_probability(), 3);
        assert_eq!(tree.lowest_branch_height(), 3);
        assert_eq!(tree.growth_rate(), 0.9);

        tree.form = TreeForm::Acacia;
        assert_eq!(tree.signal_energy(), 12.0);
        assert_eq!(tree.tapering(), 0.15);
        assert_eq!(tree.up_probability(), 0);
        assert_eq!(tree.lowest_branch_height(), 3);
        assert_eq!(tree.growth_rate(), 0.7);
    }

    #[test]
    fn cell_kits_match_the_source_solver_rules() {
        let mut tree = Tree::new(IVec3::new(0, 12, 0), 0);
        let mut counts = [0_u8; 8];

        tree.form = TreeForm::Deciduous;
        counts[5] = 1;
        assert_eq!(tree.solve_leaf_cell(&counts), 4);
        counts = [0; 8];
        counts[4] = 2;
        assert_eq!(tree.solve_leaf_cell(&counts), 3);
        assert_eq!(tree.leaf_cell_value(4, DOWN), 4);

        tree.form = TreeForm::Conifer;
        counts = [0; 8];
        counts[4] = 1;
        assert_eq!(tree.solve_leaf_cell(&counts), 3);
        assert_eq!(tree.leaf_cell_value(4, UP), 4);
        assert_eq!(tree.leaf_cell_value(4, DOWN), 0);

        tree.form = TreeForm::Acacia;
        counts = [0; 8];
        counts[4] = 1;
        assert_eq!(tree.solve_leaf_cell(&counts), 2);
        assert_eq!(tree.leaf_cell_value(4, UP), 3);
        assert_eq!(tree.leaf_cell_value(4, DOWN), 0);
    }

    #[test]
    fn eco_machina_hpd_uses_dynamic_trees_radius_directly() {
        let root = IVec3::new(0, 12, 0);
        let mut tree = Tree::new(root, 0);
        tree.branches.clear();
        tree.leaves.clear();

        let branch = root + IVec3::Y;
        tree.branches.insert(branch, 3);
        tree.branches.insert(branch + IVec3::Y, 7);
        let segments = tree.eco_machina_segments();

        // Two individual wood-block spines. Their half-widths are the actual
        // DT radii 3/16 and 7/16, not a synthetic chain-length taper.
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].style[1], SEGMENT_WOOD_SPINE);
        assert_eq!(segments[1].style[1], SEGMENT_WOOD_SPINE);
        assert_eq!(segments[0].style[0], tree.form.render_id());
        assert_eq!(segments[0].start_radius, [0.5, 13.0, 0.5, 3.0 / 16.0]);
        assert_eq!(segments[0].end_radius, [0.5, 14.0, 0.5, 3.0 / 16.0]);
        assert_eq!(segments[1].start_radius, [0.5, 14.0, 0.5, 7.0 / 16.0]);
        assert_eq!(segments[1].end_radius, [0.5, 15.0, 0.5, 7.0 / 16.0]);
    }

    #[test]
    fn eco_machina_hpd_connects_only_secondary_chains_from_spine_in() {
        let root = IVec3::new(0, 12, 0);
        let mut tree = Tree::new(root, 0);
        tree.branches.clear();
        tree.leaves.clear();
        let trunk = root + IVec3::Y;
        tree.branches.insert(trunk, 8);
        tree.branches.insert(trunk + IVec3::Y, 6);
        tree.branches.insert(trunk + IVec3::Y * 2, 4);
        tree.branches.insert(trunk + IVec3::X, 5);

        let mut nodes = tree.build_hpd_nodes();
        Tree::compute_subtree_weights(&mut nodes, 0);
        let mut chain_count = 1;
        Tree::assign_hpd_chains(&mut nodes, 0, 0, 0, 0, &mut chain_count);
        let side = nodes
            .iter()
            .position(|node| node.position == trunk + IVec3::X)
            .expect("secondary Dynamic Trees branch is in the visualizer graph");
        assert_eq!(nodes[side].chain_depth, 1);
        assert_eq!(nodes[side].position_in_chain, 0);
        assert_eq!(nodes[side].dt_radius, 5);

        let segments = tree.eco_machina_segments();
        let connector = segments
            .iter()
            .find(|segment| segment.style[1] == SEGMENT_WOOD_CONNECTOR)
            .expect("one non-primary child gets the visualizer connector");
        assert_eq!(connector.start_radius, [0.5, 13.0, 0.5, 5.0 / 16.0]);
        assert_eq!(connector.end_radius, [1.0, 13.5, 0.5, 5.0 / 16.0]);
    }

    #[test]
    fn non_twig_growth_uses_dynamic_leaves_branch_out() {
        let root = IVec3::new(0, 12, 0);
        let mut tree = Tree::new(root, 0);
        tree.branches.clear();
        tree.leaves.clear();
        let source = root + IVec3::Y;
        let target = source + IVec3::Y;
        tree.branches.insert(source, 2);
        let mut signal = GrowSignal::new(root, 12.0);

        tree.grow_into_air(target, 2, &mut signal, &GrowthEnvironment::default());

        assert!(signal.success);
        assert_eq!(signal.radius, 2.0);
        assert_eq!(tree.branches.get(&target), Some(&1));
        assert!(!tree.leaves.is_empty());
    }

    #[test]
    fn entering_dynamic_leaves_consumes_a_second_signal_step() {
        let root = IVec3::new(0, 12, 0);
        let mut tree = Tree::new(root, 0);
        tree.branches.clear();
        tree.leaves.clear();
        let branch = root + IVec3::Y;
        let leaf = branch + IVec3::Y;
        tree.branches.insert(branch, 2);
        tree.leaves.insert(leaf, 4);
        let environment = GrowthEnvironment::default();

        let mut exhausted = GrowSignal::new(root, 2.0);
        tree.grow_branch(branch, &mut exhausted, 0, &environment);
        assert!(!exhausted.success);
        assert_eq!(tree.leaves.get(&leaf), Some(&4));
        assert!(!tree.branches.contains_key(&leaf));

        let mut viable = GrowSignal::new(root, 3.0);
        tree.grow_branch(branch, &mut viable, 0, &environment);
        assert!(viable.success);
        assert_eq!(tree.branches.get(&leaf), Some(&1));
        assert!(viable.radius > 2.0);
    }

    #[test]
    fn fractional_growth_rate_is_a_species_grow_probability() {
        let root = IVec3::new(0, 12, 0);
        let mut tree = Tree::new(root, 0);
        tree.form = TreeForm::Deciduous;
        tree.branches.clear();
        tree.leaves.clear();
        tree.random_state = 0x1357_9bdf;

        let original_state = tree.random_state;
        let expected = tree.growth_rate() > tree.random();
        tree.random_state = original_state;

        assert_eq!(tree.grow_in(&GrowthEnvironment::default()), expected);
        assert_eq!(tree.branches.contains_key(&(root + IVec3::Y)), expected);
    }

    #[test]
    fn growth_environment_blocks_terrain_and_foreign_tree_parts() {
        let root = IVec3::new(0, 12, 0);
        let mut tree = Tree::new(root, 0);
        tree.branches.clear();
        tree.leaves.clear();
        let target = root + IVec3::new(0, 2, 0);
        let mut environment = GrowthEnvironment::default();

        environment.terrain.insert(target);
        assert_eq!(tree.try_place_leaf_in(target, Some(4), &environment), 0);

        environment.terrain.clear();
        environment.tree_part_owner.insert(target, root + IVec3::X);
        assert_eq!(tree.try_place_leaf_in(target, Some(4), &environment), 0);
    }

    #[test]
    fn new_leaves_need_skylight_and_respect_source_smother_limits() {
        let root = IVec3::new(0, 12, 0);
        let mut tree = Tree::new(root, 0);
        tree.branches.clear();
        tree.leaves.clear();
        let leaf = root + IVec3::new(0, 2, 0);

        let mut cave = GrowthEnvironment::default();
        cave.terrain.insert(leaf + IVec3::Y);
        assert_eq!(tree.try_place_leaf_in(leaf, Some(4), &cave), 0);

        tree.form = TreeForm::Conifer;
        let mut smothered = GrowthEnvironment::default();
        for height in 1..=3 {
            let above = leaf + IVec3::Y * height;
            smothered.tree_part_owner.insert(above, root);
            smothered
                .tree_parts
                .insert(above, EnvironmentTreePart::Leaf);
        }
        assert_eq!(tree.try_place_leaf_in(leaf, Some(4), &smothered), 0);
    }

    #[test]
    fn unsupported_twigs_follow_dynamic_trees_rot_and_leaf_recovery() {
        let root = IVec3::new(0, 12, 0);
        let twig = root + IVec3::Y;
        let environment = GrowthEnvironment::default();

        let mut fertile = Tree::new(root, 0);
        fertile.branches.clear();
        fertile.leaves.clear();
        fertile.branches.insert(twig, 1);
        fertile.fertility = 15;
        assert!(fertile.handle_rot_in(&environment));
        assert!(fertile.branches.contains_key(&twig));
        assert!(!fertile.leaves.is_empty());

        let mut depleted = Tree::new(root, 0);
        depleted.branches.clear();
        depleted.leaves.clear();
        depleted.branches.insert(twig, 1);
        depleted.fertility = 0;
        assert!(depleted.handle_rot_in(&environment));
        assert!(depleted.branches.is_empty());
    }

    #[test]
    fn gpu_tree_segment_layout_matches_wgsl_storage_stride() {
        assert_eq!(std::mem::size_of::<GpuTreeSegment>(), 48);
        assert_eq!(std::mem::offset_of!(GpuTreeSegment, end_radius), 16);
        assert_eq!(std::mem::offset_of!(GpuTreeSegment, style), 32);
        assert_eq!(std::mem::size_of::<GpuTreeBvhNode>(), 48);
        assert_eq!(std::mem::offset_of!(GpuTreeBvhNode, maximum), 16);
        assert_eq!(std::mem::offset_of!(GpuTreeBvhNode, data), 32);
    }

    #[test]
    fn tree_blas_reorders_only_segments_and_keeps_their_bounds() {
        let make_segment = |start: [f32; 3], end: [f32; 3]| GpuTreeSegment {
            start_radius: [start[0], start[1], start[2], 0.25],
            end_radius: [end[0], end[1], end[2], 0.25],
            style: [0, SEGMENT_WOOD_SPINE, 0, 0],
        };
        let mut primitives = [
            segment_primitive(make_segment([4.0, 1.0, 0.0], [5.0, 1.0, 0.0])),
            segment_primitive(make_segment([-3.0, 2.0, 0.0], [-2.0, 2.0, 0.0])),
            segment_primitive(make_segment([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])),
            segment_primitive(make_segment([8.0, 3.0, 0.0], [9.0, 3.0, 0.0])),
            segment_primitive(make_segment([-8.0, 4.0, 0.0], [-7.0, 4.0, 0.0])),
        ];
        let mut output = vec![GpuTreeSegment::zeroed(); primitives.len()];
        let mut nodes = vec![GpuTreeBvhNode::zeroed(); 8];
        let mut segment_cursor = 0;
        let mut node_cursor = 0;
        let root = build_tree_blas(
            &mut primitives,
            &mut output,
            &mut segment_cursor,
            &mut nodes,
            &mut node_cursor,
        );
        assert_eq!(segment_cursor, output.len());
        assert_eq!(node_cursor, 3);
        assert_eq!(nodes[root].data[2], 0);
        assert!(nodes[root].minimum[0] <= -8.25);
        assert!(nodes[root].maximum[0] >= 9.25);
    }
}
