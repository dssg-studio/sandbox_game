//! A small ray-traced voxel sandbox.  Geometry is evaluated in the WGSL fragment
//! shader: voxel cells use DDA ray traversal. Dynamic Trees data is currently
//! displayed through its native branch/leaf cells.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::Instant,
};

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec3};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

const CHUNK_SIZE: i32 = 16;
const WORLD_HEIGHT: i32 = 48;
const DETAIL_RADIUS: i32 = 4;
/// Distant Horizons-style horizon distance, measured in chunks.
const LOD_RADIUS: i32 = 256;
const DETAIL_DIAMETER: i32 = DETAIL_RADIUS * 2 + 1;
const LOD_GRID_SIZE: i32 = 33;
const LOD_LEVEL_FACTORS: [i32; 5] = [1, 2, 4, 8, 16];
const LOD_LEVEL_COUNT: usize = LOD_LEVEL_FACTORS.len();
const LOD_SAMPLE_COUNT: usize = LOD_LEVEL_COUNT * (LOD_GRID_SIZE * LOD_GRID_SIZE) as usize;
const MAX_TREES: usize = 192;
const MAX_BRANCHES_PER_TREE: usize = 128;
const MAX_GPU_BRANCHES: usize = MAX_TREES * MAX_BRANCHES_PER_TREE;
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
const TREE_LEAVES: u32 = 6;
const ROOTY_SOIL: u32 = 9;

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
}

struct Chunk {
    // x + 16 * (z + 16 * y)
    blocks: Vec<Block>,
    trees: Vec<Tree>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTree {
    // Sphere used for a cheap per-tree broad phase.
    bounds: [f32; 4],
    // Branch offset/count, leaf offset/count.
    layout: [f32; 4],
    // Stable tint seed, soil fertility, age in pulses, unused.
    appearance: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuBranch {
    // Dynamic Trees' central cube: world-space centre and radius in blocks.
    centre_radius: [f32; 4],
    // Side radii in Direction order: down/up/north/south/west/east. Two
    // padding values preserve the 16-byte storage-array stride required by
    // WGSL. A sleeve uses each radius to extend the core to that block face.
    connections: [u32; 8],
}

// Dynamic Trees stores a tree as a sparse collection of branch and leaf
// blocks. The renderer consumes the same radius and neighbour data as the
// original baked branch model, but ray-intersects those cuboids directly.
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
        if !tree.generate_jocode_worldgen(seed) {
            // Kept only as a defensive fallback for a malformed local data
            // registry; the bundled species all have JoCode entries.
            tree.grow_pulse();
        }
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

    /// Equivalent to `BranchBlock#getConnectionData`: the core reads the
    /// radius from each neighbouring branch, clamps it to its own radius and
    /// connects twigs to dynamic leaves. Rooty soil has a connection radius of
    /// eight, which becomes the root branch's own radius after clamping.
    fn connection_radii(&self, position: IVec3, core_radius: u8) -> [u32; 8] {
        let mut connections = [0_u32; 8];
        for (direction, offset) in DIRECTIONS.iter().enumerate() {
            let neighbour = position + *offset;
            let radius = if let Some(radius) = self.branches.get(&neighbour) {
                *radius
            } else if neighbour == self.root {
                8 // SoilBlock#getRadiusForConnection
            } else if core_radius == 1 && self.leaves.contains_key(&neighbour) {
                1 // LeavesProperties#getRadiusForConnection for a twig
            } else {
                0
            };
            connections[direction] = u32::from(radius.min(core_radius));
        }
        connections
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
    // LOD grid side, LOD count, unused, camera aspect ratio
    lod_world: [f32; 4],
    // elapsed world time, tree count, detailed world height, unused
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
            lod_world: [0.0; 4],
            simulation: [0.0; 4],
        }
    }
}

struct World {
    detailed: HashMap<ChunkPos, Chunk>,
    center: Option<ChunkPos>,
    detail_origin: ChunkPos,
    detailed_blocks: Vec<Block>,
    lod_samples: Vec<u32>,
    lod_levels: [GpuLodLevel; LOD_LEVEL_COUNT],
    gpu_trees: Vec<GpuTree>,
    gpu_branches: Vec<GpuBranch>,
    tree_count: usize,
    growth_time: f32,
    last_tree_tick: f32,
}

impl World {
    fn new() -> Self {
        Self {
            detailed: HashMap::new(),
            center: None,
            detail_origin: ChunkPos { x: 0, z: 0 },
            detailed_blocks: vec![
                AIR;
                (DETAIL_DIAMETER * CHUNK_SIZE * DETAIL_DIAMETER * CHUNK_SIZE * WORLD_HEIGHT)
                    as usize
            ],
            lod_samples: vec![0; LOD_SAMPLE_COUNT],
            lod_levels: [GpuLodLevel::zeroed(); LOD_LEVEL_COUNT],
            gpu_trees: vec![GpuTree::zeroed(); MAX_TREES],
            gpu_branches: vec![GpuBranch::zeroed(); MAX_GPU_BRANCHES],
            tree_count: 0,
            growth_time: 0.0,
            last_tree_tick: 0.0,
        }
    }

    fn growth_environment(&self) -> GrowthEnvironment {
        let mut environment = GrowthEnvironment::default();
        for chunk_z in 0..DETAIL_DIAMETER {
            for chunk_x in 0..DETAIL_DIAMETER {
                let position = ChunkPos {
                    x: self.detail_origin.x + chunk_x,
                    z: self.detail_origin.z + chunk_z,
                };
                let chunk = self
                    .detailed
                    .get(&position)
                    .expect("the detail window was streamed before tree growth");

                for y in 0..WORLD_HEIGHT {
                    for z in 0..CHUNK_SIZE {
                        for x in 0..CHUNK_SIZE {
                            let index = (x + CHUNK_SIZE * (z + CHUNK_SIZE * y)) as usize;
                            if chunk.blocks[index] != AIR {
                                environment.terrain.insert(IVec3::new(
                                    position.x * CHUNK_SIZE + x,
                                    y,
                                    position.z * CHUNK_SIZE + z,
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
        }
        environment
    }

    /// Streams a square of full chunks and a much larger low-detail ring.
    /// Returns true only when a GPU upload of the detailed terrain is needed.
    fn stream_around(&mut self, position: Vec3) -> bool {
        let new_center = ChunkPos::from_world(position);
        if self.center == Some(new_center) {
            return false;
        }
        self.center = Some(new_center);
        self.detail_origin = ChunkPos {
            x: new_center.x - DETAIL_RADIUS,
            z: new_center.z - DETAIL_RADIUS,
        };

        for z in -DETAIL_RADIUS..=DETAIL_RADIUS {
            for x in -DETAIL_RADIUS..=DETAIL_RADIUS {
                let position = ChunkPos {
                    x: new_center.x + x,
                    z: new_center.z + z,
                };
                self.detailed
                    .entry(position)
                    .or_insert_with(|| generate_chunk(position));
            }
        }

        // Keep one safety ring so walking back and forth does not constantly
        // regenerate chunks, while still proving that chunks genuinely unload.
        self.detailed
            .retain(|position, _| position.distance(new_center) <= LOD_RADIUS + 1);

        self.rebuild_detailed_buffer();
        self.rebuild_lod_buffer();
        self.rebuild_tree_buffer();
        true
    }

    fn advance(&mut self, seconds: f32) -> bool {
        self.growth_time += seconds;
        // Keep the host tick decoupled from rendering. Each tick invokes
        // Species#grow once; that method applies the species' probabilistic
        // growth rate and may send one or more root-to-tip GrowSignals.
        let ticks =
            ((self.growth_time - self.last_tree_tick) / TREE_GROWTH_TICK_SECONDS).floor() as u32;
        if ticks > 0 {
            self.last_tree_tick += ticks as f32 * TREE_GROWTH_TICK_SECONDS;
            let environment = self.growth_environment();
            let mut changed = false;
            for chunk_z in 0..DETAIL_DIAMETER {
                for chunk_x in 0..DETAIL_DIAMETER {
                    let position = ChunkPos {
                        x: self.detail_origin.x + chunk_x,
                        z: self.detail_origin.z + chunk_z,
                    };
                    let chunk = self
                        .detailed
                        .get_mut(&position)
                        .expect("the detail window was streamed before trees advance");
                    chunk.trees.retain_mut(|tree| {
                        for _ in 0..ticks {
                            changed |= tree.update_in(&environment);
                        }
                        // Species#update destroys the rooty soil when the
                        // last branch rots. Removing this standalone tree has
                        // the same visible result: its terrain cell reverts
                        // to the underlying soil on the next rebuild.
                        !tree.branches.is_empty()
                    });
                }
            }
            if changed {
                self.rebuild_detailed_buffer();
                self.rebuild_tree_buffer();
            }
            changed
        } else {
            false
        }
    }

    fn rebuild_detailed_buffer(&mut self) {
        let width = (DETAIL_DIAMETER * CHUNK_SIZE) as usize;
        let depth = width;
        self.detailed_blocks.clear();
        self.detailed_blocks
            .reserve(width * depth * WORLD_HEIGHT as usize);

        for y in 0..WORLD_HEIGHT as usize {
            for chunk_z in 0..DETAIL_DIAMETER {
                for local_z in 0..CHUNK_SIZE as usize {
                    for chunk_x in 0..DETAIL_DIAMETER {
                        let position = ChunkPos {
                            x: self.detail_origin.x + chunk_x,
                            z: self.detail_origin.z + chunk_z,
                        };
                        let chunk = self
                            .detailed
                            .get(&position)
                            .expect("the detail window was streamed before being packed");
                        let offset = 16 * (local_z + 16 * y);
                        self.detailed_blocks
                            .extend_from_slice(&chunk.blocks[offset..offset + 16]);
                    }
                }
            }
        }
        debug_assert_eq!(
            self.detailed_blocks.len(),
            width * depth * WORLD_HEIGHT as usize
        );
        self.paint_dynamic_trees();
    }

    /// Projects Dynamic Trees' dynamic leaf blocks into the streamed DDA
    /// volume. Branches use the exact core-and-sleeve model in the separate
    /// branch buffer, while leaves retain their normal full block shape.
    fn paint_dynamic_trees(&mut self) {
        let mut leaf_cells = Vec::new();
        let mut rooty_soil_cells = Vec::new();
        for chunk_z in 0..DETAIL_DIAMETER {
            for chunk_x in 0..DETAIL_DIAMETER {
                let position = ChunkPos {
                    x: self.detail_origin.x + chunk_x,
                    z: self.detail_origin.z + chunk_z,
                };
                let chunk = self
                    .detailed
                    .get(&position)
                    .expect("the detail window was streamed before trees were packed");
                for tree in &chunk.trees {
                    leaf_cells.extend(tree.leaves.keys().copied());
                    rooty_soil_cells.push(tree.root);
                }
            }
        }

        let width = DETAIL_DIAMETER * CHUNK_SIZE;
        let origin_x = self.detail_origin.x * CHUNK_SIZE;
        let origin_z = self.detail_origin.z * CHUNK_SIZE;
        let cell_index = |position: IVec3| -> Option<usize> {
            let x = position.x - origin_x;
            let z = position.z - origin_z;
            if !(0..width).contains(&x)
                || !(0..width).contains(&z)
                || !(0..WORLD_HEIGHT).contains(&position.y)
            {
                return None;
            }
            Some((x + width * (z + width * position.y)) as usize)
        };

        for position in rooty_soil_cells {
            if let Some(index) = cell_index(position) {
                let height = (self.detailed_blocks[index] >> 8) & 255;
                self.detailed_blocks[index] = block(ROOTY_SOIL, height);
            }
        }
        for position in leaf_cells {
            if let Some(index) = cell_index(position)
                && self.detailed_blocks[index] == AIR
            {
                self.detailed_blocks[index] = block(TREE_LEAVES, 16);
            }
        }
    }

    fn active_tree_count(&self) -> usize {
        (0..DETAIL_DIAMETER)
            .flat_map(|chunk_z| (0..DETAIL_DIAMETER).map(move |chunk_x| (chunk_x, chunk_z)))
            .map(|(chunk_x, chunk_z)| ChunkPos {
                x: self.detail_origin.x + chunk_x,
                z: self.detail_origin.z + chunk_z,
            })
            .filter_map(|position| self.detailed.get(&position))
            .map(|chunk| chunk.trees.len())
            .sum()
    }

    fn rebuild_lod_buffer(&mut self) {
        let center = self
            .center
            .expect("LOD data is built after the stream center is set");
        let samples_per_level = (LOD_GRID_SIZE * LOD_GRID_SIZE) as usize;
        for (level, factor) in LOD_LEVEL_FACTORS.iter().copied().enumerate() {
            let half_width = LOD_GRID_SIZE / 2;
            let origin_chunk = ChunkPos {
                x: center.x - half_width * factor,
                z: center.z - half_width * factor,
            };
            let cell_size = CHUNK_SIZE * factor;
            let sample_offset = level * samples_per_level;
            self.lod_levels[level] = GpuLodLevel {
                data: [
                    (origin_chunk.x * CHUNK_SIZE) as f32,
                    (origin_chunk.z * CHUNK_SIZE) as f32,
                    cell_size as f32,
                    sample_offset as f32,
                ],
            };
            for z in 0..LOD_GRID_SIZE {
                for x in 0..LOD_GRID_SIZE {
                    let world_x = origin_chunk.x * CHUNK_SIZE + x * cell_size + cell_size / 2;
                    let world_z = origin_chunk.z * CHUNK_SIZE + z * cell_size + cell_size / 2;
                    self.lod_samples[sample_offset + (x + LOD_GRID_SIZE * z) as usize] =
                        u32::from(terrain_height_units(world_x, world_z));
                }
            }
        }
    }

    fn rebuild_tree_buffer(&mut self) {
        self.gpu_trees.fill(GpuTree::zeroed());
        self.gpu_branches.fill(GpuBranch::zeroed());
        let mut tree_index = 0;
        let mut branch_index = 0;

        'trees: for chunk_z in 0..DETAIL_DIAMETER {
            for chunk_x in 0..DETAIL_DIAMETER {
                let position = ChunkPos {
                    x: self.detail_origin.x + chunk_x,
                    z: self.detail_origin.z + chunk_z,
                };
                let chunk = self
                    .detailed
                    .get(&position)
                    .expect("the detail window was streamed before tree shapes were packed");
                for tree in &chunk.trees {
                    if tree_index >= MAX_TREES {
                        break 'trees;
                    }
                    let start = branch_index;
                    let mut minimum = Vec3::splat(f32::INFINITY);
                    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
                    for (&branch_position, &radius) in &tree.branches {
                        if branch_index >= MAX_GPU_BRANCHES {
                            break 'trees;
                        }
                        let centre = branch_position.as_vec3() + Vec3::splat(0.5);
                        let branch_radius = f32::from(radius) / 16.0;
                        let render_extent = branch_radius.max(1.0);
                        minimum =
                            minimum.min(centre - Vec3::new(render_extent, 1.0, render_extent));
                        maximum =
                            maximum.max(centre + Vec3::new(render_extent, 1.0, render_extent));
                        self.gpu_branches[branch_index] = GpuBranch {
                            centre_radius: [centre.x, centre.y, centre.z, branch_radius],
                            connections: tree.connection_radii(branch_position, radius),
                        };
                        branch_index += 1;
                    }
                    let count = branch_index - start;
                    if count == 0 {
                        continue;
                    }
                    let centre = (minimum + maximum) * 0.5;
                    self.gpu_trees[tree_index] = GpuTree {
                        bounds: [centre.x, centre.y, centre.z, (maximum - centre).length()],
                        layout: [start as f32, count as f32, 0.0, 0.0],
                        appearance: [0.0; 4],
                    };
                    tree_index += 1;
                }
            }
        }
        self.tree_count = tree_index;
    }
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
    // Each chunk can grow one or two independent trees. Their branch cells
    // retain the Dynamic Trees core/sleeve shape instead of becoming cubes.
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
    Chunk { blocks, trees }
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
    branch_buffer: wgpu::Buffer,
    camera: Camera,
    world: World,
    world_time: f32,
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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ray voxel device"),
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
        world.stream_around(camera.position);

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
            "distant horizons clipmap heights",
            &world.lod_samples,
        );
        let lod_info_buffer = storage_buffer(
            &device,
            "distant horizons clipmap metadata",
            &world.lod_levels,
        );
        let tree_buffer = storage_buffer(&device, "Dynamic Trees bounds", &world.gpu_trees);
        let branch_buffer =
            storage_buffer(&device, "dynamic tree branch graph", &world.gpu_branches);

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ray world bind group layout"),
            entries: &[
                buffer_layout_entry(0, wgpu::BufferBindingType::Uniform),
                buffer_layout_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_layout_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_layout_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_layout_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                buffer_layout_entry(5, wgpu::BufferBindingType::Storage { read_only: true }),
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
                    resource: branch_buffer.as_entire_binding(),
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
            branch_buffer,
            camera,
            world,
            world_time: 18.0,
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
        self.camera.move_with_keys(keys, seconds);
        self.world_time += seconds;
        let terrain_changed = self.world.stream_around(self.camera.position);
        let trees_changed = self.world.advance(seconds);
        if terrain_changed || trees_changed {
            self.queue.write_buffer(
                &self.detail_buffer,
                0,
                bytemuck::cast_slice(&self.world.detailed_blocks),
            );
            if terrain_changed {
                self.queue.write_buffer(
                    &self.lod_buffer,
                    0,
                    bytemuck::cast_slice(&self.world.lod_samples),
                );
                self.queue.write_buffer(
                    &self.lod_info_buffer,
                    0,
                    bytemuck::cast_slice(&self.world.lod_levels),
                );
            }
        }
        if terrain_changed || trees_changed {
            self.queue.write_buffer(
                &self.tree_buffer,
                0,
                bytemuck::cast_slice(&self.world.gpu_trees),
            );
            self.queue.write_buffer(
                &self.branch_buffer,
                0,
                bytemuck::cast_slice(&self.world.gpu_branches),
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
        uniforms.lod_world = [
            LOD_GRID_SIZE as f32,
            LOD_LEVEL_COUNT as f32,
            0.0,
            self.size.width as f32 / self.size.height.max(1) as f32,
        ];
        uniforms.simulation = [
            self.world_time,
            self.world.tree_count as f32,
            WORLD_HEIGHT as f32,
            0.0,
        ];
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
    }

    fn render(&mut self) {
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
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ray world encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("full-screen ray tracing pass"),
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
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
        self.queue.present(output);
    }

    fn status(&self) -> String {
        format!(
            "RayVoxel — {} detailed chunks · 256-chunk LOD horizon · {} Dynamic Trees (block mode)",
            self.world.detailed.len(),
            self.world.active_tree_count()
        )
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
    let event_loop = EventLoop::new()?;
    event_loop.run_app(&mut App::new())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(world.stream_around(Vec3::new(0.0, 20.0, 0.0)));
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
    fn branch_connections_clamp_and_rooty_soil_connects_like_dynamic_trees() {
        let root = IVec3::new(0, 12, 0);
        let mut tree = Tree::new(root, 0);
        tree.branches.clear();
        tree.leaves.clear();

        let branch = root + IVec3::Y;
        tree.branches.insert(branch, 3);
        tree.branches.insert(branch + IVec3::Y, 7);
        tree.leaves.insert(branch + IVec3::X, 4);
        assert_eq!(tree.connection_radii(branch, 3), [3, 3, 0, 0, 0, 0, 0, 0]);
        assert_eq!(tree.connection_radii(branch, 1), [1, 1, 0, 0, 0, 1, 0, 0]);
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
    fn gpu_branch_layout_matches_wgsl_branch_cell_stride() {
        assert_eq!(std::mem::size_of::<GpuBranch>(), 48);
        assert_eq!(std::mem::offset_of!(GpuBranch, connections), 16);
    }
}
