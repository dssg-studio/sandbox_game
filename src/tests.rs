use super::*;

#[test]
fn raytracing_shader_is_valid_wgsl() {
    let module = wgpu::naga::front::wgsl::parse_str(include_str!("shader.wgsl"))
        .expect("ray-tracing shader must parse as WGSL");
    let mut validator = wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("ray-tracing shader must pass WGSL validation");
}

#[test]
fn direct_sunlight_uses_atmospheric_spectral_transmittance() {
    let shader = include_str!("shader.wgsl");
    assert!(shader.contains("fn direct_sun_illuminance"));
    assert!(shader.contains("direct_sun_illuminance(point) * direct_intensity"));
    assert!(shader.contains("ATMOSPHERE_SOLAR_SPECTRUM"));
    assert!(shader.contains("ATMOSPHERE_OZONE_ABSORPTION"));
    assert!(shader.contains("DISPLAY_EXPOSURE"));
}

#[test]
fn lod_uses_world_stable_dithered_handoff_for_exact_tree_geometry() {
    let shader = include_str!("shader.wgsl");
    assert!(shader.contains("fn trace_lod_for_detail_fade"));
    assert!(shader.contains("fn dh_detail_fade_noise"));
    assert!(shader.contains("dh_detail_fade_amount(exact.t)"));
    // AO and direct sun shadow rays deliberately retain the disjoint data
    // sources, avoiding false self-occlusion from a coarse LOD cell.
    assert!(shader.contains("return trace_lod_internal(ro, rd, max_distance, false)"));
}

#[test]
fn atmosphere_precomputes_multiple_scattering_without_per_pixel_extra_rays() {
    let shader = include_str!("shader.wgsl");
    assert!(shader.contains("cs_atmosphere_multiple_scattering"));
    assert!(shader.contains("atmosphere_multiple_scattering_source"));
    assert!(shader.contains("sample_atmosphere_multiple_scattering"));
}

#[test]
fn game_clock_matches_the_sun_cycle() {
    assert_eq!(game_time_label(0.0), "06:00");
    assert_eq!(game_time_label(WORLD_DAY_SECONDS * 0.25), "12:00");
    assert_eq!(game_time_label(WORLD_DAY_SECONDS * 0.5), "18:00");
    assert_eq!(game_time_label(WORLD_DAY_SECONDS * 0.75), "00:00");
    assert_eq!(game_time_label(WORLD_DAY_SECONDS), "06:00");
}

#[test]
fn selection_ray_keeps_sixteenth_slab_bounds_exact() {
    let minimum = Vec3::new(3.0, 9.0, 4.0);
    let maximum = minimum + Vec3::new(1.0, 1.0 / 16.0, 1.0);
    let (entry, exit) = ray_aabb_interval_cpu(
        Vec3::new(3.5, 9.0 + 1.0 / 32.0, 1.0),
        Vec3::Z,
        minimum,
        maximum,
    )
    .expect("ray through a 1/16 slab must select it");
    assert!((entry - 3.0).abs() < 1.0e-5);
    assert!((exit - 4.0).abs() < 1.0e-5);
}

#[test]
fn mouse_camera_look_rotates_and_clamps_pitch() {
    let mut camera = Camera::new();
    let initial_yaw = camera.yaw;
    let initial_pitch = camera.pitch;
    camera.rotate_by_mouse(Vec2::new(100.0, -50.0));
    assert!(camera.yaw > initial_yaw);
    assert!(camera.pitch > initial_pitch);

    camera.rotate_by_mouse(Vec2::new(0.0, -1_000_000.0));
    assert_eq!(camera.pitch, 1.45);
    camera.rotate_by_mouse(Vec2::new(0.0, 1_000_000.0));
    assert_eq!(camera.pitch, -1.45);
}

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
fn surface_placement_uses_the_exact_sixteenth_slab_top() {
    let support = SupportSurface::from_packed(IVec3::new(4, 11, -3), block(GRASS, 5))
        .expect("a grass slab exposes an upward support surface");
    let placement = resolve_placement_on_support(support, PlacementPolicy::SurfaceAligned)
        .expect("ordinary surface objects can stand on a grass slab");

    match placement {
        ResolvedPlacement::SurfaceAligned { base_y, support } => {
            assert_eq!(support.top_y_sixteenths, 181);
            assert!((base_y - 11.3125).abs() < f32::EPSILON);
        }
        ResolvedPlacement::DynamicTreeRoot { .. } => panic!("surface policy returned tree root"),
    }
}

#[test]
fn dynamic_tree_root_normalizes_a_slab_to_full_rooty_soil() {
    let mut blocks = vec![AIR; CHUNK_BLOCK_COUNT];
    let local_root = IVec3::new(5, 13, 9);
    let world_root = IVec3::new(-27, 13, 41);
    let root_index = chunk_block_index(local_root.x, local_root.y, local_root.z);
    blocks[root_index] = block(GRASS, 7);

    let root = prepare_dynamic_tree_root(&mut blocks, local_root, world_root)
        .expect("a tree accepts fertile partial soil by converting it to rooty soil");

    assert_eq!(root, world_root);
    assert_eq!(block_material(blocks[root_index]), ROOTY_SOIL);
    assert_eq!(block_height_sixteenths(blocks[root_index]), 16);
    assert_eq!(
        blocks[chunk_block_index(local_root.x, local_root.y + 1, local_root.z)],
        AIR
    );
}

#[test]
fn dynamic_tree_root_rejects_non_soil_or_obstructed_support() {
    let local_root = IVec3::new(3, 12, 7);
    let world_root = IVec3::new(3, 12, 7);
    let root_index = chunk_block_index(local_root.x, local_root.y, local_root.z);

    let mut water = vec![AIR; CHUNK_BLOCK_COUNT];
    water[root_index] = block(WATER, 16);
    assert_eq!(
        prepare_dynamic_tree_root(&mut water, local_root, world_root),
        None
    );

    let mut obstructed = vec![AIR; CHUNK_BLOCK_COUNT];
    obstructed[root_index] = block(DIRT, 16);
    obstructed[chunk_block_index(local_root.x, local_root.y + 1, local_root.z)] = block(STONE, 16);
    assert_eq!(
        prepare_dynamic_tree_root(&mut obstructed, local_root, world_root),
        None
    );
}

#[test]
fn generated_dynamic_trees_have_full_rooty_soil_in_source_chunks() {
    let mut tree_count = 0;
    for chunk_z in -3..=3 {
        for chunk_x in -3..=3 {
            let position = ChunkPos {
                x: chunk_x,
                z: chunk_z,
            };
            let chunk = generate_chunk(position);
            for tree in &chunk.trees {
                tree_count += 1;
                let local_x = tree.root.x.rem_euclid(CHUNK_SIZE);
                let local_z = tree.root.z.rem_euclid(CHUNK_SIZE);
                let packed = chunk.blocks[chunk_block_index(local_x, tree.root.y, local_z)];
                assert_eq!(block_material(packed), ROOTY_SOIL);
                assert_eq!(block_height_sixteenths(packed), 16);
                assert_eq!(
                    tree.root.y,
                    i32::from(terrain_height_units(tree.root.x, tree.root.z) / 16)
                );
            }
        }
    }
    assert!(
        tree_count > 0,
        "the sample area must contain generated trees"
    );
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
    assert_eq!(world.lod_samples.len(), LOD_GPU_SAMPLE_COUNT);
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
fn detailed_occupancy_matches_non_air_blocks_at_every_level() {
    let mut world = World::new();
    world.detailed_blocks.fill(AIR);
    let slot = 5;
    let local = IVec3::new(7, 20, 13);
    let local_index = local.x as usize
        + CHUNK_SIZE as usize * (local.z as usize + CHUNK_SIZE as usize * local.y as usize);
    let block_index = slot * CHUNK_BLOCK_COUNT + local_index;
    world.detailed_blocks[block_index] = block(STONE, 16);
    world.rebuild_occupancy_slot(slot);

    assert_ne!(
        world.detail_occupancy[block_index / 32] & (1 << (block_index % 32)),
        0
    );
    let brick_index = (local.x / DETAIL_BRICK_SIZE) as usize
        + (CHUNK_SIZE / DETAIL_BRICK_SIZE) as usize
            * ((local.z / DETAIL_BRICK_SIZE) as usize
                + (CHUNK_SIZE / DETAIL_BRICK_SIZE) as usize
                    * (local.y / DETAIL_BRICK_SIZE) as usize);
    let brick_bit = (DETAIL_BRICK_OCCUPANCY_OFFSET + slot * DETAIL_BRICK_OCCUPANCY_WORDS_PER_CHUNK)
        * 32
        + brick_index;
    assert_ne!(
        world.detail_occupancy[brick_bit / 32] & (1 << (brick_bit % 32)),
        0
    );
    let coarse_bit = DETAIL_COARSE_OCCUPANCY_OFFSET * 32
        + slot * (WORLD_HEIGHT / CHUNK_SIZE) as usize
        + (local.y / CHUNK_SIZE) as usize;
    assert_ne!(
        world.detail_occupancy[coarse_bit / 32] & (1 << (coarse_bit % 32)),
        0
    );
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
fn lod_full_data_keeps_surface_and_cliff_materials() {
    let section = generate_lod_data_source(LodSectionKey {
        detail: 0,
        x: 0,
        z: 0,
    });
    assert_eq!(section.columns.len(), LOD_SECTION_COLUMN_COUNT);
    let surface_slices = section
        .full_columns
        .iter()
        .flat_map(|column| column.iter().copied())
        .filter(|slice| slice.minimum == 0 && slice.maximum > 0)
        .collect::<Vec<_>>();
    assert!(!surface_slices.is_empty());
    // Tree roots can legitimately replace a terrain span at Y=0.  Require
    // that terrain material is still present rather than misclassifying roots
    // as invalid terrain data.
    assert!(surface_slices.iter().any(|slice| {
        matches!(slice.top_material, GRASS | STONE | WATER)
            && matches!(slice.side_material, DIRT | STONE | WATER)
    }));
}

#[test]
fn lod_full_data_keeps_dynamic_tree_wood_and_foliage_slices() {
    let source_chunk = (-8..=8)
        .flat_map(|z| (-8..=8).map(move |x| ChunkPos { x, z }))
        .find(|&position| tree_generation_candidate(position, 0).is_some())
        .expect("the deterministic worldgen area contains a tree");
    let section = generate_lod_data_source(LodSectionKey {
        detail: 0,
        x: (source_chunk.x * CHUNK_SIZE).div_euclid(LOD_SECTION_SIDE * LOD_LEVEL_FACTORS[0]),
        z: (source_chunk.z * CHUNK_SIZE).div_euclid(LOD_SECTION_SIDE * LOD_LEVEL_FACTORS[0]),
    });
    let slices = section
        .columns
        .iter()
        .flat_map(|column| column.slices)
        .filter_map(unpack_lod_render_slice)
        .collect::<Vec<_>>();
    assert!(
        slices
            .iter()
            .any(|slice| matches!(slice.top_material, 10..=12))
    );
    assert!(slices.iter().any(|slice| {
        matches!(
            slice.top_material,
            OAK_LEAVES | SPRUCE_LEAVES | ACACIA_LEAVES
        )
    }));
}

#[test]
fn lod_worldgen_produces_a_self_contained_source_at_every_requested_detail() {
    for detail in 0..LOD_LEVEL_COUNT as u8 {
        let section = generate_lod_data_source(LodSectionKey { detail, x: 0, z: 0 });
        assert_eq!(section.key.detail, detail);
        assert_eq!(section.columns.len(), LOD_SECTION_COLUMN_COUNT);
        assert!(section.full_columns.iter().any(|column| !column.is_empty()));
        assert!(section.columns.iter().all(|column| {
            column.slices[LOD_VERTICAL_SLICE_COUNTS[detail as usize]..]
                .iter()
                .all(|slice| *slice == 0)
        }));
    }
}

#[test]
fn lod_worldgen_queue_cancels_only_waiting_sources_outside_the_render_cut() {
    let queue = WorldGenerationQueue::new();
    let visible = LodSectionKey {
        detail: 2,
        x: 3,
        z: -1,
    };
    let obsolete = LodSectionKey {
        detail: 4,
        x: -9,
        z: 7,
    };
    queue.set_generation_target(ChunkPos { x: 0, z: 0 });
    queue.submit_retrieval_task(DataSourceRetrievalTask { key: visible });
    queue.submit_retrieval_task(DataSourceRetrievalTask { key: obsolete });
    queue.remove_retrieval_requests_not_in(&HashSet::from([visible]));
    let request = queue.take_next_task();
    assert_eq!(request.key, visible);
    queue.finish_task(visible);
}

#[test]
fn lod_worldgen_queue_uses_the_latest_generation_target() {
    let queue = WorldGenerationQueue::new();
    let old_view = LodSectionKey {
        detail: 0,
        x: 0,
        z: 0,
    };
    let new_view = LodSectionKey {
        detail: 0,
        x: 10,
        z: 0,
    };
    queue.set_generation_target(ChunkPos { x: 0, z: 0 });
    queue.submit_retrieval_task(DataSourceRetrievalTask { key: old_view });
    queue.submit_retrieval_task(DataSourceRetrievalTask { key: new_view });

    // DH changes its generation target separately from the waiting-task map.
    // The next worker must therefore serve the new camera side first.
    queue.set_generation_target(ChunkPos { x: 160, z: 0 });
    let request = queue.take_next_task();
    assert_eq!(request.key, new_view);
    queue.finish_task(new_view);
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
