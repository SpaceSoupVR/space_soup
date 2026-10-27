    use super::*;
    use crate::renderer::mesh::skin::{blend_joint_local, ClipBlendMode, GltfAnimationPose, IDLE_BLEND};

    fn workspace_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
    }

    #[test]
    fn ancestor_bake_handles_m4a1_wrapper_chain() {
        let path = workspace_root().join("game/models/m4a1.glb");
        let (doc, _buffers, _images) = gltf::import(&path).expect("m4a1.glb should load");

        let all_nodes: Vec<gltf::Node> = doc.nodes().collect();
        let mut parent_of_node: HashMap<usize, usize> = HashMap::new();
        for node in doc.nodes() {
            for child in node.children() {
                parent_of_node.insert(child.index(), node.index());
            }
        }

        let charging_handle = doc
            .nodes()
            .find(|n| n.name() == Some("m4a1 charging handle_10"))
            .expect("m4a1.glb should have a node named 'm4a1 charging handle_10'");

        let joint_of_node: HashMap<usize, usize> = HashMap::new();
        let (parent_joint, t2, r2, s2) = ancestor_joint_and_baked_local(
            &charging_handle,
            &all_nodes,
            &parent_of_node,
            &joint_of_node,
        );
        assert_eq!(parent_joint, None, "no ancestor of this node is itself a joint");

        let (t, r, s) = charging_handle.transform().decomposed();
        let mut expected =
            Mat4::from_scale_rotation_translation(Vec3::from(s), Quat::from_array(r), Vec3::from(t));
        let mut idx = parent_of_node.get(&charging_handle.index()).copied();
        let mut acc = Mat4::IDENTITY;
        while let Some(i) = idx {
            let (at, ar, asc) = all_nodes[i].transform().decomposed();
            acc = Mat4::from_scale_rotation_translation(Vec3::from(asc), Quat::from_array(ar), Vec3::from(at)) * acc;
            idx = parent_of_node.get(&i).copied();
        }
        expected = acc * expected;
        let (es, er, et) = expected.to_scale_rotation_translation();
        assert!(et.distance(t2) < 1e-4, "expected translation {et:?}, got {t2:?}");
        assert!(es.distance(s2) < 1e-4, "expected scale {es:?}, got {s2:?}");
        assert!(er.angle_between(r2) < 1e-3, "expected rotation {er:?}, got {r2:?}");
    }

    fn python_bin() -> Option<String> {

        let script_dir = workspace_root().join("scene_editor_web");
        let venv = workspace_root().join("scene_editor_web/venv/bin/python");
        let probe = format!(
            "import sys; sys.path.insert(0, {script_dir:?}); import gltf_animation"
        );

        let mut candidates: Vec<String> = Vec::new();
        if venv.is_file() {
            candidates.push(venv.to_string_lossy().into_owned());
        }
        candidates.push("python3".to_string());
        candidates.push("python".to_string());

        for candidate in candidates {
            let ok = std::process::Command::new(&candidate)
                .args(["-c", &probe])
                .status()
                .is_ok_and(|s| s.success());
            if ok {
                return Some(candidate);
            }
        }
        None
    }

    #[test]
    fn m4a1_orphan_node_promotion_end_to_end() {
        let Some((device, queue, layout)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let Some(python) = python_bin() else {
            eprintln!("skipping: no working python3/python on PATH to build the fixture");
            return;
        };

        let original_path = workspace_root().join("game/models/m4a1.glb");
        let script_dir = workspace_root().join("scene_editor_web");

        let work_dir = std::env::temp_dir().join(format!("m4a1_fixture_{}", std::process::id()));
        std::fs::create_dir_all(&work_dir).expect("create fixture dir");
        let test_path = work_dir.join("m4a1.glb");
        std::fs::copy(&original_path, &test_path).expect("copy fixture");

        for entry in std::fs::read_dir(original_path.parent().unwrap()).expect("read models dir").flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("m4a1_anim_") && name.ends_with(".bin") {
                std::fs::copy(entry.path(), work_dir.join(&name)).ok();
            }
        }

        let strip = format!(
            "import sys; sys.path.insert(0, {script_dir:?}); \
             from gltf_animation import read_joint_animation_clips, delete_joint_animation_clip; \
             from pathlib import Path; p = Path({test_path:?}); \
             [delete_joint_animation_clip(p, c['name']) for c in read_joint_animation_clips(p)]",
        );
        let status = std::process::Command::new(&python)
            .arg("-c")
            .arg(&strip)
            .status()
            .expect("run python3 to strip existing clips");
        assert!(status.success(), "stripping existing clips failed");

        let baseline = GltfMesh::load(&device, &queue, &layout, &test_path)
            .expect("baseline m4a1.glb should load");
        assert!(baseline.skin.is_none(), "file with no orphan animations must stay unskinned");
        assert!(!baseline.primitives.is_empty(), "baseline should have static primitives");
        assert!(baseline.bounding_radius > 0.0);

        let py = format!(
            "import sys; sys.path.insert(0, {script_dir:?}); from gltf_animation import write_joint_animation_clip; \
             from pathlib import Path; \
             write_joint_animation_clip(Path({test_path:?}), 'charging_handle_pull', {{'m4a1 charging handle_10': [ \
             {{'t': 0.0, 'position': [0,0,0], 'rotation': [0,0,0,1], 'scale': [1,1,1]}}, \
             {{'t': 0.3, 'position': [0,0,-0.05], 'rotation': [0,0,0,1], 'scale': [1,1,1]}}]}})",
        );
        let status = std::process::Command::new(&python)
            .arg("-c")
            .arg(&py)
            .status()
            .expect("run python3 to write the test clip");
        assert!(status.success(), "write_joint_animation_clip failed");

        let animated = GltfMesh::load(&device, &queue, &layout, &test_path).expect("animated m4a1 should load");
        std::fs::remove_dir_all(&work_dir).ok();

        assert!(animated.primitives.is_empty(), "all geometry must be routed through the skin pipeline once any node is promoted");
        let skin = animated.skin.expect("orphan animation should produce a skin");
        assert!(skin.joint_names.len() >= 19, "expected at least the 19 named m4a1 parts as joints, got {}", skin.joint_names.len());
        assert!(!skin.primitives.is_empty(), "no geometry should be silently dropped");
        assert!(animated.bounding_radius > 0.0, "bounding_radius must not regress to 0 for a fully-skinned mesh");

        let ji = skin
            .joint_names
            .iter()
            .position(|n| n == "m4a1 charging handle_10")
            .expect("charging handle should be a joint");
        let pose = &skin.animations[0];
        assert_eq!(pose.name, "charging_handle_pull");
        let (t, _r, _s) = pose.joint_transforms[ji].expect("charging handle should have a sampled pose");
        assert!(
            t.distance(Vec3::new(0.0, 0.0, -0.05)) < 1e-4,
            "expected end-keyframe (pulled) position (0,0,-0.05), got {t:?}"
        );

        let at_rest = skin.skin_matrices_blended(0, 0.0);
        let at_full = skin.skin_matrices_blended(0, 1.0);
        let rest_off = at_rest[ji].w_axis.truncate();
        let full_off = at_full[ji].w_axis.truncate();
        assert!(
            full_off.distance(rest_off) > 0.01,
            "full blend should move the charging handle joint away from rest ({rest_off:?} -> {full_off:?})"
        );

        let filler_count = pose.joint_transforms.iter().filter(|p| p.is_none()).count();
        assert!(filler_count >= 18, "every other joint should be an inert filler with no animation entry, got {filler_count} inert of {}", skin.joint_names.len());
    }

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
    }

    #[test]
    fn a_cave_bake_loads_its_per_vertex_layer_weights() {
        let Some((device, queue, layout)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let mesh = GltfMesh::load(&device, &queue, &layout, &fixture("layered_cave.glb"))
            .expect("the editor's cave glb should load");
        let prim = &mesh.primitives[0];

        let layered = prim
            .layered
            .as_ref()
            .expect("a mesh marked `shading: layered` should carry layered vertices");
        assert_eq!(layered.vertices.len(), prim.vertices.len());

        assert_eq!(layered.vertices[0].weights, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(layered.vertices[1].weights, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(layered.vertices[2].weights, [0.0, 0.0, 0.5, 0.5]);

        for i in 0..3 {
            assert_eq!(layered.vertices[i].position, prim.vertices[i].position);
            assert_eq!(layered.vertices[i].normal, prim.vertices[i].normal);
        }
    }

    #[test]
    fn a_model_with_vertex_colours_but_no_marker_is_not_layered() {

        let Some((device, queue, layout)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let mesh = GltfMesh::load(&device, &queue, &layout, &fixture("unmarked_cave.glb"))
            .expect("the unmarked glb should load");
        assert!(
            mesh.primitives[0].layered.is_none(),
            "an unmarked model opted itself into layered shading",
        );
    }

    #[test]
    fn an_ordinary_textured_model_is_not_layered() {
        let Some((device, queue, layout)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let mesh = GltfMesh::load(&device, &queue, &layout, &workspace_root().join("game/models/m4a1.glb"))
            .expect("m4a1.glb should load");
        assert!(mesh.primitives.iter().all(|p| p.layered.is_none()));
    }

    pub(crate) fn headless_gpu() -> Option<(wgpu::Device, wgpu::Queue, wgpu::BindGroupLayout)> {
        headless_gpu_with(wgpu::Features::empty())
    }

    fn headless_gpu_with(required_features: wgpu::Features) -> Option<(wgpu::Device, wgpu::Queue, wgpu::BindGroupLayout)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            apply_limit_buckets: false,
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        if !adapter.features().contains(required_features) {
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features,
            required_limits: crate::renderer::uniforms::scene_limits(wgpu::Limits::default()),
            ..Default::default()
        }))
        .ok()?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_mesh_texture_bgl"),
            entries: &crate::renderer::mesh_pipeline::material_texture_layout_entries(),
        });
        Some((device, queue, layout))
    }

    /// A model's maps encoded ahead (`bin/model_astc`) upload as ASTC, and an
    /// encoding that no longer matches its images falls back to RGBA.
    #[test]
    fn model_maps_load_as_astc_and_stale_ones_fall_back() {
        use crate::renderer::brush_astc::BrushAstc;
        use crate::renderer::terrain_pipeline::{astcenc_level, AstcArray, ASTC_BLOCK};
        let astcenc = workspace_root().join("../tools/astcenc-5.7.0/bin/astcenc");
        let source = workspace_root().join("games/artytest1/models/doors/latch_door.glb");
        if !astcenc.exists() || !source.exists() {
            eprintln!("skipping: no astcenc or no latch_door.glb");
            return;
        }
        let Some((device, queue, layout)) = headless_gpu_with(wgpu::Features::TEXTURE_COMPRESSION_ASTC) else {
            eprintln!("skipping: this GPU reads no ASTC");
            return;
        };
        let dir = std::env::temp_dir().join(format!("model_astc_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let glb = dir.join("latch_door.glb");
        std::fs::copy(&source, &glb).unwrap();

        let (doc, _buffers, images) = gltf::import(&glb).unwrap();
        let astc_dir = model_astc_dir(&glb);
        std::fs::create_dir_all(&astc_dir).unwrap();
        let mut files = Vec::new();
        for material in doc.materials() {
            let base = material_images(&material, &images).base;
            if base.1 * base.2 <= 1 || base.1 % 4 != 0 || base.2 % 4 != 0 {
                continue;
            }
            let chain = model_mips(&base);
            let data: Vec<Vec<u8>> = chain.iter().map(|l| astcenc_level(&astcenc, "fast", true, &l.rgba, l.width, l.height, &dir).unwrap()).collect();
            let file = BrushAstc {
                source: model_image_fingerprint(ModelMap::Base, &base),
                array: AstcArray { block: ASTC_BLOCK, srgb: true, width: base.1, height: base.2, levels: chain.len() as u32, data: vec![data] },
            };
            let path = model_astc_path(&astc_dir, material.index(), ModelMap::Base);
            std::fs::write(&path, file.to_bytes()).unwrap();
            files.push((path, file));
        }
        assert!(!files.is_empty(), "latch_door.glb should have a textured material");

        let formats = |mesh: &GltfMesh| mesh.primitives.iter().map(|p| p.texture.texture.format()).collect::<Vec<_>>();
        let mesh = GltfMesh::load(&device, &queue, &layout, &glb).expect("latch_door.glb should load");
        let astc = formats(&mesh).iter().filter(|f| matches!(f, wgpu::TextureFormat::Astc { .. })).count();
        assert!(astc > 0, "the encoded maps should upload as ASTC, got {:?}", formats(&mesh));

        for (path, file) in &files {
            let stale = BrushAstc { source: file.source ^ 1, array: file.array.clone() };
            std::fs::write(path, stale.to_bytes()).unwrap();
        }
        let mesh = GltfMesh::load(&device, &queue, &layout, &glb).expect("latch_door.glb should load");
        assert!(
            formats(&mesh).iter().all(|f| !matches!(f, wgpu::TextureFormat::Astc { .. })),
            "a stale encoding must not be uploaded, got {:?}",
            formats(&mesh)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn real_skinned_fixtures_still_load_correctly() {
        let Some((device, queue, layout)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let root = workspace_root();
        for rel in [
            "game/models/left_hand.glb",
            "game/models/right_hand.glb",
            "game/models/boy/boy.glb",
            "game/models/ar15/870022.gltf",
        ] {
            let path = root.join(rel);
            let mesh = GltfMesh::load(&device, &queue, &layout, &path)
                .unwrap_or_else(|e| panic!("{rel} should still load: {e}"));
            let skin = mesh.skin.as_ref().unwrap_or_else(|| panic!("{rel} should be skinned"));
            assert!(!skin.joint_names.is_empty(), "{rel}: expected real joints");
            assert!(!skin.primitives.is_empty(), "{rel}: expected skinned primitives, none dropped");
            assert!(mesh.primitives.is_empty(), "{rel}: fully-skinned fixture should have zero static primitives");
            assert!(mesh.bounding_radius > 0.0, "{rel}: bounding_radius should not be zero");
            println!(
                "{rel}: {} joints, {} skinned primitives, radius {:.3}",
                skin.joint_names.len(),
                skin.primitives.len(),
                mesh.bounding_radius
            );
        }
    }

    #[test]
    fn a_bodys_see_through_parts_are_marked_blended() {
        let Some((device, queue, layout)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let root = workspace_root();
        for (rel, some_blended) in [("game/models/boy/boy.glb", true), ("game/models/left_hand.glb", false)] {
            let mesh = GltfMesh::load(&device, &queue, &layout, &root.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
            let prims = &mesh.skin.as_ref().expect("skinned").primitives;
            let blended = prims.iter().filter(|p| p.blended).count();
            println!("{rel}: {blended} of {} skinned primitives blended", prims.len());
            assert_eq!(blended > 0, some_blended, "{rel}");
            assert!(blended < prims.len(), "{rel}: every part blended");
        }
    }

    fn pose(name: &str, entries: Vec<(usize, Vec3)>, joints: usize) -> GltfAnimationPose {
        let mut joint_transforms = vec![None; joints];
        for (ji, t) in entries {
            joint_transforms[ji] = Some((t, Quat::IDENTITY, Vec3::ONE));
        }
        GltfAnimationPose { name: name.to_string(), joint_transforms }
    }

    fn two_clips_sharing_the_bolt() -> (Vec<(Vec3, Quat, Vec3)>, Vec<GltfAnimationPose>) {
        let bind = vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE); 3];
        let animations = vec![

            pose("charging_handle", vec![(0, Vec3::new(-1.0, 0.0, 0.0)), (1, Vec3::new(-2.0, 0.0, 0.0))], 3),

            pose("fire_cycle", vec![(0, Vec3::new(-0.5, 0.0, 0.0))], 3),
        ];
        (bind, animations)
    }

    #[test]
    fn an_idle_clip_does_not_shadow_an_active_one_sharing_a_joint() {
        let (bind, anims) = two_clips_sharing_the_bolt();

        let local = blend_joint_local(&bind, &anims, &[(0, 0.0, ClipBlendMode::Override), (1, 1.0, ClipBlendMode::Override)]);
        assert_eq!(
            local[0].0,
            Vec3::new(-0.5, 0.0, 0.0),
            "the bolt should follow the active fire_cycle, not be pinned at rest by an idle charging_handle"
        );
    }

    #[test]
    fn the_higher_priority_clip_still_wins_when_both_are_active() {
        let (bind, anims) = two_clips_sharing_the_bolt();
        let local = blend_joint_local(&bind, &anims, &[(0, 1.0, ClipBlendMode::Override), (1, 1.0, ClipBlendMode::Override)]);
        assert_eq!(local[0].0, Vec3::new(-1.0, 0.0, 0.0), "first active clip in priority order owns the joint");
    }

    #[test]
    fn a_joint_only_one_clip_drives_is_unaffected_by_priority() {
        let (bind, anims) = two_clips_sharing_the_bolt();

        let local = blend_joint_local(&bind, &anims, &[(0, 1.0, ClipBlendMode::Override), (1, 1.0, ClipBlendMode::Override)]);
        assert_eq!(local[1].0, Vec3::new(-2.0, 0.0, 0.0));
    }

    #[test]
    fn every_clip_idle_leaves_the_whole_skeleton_at_bind() {
        let (bind, anims) = two_clips_sharing_the_bolt();
        let local = blend_joint_local(&bind, &anims, &[(0, 0.0, ClipBlendMode::Override), (1, 0.0, ClipBlendMode::Override)]);
        assert_eq!(local, bind);
    }

    #[test]
    fn a_joint_no_clip_drives_stays_at_bind() {
        let (bind, anims) = two_clips_sharing_the_bolt();
        let local = blend_joint_local(&bind, &anims, &[(0, 1.0, ClipBlendMode::Override), (1, 1.0, ClipBlendMode::Override)]);
        assert_eq!(local[2].0, Vec3::ZERO, "the trigger is in neither clip");
    }

    #[test]
    fn a_partial_blend_interpolates_from_bind() {
        let (bind, anims) = two_clips_sharing_the_bolt();
        let local = blend_joint_local(&bind, &anims, &[(1, 0.5, ClipBlendMode::Override)]);
        assert!((local[0].0.x - -0.25).abs() < 1e-6, "got {:?}", local[0].0);
    }

    #[test]
    fn a_blend_just_above_the_idle_threshold_counts() {
        let (bind, anims) = two_clips_sharing_the_bolt();
        let local = blend_joint_local(&bind, &anims, &[(1, IDLE_BLEND * 2.0, ClipBlendMode::Override)]);
        assert!(local[0].0.x < 0.0, "a small but real blend must move the joint, got {:?}", local[0].0);
    }

    #[test]
    fn an_additive_clip_adds_to_the_override_layer() {
        let (bind, anims) = two_clips_sharing_the_bolt();

        let local = blend_joint_local(
            &bind, &anims,
            &[(0, 1.0, ClipBlendMode::Override), (1, 1.0, ClipBlendMode::Additive)],
        );
        assert!((local[0].0.x - -1.5).abs() < 1e-6, "got {:?}", local[0].0);
    }

    #[test]
    fn an_additive_clip_scales_its_offset_by_its_own_blend() {
        let (bind, anims) = two_clips_sharing_the_bolt();
        let local = blend_joint_local(
            &bind, &anims,
            &[(0, 1.0, ClipBlendMode::Override), (1, 0.5, ClipBlendMode::Additive)],
        );
        assert!((local[0].0.x - -1.25).abs() < 1e-6, "got {:?}", local[0].0);
    }

    #[test]
    fn an_additive_clip_works_with_no_override_layer() {
        let (bind, anims) = two_clips_sharing_the_bolt();
        let local = blend_joint_local(&bind, &anims, &[(1, 1.0, ClipBlendMode::Additive)]);
        assert!((local[0].0.x - -0.5).abs() < 1e-6, "got {:?}", local[0].0);
    }

    #[test]
    fn two_additive_clips_compose_rather_than_replace() {
        let bind = vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE); 3];
        let anims = vec![
            pose("a", vec![(0, Vec3::new(-1.0, 0.0, 0.0))], 3),
            pose("b", vec![(0, Vec3::new(0.0, -2.0, 0.0))], 3),
        ];
        let local = blend_joint_local(
            &bind, &anims,
            &[(0, 1.0, ClipBlendMode::Additive), (1, 1.0, ClipBlendMode::Additive)],
        );
        assert!((local[0].0.x - -1.0).abs() < 1e-6);
        assert!((local[0].0.y - -2.0).abs() < 1e-6);
    }

    #[test]
    fn an_idle_additive_clip_contributes_nothing() {
        let (bind, anims) = two_clips_sharing_the_bolt();
        let with_idle = blend_joint_local(
            &bind, &anims,
            &[(0, 1.0, ClipBlendMode::Override), (1, 0.0, ClipBlendMode::Additive)],
        );
        let without = blend_joint_local(&bind, &anims, &[(0, 1.0, ClipBlendMode::Override)]);
        assert_eq!(with_idle[0].0, without[0].0);
    }

    #[test]
    fn override_is_the_default_and_behaves_as_before() {
        assert_eq!(ClipBlendMode::default(), ClipBlendMode::Override);
        let (bind, anims) = two_clips_sharing_the_bolt();
        let local = blend_joint_local(
            &bind, &anims,
            &[(0, 1.0, ClipBlendMode::default()), (1, 1.0, ClipBlendMode::default())],
        );
        assert_eq!(local[0].0, Vec3::new(-1.0, 0.0, 0.0), "first active Override still wins outright");
    }

    #[test]
    fn additive_rotation_composes_from_the_bind_delta() {
        let bind = vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE); 1];
        let quarter = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let anims = vec![GltfAnimationPose {
            name: "twist".into(),
            joint_transforms: vec![Some((Vec3::ZERO, quarter, Vec3::ONE))],
        }];
        let half = blend_joint_local(&bind, &anims, &[(0, 0.5, ClipBlendMode::Additive)]);
        let expected = Quat::IDENTITY.slerp(quarter, 0.5);
        assert!(half[0].1.angle_between(expected) < 1e-4);
    }

    #[test]
    fn additive_scale_multiplies_rather_than_adds() {
        let bind = vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE); 1];
        let anims = vec![GltfAnimationPose {
            name: "grow".into(),
            joint_transforms: vec![Some((Vec3::ZERO, Quat::IDENTITY, Vec3::splat(2.0)))],
        }];
        let full = blend_joint_local(&bind, &anims, &[(0, 1.0, ClipBlendMode::Additive)]);
        assert!((full[0].2.x - 2.0).abs() < 1e-6, "got {:?}", full[0].2);
        let half = blend_joint_local(&bind, &anims, &[(0, 0.5, ClipBlendMode::Additive)]);
        assert!((half[0].2.x - 1.5).abs() < 1e-6, "got {:?}", half[0].2);
    }
