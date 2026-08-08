    use super::*;

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

    /// A Python 3 that actually runs.
    ///
    /// The fixture work here goes through scene_editor_web/gltf_animation.py, so
    /// this test needs an interpreter. `python3` is not on PATH on a stock
    /// Windows box -- the name resolves to a Microsoft Store stub that prints an
    /// advert and exits non-zero -- so try `python` too, and verify it really
    /// runs rather than trusting the name. Skips like the GPU check above when
    /// there is none, because a missing dev tool is not a renderer regression.
    fn python_bin() -> Option<&'static str> {
        for candidate in ["python3", "python"] {
            let ok = std::process::Command::new(candidate)
                .args(["-c", "import json,struct,pathlib"])
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
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

        // Work on a private copy, and strip whatever clips the shared asset
        // currently carries so the "before" half of this test is controlled here.
        //
        // This used to assert that game/models/m4a1.glb itself had no orphan
        // animations. That was true when written and stopped being true the moment
        // an animator saved a real clip into it, which turned this red for reasons
        // that have nothing to do with node promotion. A renderer test must not
        // depend on the current contents of a hand-edited shared model.
        let work_dir = std::env::temp_dir().join(format!("m4a1_fixture_{}", std::process::id()));
        std::fs::create_dir_all(&work_dir).expect("create fixture dir");
        let test_path = work_dir.join("m4a1.glb");
        std::fs::copy(&original_path, &test_path).expect("copy fixture");
        // The clips reference external _anim_*.bin buffers, so the copy needs them
        // too or stripping (and loading) hits a missing file.
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
        let status = std::process::Command::new(python)
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
        let status = std::process::Command::new(python)
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

    fn headless_gpu() -> Option<(wgpu::Device, wgpu::Queue, wgpu::BindGroupLayout)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_mesh_texture_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        Some((device, queue, layout))
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

    use crate::renderer::mesh::skin::{blend_joint_local, ClipBlendMode, GltfAnimationPose, IDLE_BLEND};

    // ── Multi-clip blending ──────────────────────────────────────────────────
    //
    // Pure pose math, so no GPU: blend_joint_local is deliberately free-standing
    // for exactly this reason.

    fn pose(name: &str, entries: Vec<(usize, Vec3)>, joints: usize) -> GltfAnimationPose {
        let mut joint_transforms = vec![None; joints];
        for (ji, t) in entries {
            joint_transforms[ji] = Some((t, Quat::IDENTITY, Vec3::ONE));
        }
        GltfAnimationPose { name: name.to_string(), joint_transforms }
    }

    /// bolt = joint 0, charging handle = joint 1, trigger = joint 2.
    fn two_clips_sharing_the_bolt() -> (Vec<(Vec3, Quat, Vec3)>, Vec<GltfAnimationPose>) {
        let bind = vec![(Vec3::ZERO, Quat::IDENTITY, Vec3::ONE); 3];
        let animations = vec![
            // clip 0: charging_handle -- moves the bolt AND the handle
            pose("charging_handle", vec![(0, Vec3::new(-1.0, 0.0, 0.0)), (1, Vec3::new(-2.0, 0.0, 0.0))], 3),
            // clip 1: fire_cycle -- moves the bolt only, a different distance
            pose("fire_cycle", vec![(0, Vec3::new(-0.5, 0.0, 0.0))], 3),
        ];
        (bind, animations)
    }

    #[test]
    fn an_idle_clip_does_not_shadow_an_active_one_sharing_a_joint() {
        let (bind, anims) = two_clips_sharing_the_bolt();
        // charging_handle is listed first and is fully idle; fire_cycle is driving.
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
        // The handle is only in clip 0, so it moves even though clip 1 is louder.
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

    // Guards the threshold itself: a blend just above IDLE_BLEND must still count,
    // or a slow pull would drop frames near the start of its travel.
    #[test]
    fn a_blend_just_above_the_idle_threshold_counts() {
        let (bind, anims) = two_clips_sharing_the_bolt();
        let local = blend_joint_local(&bind, &anims, &[(1, IDLE_BLEND * 2.0, ClipBlendMode::Override)]);
        assert!(local[0].0.x < 0.0, "a small but real blend must move the joint, got {:?}", local[0].0);
    }

    // ── Additive layering ────────────────────────────────────────────────────
    //
    // The case this exists for: recoil riding a cycling bolt. The Override layer
    // says where the bolt is; recoil nudges it from there.

    #[test]
    fn an_additive_clip_adds_to_the_override_layer() {
        let (bind, anims) = two_clips_sharing_the_bolt();
        // clip 0 puts the bolt at -1.0; clip 1 additively offsets by -0.5.
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

    // Without an Override layer an additive clip still works, offsetting bind --
    // so a recoil clip is meaningful on its own, not only on top of something.
    #[test]
    fn an_additive_clip_works_with_no_override_layer() {
        let (bind, anims) = two_clips_sharing_the_bolt();
        let local = blend_joint_local(&bind, &anims, &[(1, 1.0, ClipBlendMode::Additive)]);
        assert!((local[0].0.x - -0.5).abs() < 1e-6, "got {:?}", local[0].0);
    }

    // Two additive clips must compose. Storing absolute poses instead of offsets
    // would make the last one erase the first.
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

    // Every existing clip is Override, so the default must reproduce exactly what
    // the runtime did before layering existed.
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
