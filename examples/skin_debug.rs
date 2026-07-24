use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

fn main() {
    let path = std::env::args().nth(1).expect("usage: skin_debug <path-to-glb>");
    let (doc, _buffers, _images) = gltf::import(&path).expect("failed to open glb");

    for skin in doc.skins() {
        let joint_nodes: Vec<gltf::Node> = skin.joints().collect();
        let joint_names: Vec<String> = joint_nodes
            .iter()
            .map(|j| j.name().unwrap_or("").to_string())
            .collect();

        let node_index_to_joint: HashMap<usize, usize> = joint_nodes
            .iter()
            .enumerate()
            .map(|(ji, node)| (node.index(), ji))
            .collect();
        let mut parent_of_node: HashMap<usize, usize> = HashMap::new();
        for node in doc.nodes() {
            for child in node.children() {
                parent_of_node.insert(child.index(), node.index());
            }
        }
        let joint_parents: Vec<Option<usize>> = joint_nodes
            .iter()
            .map(|node| {
                parent_of_node
                    .get(&node.index())
                    .and_then(|pidx| node_index_to_joint.get(pidx).copied())
            })
            .collect();

        let all_nodes: Vec<gltf::Node> = doc.nodes().collect();
        let joint_local_bind: Vec<(Vec3, Quat, Vec3)> = joint_nodes
            .iter()
            .enumerate()
            .map(|(ji, node)| {
                let (t, r, s) = node.transform().decomposed();
                let mut local = Mat4::from_scale_rotation_translation(
                    Vec3::from(s),
                    Quat::from_array(r),
                    Vec3::from(t),
                );

                if joint_parents[ji].is_none() {
                    let mut ancestor_idx = parent_of_node.get(&node.index()).copied();
                    let mut ancestor_mat = Mat4::IDENTITY;
                    while let Some(idx) = ancestor_idx {
                        if node_index_to_joint.contains_key(&idx) {
                            break;
                        }
                        let (at, ar, asc) = all_nodes[idx].transform().decomposed();
                        let anc_local = Mat4::from_scale_rotation_translation(
                            Vec3::from(asc),
                            Quat::from_array(ar),
                            Vec3::from(at),
                        );
                        ancestor_mat = anc_local * ancestor_mat;
                        ancestor_idx = parent_of_node.get(&idx).copied();
                    }
                    local = ancestor_mat * local;
                }

                let (s2, r2, t2) = local.to_scale_rotation_translation();
                (t2, r2, s2)
            })
            .collect();

        let idx = |name: &str| joint_names.iter().position(|n| n == name);

        println!("=== skin: {} joints ===", joint_names.len());
        for chain_name in ["Right arm", "Right elbow", "Right wrist", "Right leg", "Right knee", "Right ankle"] {
            if let Some(ji) = idx(chain_name) {
                let (t, _, _) = joint_local_bind[ji];
                println!(
                    "{chain_name:15} idx={ji:3} local_translation={:?} length={:.5}",
                    t,
                    t.length()
                );
            } else {
                println!("{chain_name:15} NOT FOUND");
            }
        }

        let mut hier: Vec<Mat4> = vec![Mat4::IDENTITY; joint_names.len()];
        for ji in 0..joint_names.len() {
            let (t, r, s) = joint_local_bind[ji];
            let local_mat = Mat4::from_scale_rotation_translation(s, r, t);
            hier[ji] = match joint_parents[ji] {
                Some(pi) => hier[pi] * local_mat,
                None => local_mat,
            };
        }
        if let Some(head_ji) = idx("Head") {
            let head_world = hier[head_ji].transform_point3(Vec3::ZERO);
            println!("=== computed bind_head_height (hierarchical Head Y) = {:.5} ===", head_world.y);
            println!("Head world position: {:?}", head_world);
        }
        if let Some(hips_ji) = idx("Hips") {
            let (t, _r, s) = joint_local_bind[hips_ji];
            println!("Hips local translation: {:?}, scale: {:?}", t, s);
        }

        println!("=== all joint names (with byte length, quoted) ===");
        for (i, n) in joint_names.iter().enumerate() {
            if n.to_lowercase().contains("shoulder")
                || n.to_lowercase().contains("arm")
                || n.to_lowercase().contains("elbow")
                || n.to_lowercase().contains("wrist")
                || n.to_lowercase().contains("leg")
                || n.to_lowercase().contains("knee")
                || n.to_lowercase().contains("ankle")
                || n.to_lowercase() == "hips"
            {
                println!("  [{i:3}] {:?} (len={})", n, n.len());
            }
        }
    }
}

