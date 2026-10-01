use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

use super::texture::LoadedTexture;
use crate::renderer::layered_mesh_pipeline::LayeredVertex;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub uv2: [f32; 2],
    /// The material's emissive colour, packed RGBA8 as `0xAABBGGRR`.
    ///
    /// WHY THIS IS ON THE VERTEX
    ///
    /// It is a per-PRIMITIVE constant, and there is nowhere better to put it.
    /// The natural home is the primitive's own bind group, but that group holds
    /// the texture, which is shared through an `Arc` between primitives that may
    /// have different materials -- and adding a fifth bind group is not an
    /// option, because mobile GPUs guarantee only four and all four are spoken
    /// for (camera, model, texture, lightmap).
    ///
    /// Four bytes a vertex is the cheapest remaining place. It buys the thing
    /// that matters for a lamp: the AUTHOR decides which parts glow, so a bulb
    /// can be emissive while the housing around it is not. How brightly, right
    /// now, is a separate per-object value -- see `ModelUniform`.
    pub emissive: u32,
}

impl MeshVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x2, 4 => Uint32
    ];

    /// Pack a linear 0..1 emissive colour into the vertex field.
    ///
    /// Stored sRGB-encoded, for the same reason the lightmap is: the byte range
    /// spends its precision where the eye has it. Unpacked in the shader.
    /// Brightest emissive this encoding can carry.
    ///
    /// glTF's `KHR_materials_emissive_strength` exists precisely because a
    /// glowing surface is not a colour -- a bulb is many times brighter than
    /// white paper, and the extension routinely carries values like 25. The
    /// hanging lamp in this project does.
    pub const MAX_EMISSIVE_STRENGTH: f32 = 64.0;

    /// Pack an emissive colour that may be far BRIGHTER THAN WHITE.
    ///
    /// The previous version clamped each channel to 1.0 before encoding, which
    /// silently discarded the whole point of the strength extension: the lamp's
    /// `emissiveFactor` of [1,1,1] times a strength of 25 arrived as 1.0, so its
    /// bulb glowed at a twenty-fifth of the authored brightness and read as an
    /// emissive that had simply not been wired up.
    ///
    /// The fix keeps one `u32`: the HUE goes in RGB, normalised so it always
    /// fits, and the SCALE goes in the alpha byte, which the shader packed and
    /// then never read. Ordinary emissives -- everything at or below white --
    /// normalise by 1.0 and are unchanged.
    pub fn pack_emissive(rgb: [f32; 3]) -> u32 {
        let enc = |v: f32| -> u32 {
            let l = v.clamp(0.0, 1.0);
            let s = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
            (s * 255.0).round() as u32
        };
        let peak = rgb[0].max(rgb[1]).max(rgb[2]);
        if !(peak > 0.0) {
            return Self::NO_EMISSIVE;
        }
        // Never below 1: a dim emissive keeps its colour in RGB and a scale of
        // one, exactly as before, so nothing that used to work changes.
        let scale = peak.max(1.0).min(Self::MAX_EMISSIVE_STRENGTH);
        let inv = 1.0 / scale;
        let a = ((scale / Self::MAX_EMISSIVE_STRENGTH) * 255.0).round() as u32;
        enc(rgb[0] * inv) | (enc(rgb[1] * inv) << 8) | (enc(rgb[2] * inv) << 16) | (a << 24)
    }

    /// Emissive black: the value for every material that does not glow.
    ///
    /// The alpha byte now carries the emissive SCALE rather than an opacity, and
    /// this value is still correct: whatever the scale, the RGB is zero and zero
    /// times anything is black.
    pub const NO_EMISSIVE: u32 = 255 << 24;

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SkinnedMeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub joint_ids: [u32; 4],
    pub joint_weights: [f32; 4],
}

impl SkinnedMeshVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Uint32x4,
        4 => Float32x4,
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }

    pub fn dominant_joint(&self) -> usize {
        let mut best = 0;
        for i in 1..4 {
            if self.joint_weights[i] > self.joint_weights[best] {
                best = i;
            }
        }
        self.joint_ids[best] as usize
    }
}

/// The same triangles, as vertices the layered-mesh pipeline can draw.
///
/// Alongside the textured form rather than instead of it. The index buffer is
/// shared -- it is the same mesh -- and keeping the ordinary vertices means a
/// build without the layered path, or a pass that has no material array to hand,
/// still has something to draw.
#[derive(Clone)]
pub struct LayeredPrimitive {
    pub vertices: Vec<LayeredVertex>,
    pub vertex_buffer: wgpu::Buffer,
}

#[derive(Clone)]
pub struct MeshPrimitive {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub texture: Arc<LoadedTexture>,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    /// Present when the file asked for layered shading AND carried the weights
    /// to do it with. Both, because a mesh that asks and does not supply would
    /// otherwise render as a single flat layer with no clue why.
    pub layered: Option<LayeredPrimitive>,
    /// Whether this primitive belongs in a shadow map.
    ///
    /// False for glass. A shadow map is BINARY -- a fragment either occludes or
    /// it does not -- so a surface that transmits most of the light through it
    /// has to be quantised to one of those, and "blocks everything" is the
    /// wrong one. Measured on the hanging lamp: its envelope has
    /// `KHR_materials_transmission` of 1.0, is invisible to the eye, and as a
    /// shadow caster it blocked 100% of its own bulb's downward light while the
    /// opaque housing around it blocked 2%. The room was black because the
    /// lamp's glass sealed the bulb inside a shadow.
    pub casts_shadow: bool,
    /// Glass or a blended material: drawn after everything opaque and the
    /// sky, as anything see-through has to be. Drawn among the opaque parts,
    /// the lamp's clear globe (alpha 0, depth written) hid whatever of its own
    /// cage was drawn after it -- which, once the cage's thin wires moved to a
    /// pass of their own, was the back half of every cage.
    pub blended: bool,
    /// The primitive's wires, chain links and rims, apart: see `thin_parts`.
    pub thin: Option<super::thin_parts::ThinParts>,
}

#[cfg(test)]
mod emissive_tests {
    use super::MeshVertex;

    fn unpack(packed: u32) -> [f32; 3] {
        // The shader's decode, in Rust: sRGB per channel, times the scale byte.
        let dec = |b: u32| {
            let c = b as f32 / 255.0;
            if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
        };
        let scale = ((packed >> 24) & 255) as f32 / 255.0 * MeshVertex::MAX_EMISSIVE_STRENGTH;
        [
            dec(packed & 255) * scale,
            dec((packed >> 8) & 255) * scale,
            dec((packed >> 16) & 255) * scale,
        ]
    }

    #[test]
    fn an_emissive_brighter_than_white_survives_the_round_trip() {
        // THE bug. glTF's KHR_materials_emissive_strength exists because a bulb
        // is many times brighter than paper, and the lamp in this project
        // carries a strength of 25. Clamping each channel to 1.0 before
        // encoding threw all of it away, so the bulb glowed at a twenty-fifth
        // of its authored brightness and read as an emissive nobody had wired
        // up -- which is exactly how it was reported.
        let back = unpack(MeshVertex::pack_emissive([25.0, 25.0, 25.0]));
        for c in back {
            assert!(
                (c - 25.0).abs() < 25.0 * 0.02,
                "a strength of 25 came back as {c}, not within 2% of 25",
            );
        }
    }

    #[test]
    fn an_ordinary_emissive_is_unchanged() {
        // Everything at or below white must round-trip as it always did, or
        // this fix silently re-grades every emissive material in the project.
        for v in [[1.0f32, 1.0, 1.0], [0.5, 0.25, 0.0], [0.0, 0.0, 0.8]] {
            let back = unpack(MeshVertex::pack_emissive(v));
            for (got, want) in back.iter().zip(v.iter()) {
                assert!((got - want).abs() < 0.02, "{v:?} came back as {back:?}");
            }
        }
    }

    #[test]
    fn the_colour_of_a_bright_emissive_is_preserved_not_just_its_level() {
        // The scale is shared across channels, so a coloured bulb must keep its
        // hue -- a warm lamp that comes back white is as wrong as a dark one.
        let back = unpack(MeshVertex::pack_emissive([20.0, 10.0, 5.0]));
        assert!((back[0] / back[1] - 2.0).abs() < 0.1, "{back:?}");
        assert!((back[1] / back[2] - 2.0).abs() < 0.1, "{back:?}");
    }

    #[test]
    fn a_material_that_does_not_glow_stays_black() {
        assert_eq!(unpack(MeshVertex::NO_EMISSIVE), [0.0, 0.0, 0.0]);
        assert_eq!(unpack(MeshVertex::pack_emissive([0.0, 0.0, 0.0])), [0.0, 0.0, 0.0]);
    }
}
