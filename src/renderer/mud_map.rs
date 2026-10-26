//! THE MUD MAP ON THE GPU: one deform map per terrain, stamped where walkers
//! step on mud, read by the ground (sink, wet, tilt, pool) and by the water
//! over it (prints under a thin film).
//!
//! Shared by the desktop renderer (ssstudio) and the headset (`XrRenderer`),
//! so a print made in either looks the same. The stamping, spatter, fading and
//! relaxation live in `spacesoup_graphics::mud`; this holds the texture beside
//! the map and turns the map's `[height, wet]` bytes into the texels the
//! shaders read (`terrain_pipeline::deform_rows`, which adds the slope).
//!
//! See docs/engine/mud-and-deform.md.

use spacesoup_graphics::mud::{MudMap, Walkers};
use wgpu::*;

use super::terrain_pipeline::{deform_rows, deform_rows_touched, TerrainImage, DEFORM_FORMAT, DEFORM_TEXEL_BYTES};

/// Longest step `MudBody::tick` lets the map settle by, so a hitch (a level
/// load, a pause) does not relax every print at once.
pub const MAX_SETTLE_STEP: f32 = 0.25;

/// Metres of sinking or piling as the map's print depths: a full print
/// (`PRINT_SINK` steps of the 127.5 either side of flat) sinks
/// `deform_metres * PRINT_SINK / 127.5` (`DEFORM_VERTEX`).
pub fn print_depths(metres: f32, deform_metres: f32) -> f32 {
    metres / (deform_metres.max(1e-4) * spacesoup_graphics::mud::PRINT_SINK / 127.5)
}

pub struct MudBody {
    pub map: MudMap,
    pub texture: Texture,
    pub view: TextureView,
    pub walkers: Walkers,
    last_tick: Option<std::time::Instant>,
}

impl MudBody {
    /// A flat map over the terrain's footprint, `min`..`max` in world x/z: the
    /// same footprint the terrain's uv (and the ground map) spans, which is
    /// what lets the shaders read the map at the ground's own uv.
    pub fn new(device: &Device, queue: &Queue, min: [f32; 2], max: [f32; 2]) -> Self {
        let map = MudMap::new(min, max, spacesoup_graphics::mud::DEFAULT_TEXEL, spacesoup_graphics::mud::MAX_SIDE);
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("terrain_deform"),
            size: Extent3d { width: map.width(), height: map.height(), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: DEFORM_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            bytemuck::cast_slice(&deform_rows(map.bytes(), map.width(), map.height(), 0, map.height())),
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(map.width() * DEFORM_TEXEL_BYTES), rows_per_image: Some(map.height()) },
            Extent3d { width: map.width(), height: map.height(), depth_or_array_layers: 1 },
        );
        let view = texture.create_view(&TextureViewDescriptor::default());
        Self { map, texture, view, walkers: Walkers::default(), last_tick: None }
    }

    /// Whether this map was made for the footprint `min`..`max`.
    pub fn spans(&self, min: [f32; 2], max: [f32; 2]) -> bool {
        self.map.min() == min && self.map.max() == max
    }

    /// The metres one texel spans, the larger way: what the ground's slope is
    /// measured across (`TerrainMaterialUniform::deform_texel`).
    pub fn texel_metres(&self) -> f32 {
        let [tx, tz] = self.map.texel_metres();
        tx.max(tz)
    }

    /// Send the rows stamping or settling changed (with the neighbours whose
    /// slopes read them).
    pub fn flush(&mut self, queue: &Queue) {
        let Some([_, y0, _, y1]) = self.map.take_dirty() else { return };
        if y1 <= y0 {
            return;
        }
        let (width, height) = (self.map.width(), self.map.height());
        for (from, to) in deform_rows_touched(y0, y1, height) {
            queue.write_texture(
                TexelCopyTextureInfo { texture: &self.texture, mip_level: 0, origin: Origin3d { x: 0, y: from, z: 0 }, aspect: TextureAspect::All },
                bytemuck::cast_slice(&deform_rows(self.map.bytes(), width, height, from, to)),
                TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(width * DEFORM_TEXEL_BYTES), rows_per_image: Some(to - from) },
                Extent3d { width, height: to - from, depth_or_array_layers: 1 },
            );
        }
    }

    /// Settle the map by the wall time since the last tick (prints fade, the
    /// wet zone bakes, the mud relaxes), then send what changed.
    pub fn tick(&mut self, queue: &Queue) {
        let now = std::time::Instant::now();
        let dt = self.last_tick.map_or(0.0, |t| now.duration_since(t).as_secs_f32()).min(MAX_SETTLE_STEP);
        self.last_tick = Some(now);
        if dt > 0.0 {
            self.map.settle(dt);
        }
        self.flush(queue);
    }

    /// Step every walker (`id`, world position) over the map, each print
    /// scaled by the mud under that foot; walkers not listed are forgotten.
    /// Returns (walkers on mud, new prints).
    pub fn walk(&mut self, walkers: &[(usize, glam::Vec3)], splat: Option<&TerrainImage>, mud_layer: i32) -> (u32, u32) {
        let present: Vec<usize> = walkers.iter().map(|w| w.0).collect();
        let (mut wading, mut prints) = (0, 0);
        for (id, p) in walkers {
            let weight = self.weight_at(splat, mud_layer, p.x, p.z);
            wading += (weight >= spacesoup_graphics::mud::MUD_WEIGHT_FLOOR) as u32;
            prints += self.walkers.step(&mut self.map, *id, [p.x, p.z], weight) as u32;
        }
        self.walkers.forget_absent(&present);
        (wading, prints)
    }

    /// A blast's crater, scaled by the mud there. Returns that mud weight.
    pub fn blast(&mut self, splat: Option<&TerrainImage>, mud_layer: i32, at: glam::Vec3, radius: f32, depth: f32) -> f32 {
        let weight = self.weight_at(splat, mud_layer, at.x, at.z);
        if weight > 0.0 {
            self.map.crater([at.x, at.z], radius, depth * weight);
        }
        weight
    }

    /// A MUD SPLASH'S CRATER, dug when it is raised
    /// (`effects::mud_splash::crater`), scaled by the mud there.
    /// `deform_metres` is the ground's `TerrainMaterialUniform::deform_metres`.
    pub fn splash_crater(&mut self, s: &crate::renderer::effects::Splash, splat: Option<&TerrainImage>, mud_layer: i32, deform_metres: f32) {
        let ([x, z], radius, depth) = crate::renderer::effects::mud_splash::crater(s);
        let weight = self.weight_at(splat, mud_layer, x, z);
        if weight > 0.0 && radius > 0.0 {
            self.map.crater([x, z], radius, print_depths(depth, deform_metres) * weight);
        }
    }

    /// Clumps come down (`effects::mud_splash::MudLandings::take`): each a
    /// pat of mud where it lands, scaled by the mud there; none off it.
    pub fn land(&mut self, landings: &[crate::renderer::effects::mud_splash::Landing], splat: Option<&TerrainImage>, mud_layer: i32, deform_metres: f32) {
        for l in landings {
            let weight = self.weight_at(splat, mud_layer, l.centre[0], l.centre[1]);
            if weight >= spacesoup_graphics::mud::MUD_WEIGHT_FLOOR {
                self.map.deposit(l.centre, l.radius, l.heading, l.stretch, print_depths(l.height, deform_metres) * weight, weight);
            }
        }
    }

    /// How much of the ground at world (`x`, `z`) is the mud layer, 0..1: the
    /// splat's share for that layer (nearest texel), or all of it for an
    /// unpainted ground whose mud layer is its first. The CPU twin of the
    /// terrain shader's `mud_weight_v`.
    pub fn weight_at(&self, splat: Option<&TerrainImage>, mud_layer: i32, x: f32, z: f32) -> f32 {
        if !(0..4).contains(&mud_layer) || !self.map.contains(x, z) {
            return 0.0;
        }
        let Some(s) = splat.filter(|s| s.width > 0 && s.height > 0) else { return if mud_layer == 0 { 1.0 } else { 0.0 } };
        let p = self.map.placement();
        let u = ((x - p[0]) * p[2]).clamp(0.0, 1.0);
        let v = ((z - p[1]) * p[3]).clamp(0.0, 1.0);
        let ix = ((u * s.width as f32) as u32).min(s.width - 1);
        let iz = ((v * s.height as f32) as u32).min(s.height - 1);
        let k = ((iz * s.width + ix) * 4) as usize;
        let w: [f32; 4] = std::array::from_fn(|c| s.rgba[k + c] as f32);
        let total: f32 = w.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        w[mud_layer as usize] / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_foot_on_mud_stamps_and_a_foot_off_it_does_not() {
        let Some((device, queue)) = super::super::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let mut mud = MudBody::new(&device, &queue, [0.0, 0.0], [4.0, 4.0]);
        // Left half layer 0 (mud), right half layer 1 (grass).
        let splat = TerrainImage { width: 2, height: 1, rgba: vec![255, 0, 0, 0, 0, 255, 0, 0] };
        assert_eq!(mud.weight_at(Some(&splat), 0, 1.0, 2.0), 1.0);
        assert_eq!(mud.weight_at(Some(&splat), 0, 3.0, 2.0), 0.0);
        assert_eq!(mud.weight_at(None, 0, 3.0, 2.0), 1.0, "unpainted ground is all its first layer");
        assert_eq!(mud.weight_at(Some(&splat), -1, 1.0, 2.0), 0.0, "no mud layer, no mud");
        let mut stamped = 0;
        for k in 0..40 {
            let z = 0.2 + 0.09 * k as f32;
            stamped += mud.walk(&[(1, glam::Vec3::new(1.0, 0.0, z)), (2, glam::Vec3::new(3.0, 0.0, z))], Some(&splat), 0).1;
        }
        assert!(stamped > 0, "a walk across the mud left no print");
        let sunk = |x: f32| (0..40).map(|k| mud.map.sunk(x, 0.2 + 0.09 * k as f32)).fold(0.0, f32::max);
        assert!(sunk(1.0) > 0.0 && sunk(3.0) == 0.0, "mud {} grass {}", sunk(1.0), sunk(3.0));
        mud.tick(&queue);
        assert!(mud.map.take_dirty().is_none(), "the tick sent what the steps changed");
        assert!(mud.spans([0.0, 0.0], [4.0, 4.0]) && !mud.spans([0.0, 0.0], [5.0, 4.0]));
    }
}
