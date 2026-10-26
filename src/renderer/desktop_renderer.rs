use std::collections::HashMap;
use crate::renderer::mesh::create_lightmap_texture;
use wgpu::util::DeviceExt;
use wgpu::*;

use super::cuboid::{build_solid_mesh_one, build_wire_mesh_one, CuboidSnapshot, SolidVertex, WireVertex};
use super::lights::LightsUniform;
use super::mesh::LoadedTexture;
use super::shadow::{self, ShadowKind, ShadowMap};
use super::uniforms::ShadowUpload;
use super::water_pipeline::{WaterPipeline, WaterUniform, WaterVertex};
use super::water_waves::{WaveField, WaveParams};
use super::{ground_map, icon, lights, mesh_pipeline, pipeline, uniforms};
use super::{Camera, Cuboid, MeshInstance, WorldPanel};
use super::mud_map::MudBody;

mod day;
pub use day::DayLight;

struct CuboidCacheEntry {
    snapshot: CuboidSnapshot,
    solid: Option<(Vec<SolidVertex>, Vec<u32>)>,
    wire: Option<(Vec<WireVertex>, Vec<u32>)>,
}

struct DesktopTerrain {
    pipeline: super::terrain_pipeline::TerrainPipeline,
    material: super::terrain_pipeline::TerrainMaterial,
    vb: Buffer,
    ib: Buffer,
    index_count: u32,
    mud: Option<MudBody>,
    splat: Option<super::terrain_pipeline::TerrainImage>,
    mud_layer: i32,
}

struct DesktopWater {
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    uniform_buffer: Buffer,
    bind_groups: [BindGroup; 2],
    index_count: u32,
    uniform: WaterUniform,
    waves: WaveField,
    bounds: (glam::Vec3, glam::Vec3),
}

pub struct DesktopGround {
    pub heights: ground_map::HeightGrid,
    pub layers: Vec<Option<super::terrain_pipeline::TerrainImage>>,
    pub splat: Option<super::terrain_pipeline::TerrainImage>,
    pub settings: super::terrain_pipeline::TerrainMaterialUniform,
}

pub const DESKTOP_GROUND_MAP_SIZE: u32 = 512;

pub struct Renderer {
    pub device: Device,
    pub queue: Queue,
    solid_pipeline: pipeline::SolidPipeline,
    wire_pipeline: pipeline::WirePipeline,

    overlay_pipeline: pipeline::SolidPipeline,

    pub format: TextureFormat,

    pub clear_color: wgpu::Color,

    terrain: Option<DesktopTerrain>,

    pub terrain_visible: bool,
    mesh_pipeline: mesh_pipeline::MeshPipeline,
    skinned_mesh_pipeline: mesh_pipeline::SkinnedMeshPipeline,
    uniform_buf: uniforms::UniformBuffer,
    lights_uniform: LightsUniform,
    shadow_map: ShadowMap,

    sun_shadow_radius: f32,
    depth_texture: Texture,
    depth_view: TextureView,
    pub width: u32,
    pub height: u32,
    cuboid_cache: HashMap<u64, CuboidCacheEntry>,
    cuboid_lightmaps: HashMap<String, LoadedTexture>,
    default_cuboid_lightmap: LoadedTexture,
    mesh_lightmaps: HashMap<String, LoadedTexture>,
    default_mesh_lightmap: LoadedTexture,

    raw_wire: (Vec<WireVertex>, Vec<u32>),

    pub sky: Option<uniforms::SkyUpload>,

    pub post: uniforms::PostUpload,

    sky_pipeline: super::sky::SkyPipeline,
    sky_dome: Option<super::sky::Sky>,

    pub sky_visible: bool,

    particle_pipeline: super::particle_pipeline::ParticlePipeline,
    particles: Vec<super::particle_pipeline::ParticleInstance>,
    water_pipeline: Option<WaterPipeline>,
    water_bodies: Vec<DesktopWater>,
    water_seconds: f64,
    ground: Option<DesktopGround>,
    ground_dirty: bool,
    effects: Option<super::desktop_effects::DesktopEffects>,
    /// The time of day on the sky dome (`set_time_of_day`).
    day: Option<day::DesktopDay>,
}

impl Renderer {
    pub fn from_device(
        device: Device,
        queue: Queue,
        format: TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let lights_uniform = LightsUniform::new(&device);

        let shadow_map = ShadowMap::new(&device);
        let uniform_buf = uniforms::UniformBuffer::new(
            &device,
            &lights_uniform,
            shadow_map.sun_depth_view(),
            shadow_map.sun_dynamic_depth_view(),
            shadow_map.spot_depth_view(),
            shadow_map.sampler(),
        );
        let solid_pipeline = pipeline::SolidPipeline::new(&device, format, &uniform_buf.layout);
        let wire_pipeline = pipeline::WirePipeline::new(&device, format, &uniform_buf.layout);
        let overlay_pipeline = pipeline::SolidPipeline::new_overlay(&device, format, &uniform_buf.layout);
        let mesh_pipeline = mesh_pipeline::MeshPipeline::new(&device, format, &uniform_buf.layout);
        let skinned_mesh_pipeline =
            mesh_pipeline::SkinnedMeshPipeline::new(&device, format, &uniform_buf.layout);
        let sky_pipeline = super::sky::SkyPipeline::new(&device, format, &uniform_buf.layout, 1);
        let particle_pipeline = super::particle_pipeline::ParticlePipeline::new(&device, format, &uniform_buf.layout);
        let (depth_texture, depth_view) = Self::make_depth(&device, width, height);
        let white_pixel = [255u8, 255, 255, 255];
        let default_cuboid_lightmap =
            create_lightmap_texture(&device, &queue, &solid_pipeline.lightmap_layout, &white_pixel, 1, 1, None);

        let default_mesh_lightmap = crate::renderer::brush_pipeline::default_brush_lightmap(
            &device,
            &queue,
            &mesh_pipeline.lightmap_layout,
        );

        Self {
            device,
            queue,
            solid_pipeline,
            wire_pipeline,
            overlay_pipeline,
            format,
            clear_color: wgpu::Color { r: 0.02, g: 0.02, b: 0.05, a: 1.0 },
            terrain: None,
            terrain_visible: true,
            mesh_pipeline,
            skinned_mesh_pipeline,
            uniform_buf,
            lights_uniform,
            shadow_map,
            sun_shadow_radius: 40.0,
            depth_texture,
            depth_view,
            width,
            height,
            cuboid_cache: HashMap::new(),
            cuboid_lightmaps: HashMap::new(),
            default_cuboid_lightmap,
            mesh_lightmaps: HashMap::new(),
            default_mesh_lightmap,
            raw_wire: (Vec::new(), Vec::new()),
            sky: None,
            post: uniforms::PostUpload::default(),
            sky_pipeline,
            sky_dome: None,
            sky_visible: true,
            particle_pipeline,
            particles: Vec::new(),
            water_pipeline: None,
            water_bodies: Vec::new(),
            water_seconds: 0.0,
            ground: None,
            ground_dirty: false,
            effects: None,
            day: None,
        }
    }

    pub fn set_particles(&mut self, items: Vec<super::particle_pipeline::ParticleInstance>) {
        self.particles = items;
    }

    fn effects_made(&mut self) -> &mut super::desktop_effects::DesktopEffects {
        let (device, queue, format, layout) = (&self.device, &self.queue, self.format, &self.uniform_buf.layout);
        self.effects.get_or_insert_with(|| super::desktop_effects::DesktopEffects::new(device, queue, format, layout))
    }

    /// The scene's effect emitters, in world space, as the headset is given
    /// them. The effects pipelines are made on the first that is not empty.
    pub fn set_effects(&mut self, emitters: Vec<super::effects::EffectEmitter>) {
        if emitters.is_empty() && self.effects.is_none() {
            return;
        }
        self.effects_made().set_emitters(emitters);
    }

    pub fn set_effect_library(&mut self, library: super::effects::stack::EffectLibrary) {
        if library.is_empty() && self.effects.is_none() {
            return;
        }
        self.effects_made().set_library(library);
    }

    /// The scene's placed effect assets; call after `set_effect_library`.
    pub fn place_effects(&mut self, placed: &[super::effects::stack::PlacedEffect]) {
        if placed.is_empty() && self.effects.is_none() {
            return;
        }
        self.effects_made().place(placed);
    }

    /// Adds an effect asset to the library, or replaces the one of its name
    /// (an editor's unsaved version): what places or fires it takes the new
    /// version on its next frame.
    pub fn put_effect(&mut self, asset: &super::effect_asset::EffectAsset) -> Result<super::effects::stack::EffectId, Vec<super::effect_asset::Issue>> {
        self.effects_made().put(asset)
    }

    pub fn effect_id(&self, name: &str) -> Option<super::effects::stack::EffectId> {
        self.effects.as_ref().and_then(|fx| fx.id(name))
    }

    pub fn add_burst(&mut self, burst: super::effects::stack::EffectBurst) {
        self.effects_made().add_burst(burst);
    }

    /// The effect particles drawn last frame: over, and screened.
    pub fn effects_counts(&self) -> (u32, u32) {
        self.effects.as_ref().map_or((0, 0), super::desktop_effects::DesktopEffects::counts)
    }

    pub fn set_effects_overdraw(&mut self, on: bool) {
        if let Some(fx) = self.effects.as_mut() {
            fx.set_overdraw(on);
        }
    }

    /// The effects' clock in seconds, and the hour of day the mist reads.
    pub fn set_effects_time(&mut self, seconds: f64, hour: Option<f32>) {
        if let Some(fx) = self.effects.as_mut() {
            (fx.seconds, fx.hour) = (seconds, hour);
        }
    }

    pub fn set_ground(&mut self, ground: Option<DesktopGround>) {
        self.ground = ground;
        self.ground_dirty = true;
    }

    /// The ground's heights alone, after a sculpt: its look stays as it was.
    pub fn set_ground_heights(&mut self, heights: ground_map::HeightGrid) {
        if let Some(g) = self.ground.as_mut() {
            g.heights = heights;
            self.ground_dirty = true;
        }
    }

    pub fn set_water(&mut self, bodies: &[(Vec<WaterVertex>, Vec<u32>, WaterUniform, WaveParams)]) {
        self.ground_dirty = true;
        if bodies.is_empty() {
            self.water_bodies.clear();
            return;
        }
        if self.water_pipeline.is_none() {
            let mut made = WaterPipeline::new(&self.device, self.format, &self.uniform_buf.layout, 1);
            if let Ok(cut) = std::env::var("DESKTOP_WATER_CUT") {
                match made.with_cut(&self.device, &cut) {
                    Some(p) => {
                        log::info!("desktop water: drawing with the '{cut}' cut");
                        made.pipeline = p;
                    }
                    None => log::warn!("desktop water: no cut named '{cut}'"),
                }
            }
            self.water_pipeline = Some(made);
        }
        let pipeline = self.water_pipeline.as_ref().expect("made above");
        let placement = self.terrain.as_ref().and_then(|t| t.mud.as_ref()).map_or([0.0; 4], |m| m.map.placement());
        let deform_view = self.terrain.as_ref().and_then(|t| t.mud.as_ref()).map(|m| &m.view);
        self.water_bodies = bodies
            .iter()
            .filter(|(v, i, _, _)| !v.is_empty() && !i.is_empty())
            .map(|(verts, indices, uniform, params)| {
                let bounds = super::water_pipeline::water_bounds(verts, params);
                let mut uniform = *uniform;
                uniform.prints = placement;
                let vertex_buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
                    label: Some("water_vb"),
                    contents: bytemuck::cast_slice(verts),
                    usage: BufferUsages::VERTEX,
                });
                let index_buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
                    label: Some("water_ib"),
                    contents: bytemuck::cast_slice(indices),
                    usage: BufferUsages::INDEX,
                });
                let uniform_buffer = self.device.create_buffer_init(&util::BufferInitDescriptor {
                    label: Some("water_uniform"),
                    contents: bytemuck::bytes_of(&uniform),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                });
                let waves = WaveField::new(&self.device, &self.queue, *params);
                waves.reprime();
                let bind_groups = pipeline.bind_groups_with(&self.device, &uniform_buffer, &waves, deform_view.unwrap_or(&pipeline.default_mud));
                DesktopWater {
                    vertex_buffer,
                    index_buffer,
                    uniform_buffer,
                    bind_groups,
                    index_count: indices.len() as u32,
                    uniform,
                    waves,
                    bounds,
                }
            })
            .collect();
        log::info!("desktop water: {} body/bodies installed", self.water_bodies.len());
    }

    pub fn water_body_count(&self) -> usize {
        self.water_bodies.len()
    }

    pub fn set_water_time(&mut self, seconds: f32) {
        let dt = (f64::from(seconds) - self.water_seconds).clamp(0.0, 0.25) as f32;
        self.water_seconds = f64::from(seconds);
        for body in &mut self.water_bodies {
            body.uniform.set_time(self.water_seconds, body.waves.params.loop_seconds);
            self.queue.write_buffer(&body.uniform_buffer, 0, bytemuck::bytes_of(&body.uniform));
        }
        if let Some(m) = self.terrain.as_mut().and_then(|t| t.mud.as_mut()) {
            m.map.settle(dt);
        }
    }

    /// A blast scars the ground: a crater of `radius` metres, sunk `depth`
    /// (0..1 of a full footprint) scaled by how muddy the ground is there, the
    /// scar soaked dark. Returns that mud weight so the caller can size the
    /// thrown-up mist.
    pub fn blast_on_mud(&mut self, at: glam::Vec3, radius: f32, depth: f32) -> f32 {
        let Some(t) = self.terrain.as_mut() else { return 0.0 };
        let Some(m) = t.mud.as_mut() else { return 0.0 };
        m.blast(t.splat.as_ref(), t.mud_layer, at, radius, depth)
    }

    pub fn walkers_on_mud(&mut self, walkers: &[(usize, glam::Vec3)]) {
        let Some(t) = self.terrain.as_mut() else { return };
        let Some(m) = t.mud.as_mut() else { return };
        let (wading, prints) = m.walk(walkers, t.splat.as_ref(), t.mud_layer);
        if prints > 0 {
            log::info!("mud: {wading} on mud, {prints} new print(s) on a {}x{} map", m.map.width(), m.map.height());
        }
    }

    fn flush_mud(&mut self) {
        if let Some(m) = self.terrain.as_mut().and_then(|t| t.mud.as_mut()) {
            m.flush(&self.queue);
        }
    }

    pub fn sea_line(&self) -> f32 {
        let Some(g) = self.ground.as_ref() else { return -1e9 };
        let (min, max) = (g.heights.min, g.heights.max);
        self.water_bodies
            .iter()
            .filter(|b| {
                let (lo, hi) = b.bounds;
                lo.x < min.x || hi.x > max.x || lo.z < min.y || hi.z > max.y
            })
            .map(|b| b.uniform.extinction[3])
            .fold(-1e9, f32::max)
    }

    fn ensure_ground_map(&mut self) {
        if !self.ground_dirty {
            return;
        }
        self.ground_dirty = false;
        let sea_line = self.sea_line();
        let placement = if let Some(g) = self.ground.as_ref() {
            let started = std::time::Instant::now();
            let (irradiance, sun) = match self.sky_dome.as_ref() {
                Some(s) => (s.irradiance, s.sun),
                None => (
                    self.sky
                        .as_ref().map_or_else(|| super::sky::SkyIrradiance::flat(super::sky::AMBIENT), |s| super::sky::SkyIrradiance { sh: s.sh.map(|v| [v[0] * s.intensity, v[1] * s.intensity, v[2] * s.intensity]) }),
                    None,
                ),
            };
            let map = ground_map::build(
                &ground_map::GroundInputs {
                    heights: &g.heights,
                    sky: &irradiance,
                    sun: sun.as_ref(),
                    sky_occlusion: None,
                    layers: &g.layers,
                    splat: g.splat.as_ref(),
                    settings: &g.settings,
                    sea_line,
                },
                DESKTOP_GROUND_MAP_SIZE,
            );
            let view = ground_map::upload(&self.device, &self.queue, &map);
            let extent = map.max - map.min;
            self.uniform_buf.set_ground_map(view);
            log::info!(
                "desktop ground map: {}x{} over {:.0} x {:.0} m, sea line {:.2}, built in {} ms",
                map.width,
                map.height,
                extent.x,
                extent.y,
                sea_line,
                started.elapsed().as_millis()
            );
            Some(([map.min.x, map.min.y, 1.0 / extent.x, 1.0 / extent.y], map.top))
        } else {
            self.uniform_buf.set_ground_map(uniforms::default_ground_map(&self.device));
            None
        };
        let mut probes = self.uniform_buf.probes();
        probes.set_outdoors(None, None, placement);
        let (probe_view, probe_sampler) = uniforms::default_probe_cube(&self.device);
        self.uniform_buf.rebind_probes(
            &self.device,
            &self.lights_uniform,
            self.shadow_map.sun_depth_view(),
            self.shadow_map.sun_dynamic_depth_view(),
            self.shadow_map.spot_depth_view(),
            self.shadow_map.sampler(),
            &probe_view,
            &probe_sampler,
            probes,
        );
    }

    pub fn set_sky_panorama(
        &mut self,
        pano: Option<&super::sky::Panorama>,
        rotation_deg: f32,
        intensity: f32,
    ) -> Option<(super::sky::SkyIrradiance, Option<super::sky::SkySun>)> {
        // A new photograph: a day running on the old one starts again from it.
        self.day = None;
        self.post.cloud_map = [[0.0; 4]; 3];
        self.sky_dome = pano.map(|p| {
            super::sky::Sky::new(&self.device, &self.queue, &self.sky_pipeline.layout, p, rotation_deg, intensity)
        });
        self.sky_dome.as_ref().map(|s| (s.irradiance, s.sun))
    }

    pub fn has_sky_panorama(&self) -> bool {
        self.sky_dome.is_some()
    }

    pub fn set_raw_wire(&mut self, vertices: Vec<WireVertex>, indices: Vec<u32>) {
        self.raw_wire = (vertices, indices);
    }

    pub fn set_cuboids_double_sided(&mut self) {
        self.solid_pipeline =
            pipeline::SolidPipeline::new_double_sided(&self.device, self.format, &self.uniform_buf.layout);
    }

    #[allow(clippy::too_many_arguments, reason = "GPU setup passes each resource explicitly; bundling them is a separate refactor")]
    pub fn set_terrain(
        &mut self,
        vertices: &[SolidVertex],
        indices: &[u32],
        layers: &[Option<super::terrain_pipeline::TerrainImage>],
        normals: &[Option<super::terrain_pipeline::TerrainImage>],
        rough: &[Option<super::terrain_pipeline::TerrainImage>],
        ao: &[Option<super::terrain_pipeline::TerrainImage>],
        splat: Option<&super::terrain_pipeline::TerrainImage>,
        settings: super::terrain_pipeline::TerrainMaterialUniform,
    ) {
        if vertices.is_empty() || indices.is_empty() {
            self.terrain = None;
            self.rebind_water();
            return;
        }
        let pipeline = super::terrain_pipeline::TerrainPipeline::new(
            &self.device,
            self.format,
            &self.uniform_buf.layout,
        );
        let mud_layer = settings.mud_layer.round() as i32;
        let mud = (mud_layer >= 0).then(|| {
            let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
            for v in vertices {
                lo = [lo[0].min(v.position[0]), lo[1].min(v.position[2])];
                hi = [hi[0].max(v.position[0]), hi[1].max(v.position[2])];
            }
            MudBody::new(&self.device, &self.queue, lo, hi)
        });
        let mut settings = settings;
        if let Some(m) = mud.as_ref() {
            settings.deform_texel = m.texel_metres();
            log::info!("terrain deform map: {}x{} at {:.3} m, mud layer {mud_layer}", m.map.width(), m.map.height(), settings.deform_texel);
        }
        let material = super::terrain_pipeline::TerrainMaterial::from_layers_deformed(
            &self.device,
            &self.queue,
            &pipeline.material_layout,
            layers,
            normals,
            rough,
            ao,
            splat,
            None,
            settings,
            mud.as_ref().map(|m| &m.view),
        );
        let vb = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("terrain_vb"),
            contents: bytemuck::cast_slice(vertices),
            usage: BufferUsages::VERTEX,
        });
        let ib = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("terrain_ib"),
            contents: bytemuck::cast_slice(indices),
            usage: BufferUsages::INDEX,
        });
        self.terrain = Some(DesktopTerrain {
            pipeline,
            material,
            vb,
            ib,
            index_count: indices.len() as u32,
            mud,
            splat: splat.cloned(),
            mud_layer,
        });
        self.rebind_water();
    }

    pub fn clear_terrain(&mut self) {
        self.terrain = None;
        self.rebind_water();
    }

    fn rebind_water(&mut self) {
        let Some(pipeline) = self.water_pipeline.as_ref() else { return };
        let placement = self.terrain.as_ref().and_then(|t| t.mud.as_ref()).map_or([0.0; 4], |m| m.map.placement());
        let view = self.terrain.as_ref().and_then(|t| t.mud.as_ref()).map_or(&pipeline.default_mud, |m| &m.view);
        for body in &mut self.water_bodies {
            body.uniform.prints = placement;
            self.queue.write_buffer(&body.uniform_buffer, 0, bytemuck::bytes_of(&body.uniform));
            body.bind_groups = pipeline.bind_groups_with(&self.device, &body.uniform_buffer, &body.waves, view);
        }
    }

    pub fn update_terrain_splat(&mut self, image: &super::terrain_pipeline::TerrainImage) -> bool {
        match self.terrain.as_ref() {
            Some(t) => t.material.update_splat(&self.queue, image),
            None => false,
        }
    }

    pub fn update_terrain_geometry(&mut self, vertices: &[SolidVertex], indices: &[u32]) {
        let Some(t) = self.terrain.as_mut() else { return };
        t.vb = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("terrain_vb"),
            contents: bytemuck::cast_slice(vertices),
            usage: BufferUsages::VERTEX,
        });
        t.ib = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("terrain_ib"),
            contents: bytemuck::cast_slice(indices),
            usage: BufferUsages::INDEX,
        });
        t.index_count = indices.len() as u32;
    }

    pub fn update_terrain_indices(&mut self, indices: &[u32]) {
        let Some(t) = self.terrain.as_mut() else { return };
        if indices.is_empty() {
            return;
        }
        t.ib = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("terrain_ib"),
            contents: bytemuck::cast_slice(indices),
            usage: BufferUsages::INDEX,
        });
        t.index_count = indices.len() as u32;
    }

    pub fn set_cuboid_lightmap(&mut self, key: &str, rgba: &[u8], width: u32, height: u32) {
        let tex = create_lightmap_texture(&self.device, &self.queue, &self.solid_pipeline.lightmap_layout, rgba, width, height, None);
        self.cuboid_lightmaps.insert(key.to_string(), tex);
    }

    pub fn set_mesh_lightmap(&mut self, key: &str, rgba: &[u8], width: u32, height: u32) {
        let tex = create_lightmap_texture(&self.device, &self.queue, &self.mesh_pipeline.lightmap_layout, rgba, width, height, None);
        self.mesh_lightmaps.insert(key.to_string(), tex);
    }

    pub fn mesh_texture_layout(&self) -> &BindGroupLayout {
        &self.mesh_pipeline.texture_layout
    }

    pub fn mesh_pipeline(&self) -> &mesh_pipeline::MeshPipeline {
        &self.mesh_pipeline
    }

    pub fn create_model_uniform(&self) -> mesh_pipeline::ModelUniform {
        self.mesh_pipeline.create_model_uniform(&self.device)
    }

    pub fn create_skinned_model_uniform(&self) -> mesh_pipeline::ModelUniform {
        self.skinned_mesh_pipeline.create_model_uniform(&self.device)
    }

    pub fn skin_joint_layout(&self) -> &BindGroupLayout {
        &self.skinned_mesh_pipeline.skin_joint_layout
    }

    pub fn create_icon_assets(&self) -> icon::IconAssets {
        icon::IconAssets::new(&self.device, &self.queue, &self.mesh_pipeline.texture_layout)
    }

    pub fn create_panel(
        &self,
        texture_format: TextureFormat,
        width_px: u32,
        height_px: u32,
        width_m: f32,
        height_m: f32,
    ) -> WorldPanel {
        WorldPanel::new(
            &self.device,
            &self.queue,
            texture_format,
            &self.mesh_pipeline,
            width_px,
            height_px,
            width_m,
            height_m,
        )
    }

    pub fn depth_view(&self) -> &TextureView {
        &self.depth_view
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let (t, v) = Self::make_depth(&self.device, width, height);
        self.depth_texture = t;
        self.depth_view = v;
        if let Some(fx) = self.effects.as_mut() {
            fx.depth_moved();
        }
    }

    pub fn render(&mut self, target_view: &TextureView, camera: &Camera, cuboids: &[Cuboid]) {
        self.render_with_meshes(target_view, camera, cuboids, &[]);
    }
    pub fn render_with_meshes(
        &mut self,
        target_view: &TextureView,
        camera: &Camera,
        cuboids: &[Cuboid],
        meshes: &[MeshInstance],
    ) {
        self.render_with_lights(target_view, camera, cuboids, meshes, &[]);
    }

    pub fn render_with_lights(
        &mut self,
        target_view: &TextureView,
        camera: &Camera,
        cuboids: &[Cuboid],
        meshes: &[MeshInstance],
        lights: &[lights::Light],
    ) {
        self.render_internal(target_view, camera, cuboids, meshes, &[], lights, &[]);
    }

    pub fn render_with_overlay(
        &mut self,
        target_view: &TextureView,
        camera: &Camera,
        cuboids: &[Cuboid],
        overlay_cuboids: &[Cuboid],
        meshes: &[MeshInstance],
        lights: &[lights::Light],
    ) {
        self.render_internal(target_view, camera, cuboids, meshes, &[], lights, overlay_cuboids);
    }

    pub fn render_with_panels(
        &mut self,
        target_view: &TextureView,
        camera: &Camera,
        cuboids: &[Cuboid],
        meshes: &[MeshInstance],
        panels: &[&WorldPanel],
        lights: &[lights::Light],
    ) {
        self.render_internal(target_view, camera, cuboids, meshes, panels, lights, &[]);
    }

    #[allow(clippy::type_complexity, reason = "returns the solid and wire vertex batches as one tuple; naming the pair is a separate refactor")]
    fn bake_cuboids(
        &mut self,
        cuboids: &[Cuboid],
    ) -> (
        (Vec<SolidVertex>, Vec<u32>, Vec<(Option<String>, u32, u32)>),
        (Vec<WireVertex>, Vec<u32>),
    ) {
        let mut seen: std::collections::HashSet<u64> =
            std::collections::HashSet::with_capacity(cuboids.len());

        let mut solid_verts: Vec<SolidVertex> = Vec::new();
        let mut solid_indices: Vec<u32> = Vec::new();
        let mut solid_ranges: Vec<(Option<String>, u32, u32)> = Vec::new();
        let mut wire_verts: Vec<WireVertex> = Vec::new();
        let mut wire_indices: Vec<u32> = Vec::new();

        for c in cuboids {
            seen.insert(c.id);
            let snapshot = c.snapshot();

            let needs_rebuild = match self.cuboid_cache.get(&c.id) {
                Some(entry) => entry.snapshot != snapshot,
                None => true,
            };

            if needs_rebuild {
                let entry = CuboidCacheEntry {
                    snapshot,
                    solid: build_solid_mesh_one(c),
                    wire: build_wire_mesh_one(c),
                };
                self.cuboid_cache.insert(c.id, entry);
            }

            let entry = self
                .cuboid_cache
                .get(&c.id)
                .expect("just inserted or already present");

            if let Some((v, i)) = &entry.solid {
                let base = solid_verts.len() as u32;
                let index_start = solid_indices.len() as u32;
                solid_verts.extend_from_slice(v);
                solid_indices.extend(i.iter().map(|x| x + base));
                solid_ranges.push((c.lightmap_key.clone(), index_start, i.len() as u32));
            }
            if let Some((v, i)) = &entry.wire {
                let base = wire_verts.len() as u32;
                wire_verts.extend_from_slice(v);
                wire_indices.extend(i.iter().map(|x| x + base));
            }
        }

        self.cuboid_cache.retain(|id, _| seen.contains(id));

        (
            (solid_verts, solid_indices, solid_ranges),
            (wire_verts, wire_indices),
        )
    }

    fn render_internal(
        &mut self,
        target_view: &TextureView,
        camera: &Camera,
        cuboids: &[Cuboid],
        meshes: &[MeshInstance],
        panels: &[&WorldPanel],
        lights: &[lights::Light],
        overlay_cuboids: &[Cuboid],
    ) {
        self.ensure_ground_map();
        self.flush_mud();
        {
            let v = camera.view();
            let right = glam::Vec3::new(v.x_axis.x, v.y_axis.x, v.z_axis.x).normalize_or_zero();
            let up = glam::Vec3::new(v.x_axis.y, v.y_axis.y, v.z_axis.y).normalize_or_zero();
            let items = std::mem::take(&mut self.particles);
            self.particle_pipeline.upload(&self.device, &self.queue, right, up, &items);
            self.particles = items;
        }
        let vp = camera.projection() * camera.view();

        let ranked = lights::rank_for_budget(lights, lights::MAX_LIGHTS);
        let lights: &[lights::Light] = &ranked;

        let sun = lights.iter().find(|l| l.kind == lights::LightKind::Directional);
        let spot_indices: Vec<usize> = lights
            .iter()
            .enumerate()
            .filter(|(_, l)| l.kind == lights::LightKind::Spot && l.casts_shadow())
            .map(|(i, _)| i)
            .take(shadow::MAX_SPOT_SHADOWS)
            .collect();

        let focus = camera.position + camera.forward() * (self.sun_shadow_radius * 0.5);
        let sun_view_proj = sun
            .map_or(glam::Mat4::IDENTITY, |l| shadow::directional_light_matrix(l.direction, focus, self.sun_shadow_radius));

        let spot_matrices: Vec<shadow::SpotShadowMatrices> = spot_indices
            .iter()
            .map(|&i| shadow::spot_shadow_matrices(&lights[i], self.shadow_map.spot_tile_dim()))
            .collect();
        let mut spot_view_proj = [glam::Mat4::IDENTITY; shadow::SHADOW_MATRICES];
        for (layer, mats) in spot_matrices.iter().enumerate() {
            spot_view_proj[layer] = mats.lookup;
        }

        self.lights_uniform
            .upload_with_shadow_layers(&self.queue, lights, &spot_indices);

        let shadow = ShadowUpload {
            sun_view_proj,
            spot_view_proj,
            sun_enabled: sun.is_some(),
            spot_count: spot_indices.len() as u32,
            sun_dynamic_view_proj: glam::Mat4::IDENTITY,
            sun_dynamic_enabled: false,
            sun_next_view_proj: glam::Mat4::IDENTITY,
            sun_blend: [0.0; 4],
        };
        let sky = self.sky.clone().unwrap_or_else(uniforms::SkyUpload::none);
        self.uniform_buf.upload_scene(
            &self.queue, vp, camera.position, &shadow, &sky, &self.post, &uniforms::PlayerUpload::default(),
        );

        let effects_drawn = match self.effects.as_mut() {
            Some(fx) => {
                // The open sky's light, as the headset's effects take it outdoors.
                let dc = sky.sh[0];
                let ambient = glam::Vec3::new(dc[0], dc[1], dc[2]) * (0.282_095 * sky.intensity);
                fx.simulate(&self.device, &self.queue, camera, self.height, lights, ambient, self.post.exposure)
            }
            None => false,
        };

        let ((solid_verts, solid_indices, solid_ranges), (mut wire_verts, mut wire_indices)) =
            self.bake_cuboids(cuboids);
        let raw_base = wire_verts.len() as u32;
        wire_verts.extend_from_slice(&self.raw_wire.0);
        wire_indices.extend(self.raw_wire.1.iter().map(|i| i + raw_base));

        let solid_vb = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("solid_vb"),
            contents: bytemuck::cast_slice(&solid_verts),
            usage: BufferUsages::VERTEX,
        });
        let solid_ib = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("solid_ib"),
            contents: bytemuck::cast_slice(&solid_indices),
            usage: BufferUsages::INDEX,
        });
        let wire_vb = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("wire_vb"),
            contents: bytemuck::cast_slice(&wire_verts),
            usage: BufferUsages::VERTEX,
        });
        let wire_ib = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("wire_ib"),
            contents: bytemuck::cast_slice(&wire_indices),
            usage: BufferUsages::INDEX,
        });

        let mut panel_buffers: Vec<(Buffer, Buffer)> = Vec::with_capacity(panels.len());
        for panel in panels {
            panel.upload_model(&self.queue);
            let vb = self.device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("panel_vb"),
                contents: bytemuck::cast_slice(panel.vertices()),
                usage: BufferUsages::VERTEX,
            });
            let ib = self.device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("panel_ib"),
                contents: bytemuck::cast_slice(panel.indices()),
                usage: BufferUsages::INDEX,
            });
            panel_buffers.push((vb, ib));
        }

        let mut draws: Vec<(&Buffer, &Buffer, u32, &BindGroup, &BindGroup, &BindGroup)> = Vec::new();

        let mut shadow_draws: Vec<shadow::ShadowMeshDraw> = Vec::new();
        let mut skinned_draws: Vec<(&Buffer, &Buffer, u32, &BindGroup, &BindGroup, &BindGroup)> =
            Vec::new();

        let mut skinned_shadow_draws: Vec<shadow::ShadowSkinnedDraw> = Vec::new();

        for instance in meshes {
            instance
                .model
                .upload(&self.queue, instance.mesh.model_matrix());

            if let Some(skin) = &instance.mesh.skin {
                if let Some(joint_bg) = &skin.joint_bind_group {
                    for prim in &skin.primitives {
                        skinned_draws.push((
                            &prim.vertex_buffer,
                            &prim.index_buffer,
                            prim.indices.len() as u32,
                            &instance.model.bind_group,
                            &prim.texture.bind_group,
                            joint_bg,
                        ));
                        skinned_shadow_draws.push((
                            &prim.vertex_buffer,
                            &prim.index_buffer,
                            prim.indices.len() as u32,
                            &instance.model.bind_group,
                            joint_bg,
                        ));
                    }
                }
            } else {
                let lightmap_bg = instance
                    .lightmap_key
                    .and_then(|k| self.mesh_lightmaps.get(k))
                    .map_or(&self.default_mesh_lightmap.bind_group, |t| &t.bind_group);

                let model = instance.mesh.model_matrix();
                let centre = model.transform_point3(glam::Vec3::ZERO);
                let scale = model.x_axis.length().max(model.y_axis.length()).max(model.z_axis.length());
                let casts = (centre - focus).length() <= self.sun_shadow_radius * 1.5 + instance.mesh.bounding_radius * scale;
                for prim in &instance.mesh.primitives {
                    draws.push((
                        &prim.vertex_buffer,
                        &prim.index_buffer,
                        prim.indices.len() as u32,
                        &instance.model.bind_group,
                        &prim.texture.bind_group,
                        lightmap_bg,
                    ));

                    if casts {
                        shadow_draws.push((
                            &prim.vertex_buffer,
                            &prim.index_buffer,
                            prim.indices.len() as u32,
                            &instance.model.bind_group,
                        ));
                    }
                }
            }
        }

        for (panel, (vb, ib)) in panels.iter().zip(panel_buffers.iter()) {
            draws.push((
                vb,
                ib,
                panel.indices().len() as u32,
                &panel.model.bind_group,
                panel.bind_group(),
                &self.default_mesh_lightmap.bind_group,
            ));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("frame"),
            });

        for body in &self.water_bodies {
            body.waves.update(&self.queue, &mut encoder, self.water_seconds);
        }

        let solid_caster = (!solid_indices.is_empty())
            .then_some((&solid_vb, &solid_ib, solid_indices.len() as u32));

        let ground_caster = self
            .terrain
            .as_ref()
            .filter(|_| self.terrain_visible)
            .map(|t| (&t.vb, &t.ib, t.index_count));
        if shadow.sun_enabled {
            self.shadow_map
                .upload_light(&self.queue, ShadowKind::Sun, shadow.sun_view_proj);
            self.shadow_map
                .record_with_ground(
                    &mut encoder,
                    ShadowKind::Sun,
                    solid_caster,
                    ground_caster,
                    &shadow_draws,
                    &skinned_shadow_draws,
                    shadow.sun_view_proj,
                );
        }

        for layer in 0..shadow.spot_count as usize {

            let pass = spot_matrices[layer].pass;
            self.shadow_map.upload_light(&self.queue, ShadowKind::Spot(layer), pass);
            self.shadow_map.record(
                &mut encoder,
                ShadowKind::Spot(layer),
                solid_caster,
                None,
                &shadow_draws,
                &skinned_shadow_draws,
                &[],
                pass,
            );
        }

        let (overlay_verts, overlay_indices, _) = super::cuboid::build_solid_mesh_with_ranges(overlay_cuboids);
        let overlay_vb = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("overlay_vb"),
            contents: bytemuck::cast_slice(&overlay_verts),
            usage: BufferUsages::VERTEX,
        });
        let overlay_ib = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("overlay_ib"),
            contents: bytemuck::cast_slice(&overlay_indices),
            usage: BufferUsages::INDEX,
        });

        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("3d_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(self.clear_color),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            if !solid_verts.is_empty() {
                pass.set_pipeline(&self.solid_pipeline.pipeline);
                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                pass.set_vertex_buffer(0, solid_vb.slice(..));
                pass.set_index_buffer(solid_ib.slice(..), IndexFormat::Uint32);
                for (lightmap_key, index_start, count) in &solid_ranges {
                    let lightmap_bg = lightmap_key
                        .as_deref()
                        .and_then(|k| self.cuboid_lightmaps.get(k))
                        .map_or(&self.default_cuboid_lightmap.bind_group, |t| &t.bind_group);
                    pass.set_bind_group(1, lightmap_bg, &[]);
                    pass.draw_indexed(*index_start..*index_start + *count, 0, 0..1);
                }
            }

            if let Some(t) = self.terrain.as_ref().filter(|_| self.terrain_visible) {
                pass.set_pipeline(&t.pipeline.pipeline);
                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                pass.set_bind_group(1, &t.material.bind_group, &[]);
                pass.set_vertex_buffer(0, t.vb.slice(..));
                pass.set_index_buffer(t.ib.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..t.index_count, 0, 0..1);
            }

            if !wire_verts.is_empty() {
                pass.set_pipeline(&self.wire_pipeline.pipeline);
                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                pass.set_vertex_buffer(0, wire_vb.slice(..));
                pass.set_index_buffer(wire_ib.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..wire_indices.len() as u32, 0, 0..1);
            }

            if !draws.is_empty() {
                pass.set_pipeline(&self.mesh_pipeline.pipeline);
                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                for (vb, ib, count, model_bg, tex_bg, lightmap_bg) in &draws {
                    pass.set_bind_group(1, *model_bg, &[]);
                    pass.set_bind_group(2, *tex_bg, &[]);
                    pass.set_bind_group(3, *lightmap_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
            }

            if !skinned_draws.is_empty() {
                pass.set_pipeline(&self.skinned_mesh_pipeline.pipeline);
                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                for (vb, ib, count, model_bg, tex_bg, joint_bg) in &skinned_draws {
                    pass.set_bind_group(1, *model_bg, &[]);
                    pass.set_bind_group(2, *tex_bg, &[]);
                    pass.set_bind_group(3, *joint_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
            }

            if let Some(wp) = self.water_pipeline.as_ref().filter(|_| !self.water_bodies.is_empty()) {
                pass.set_pipeline(&wp.pipeline);
                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                for body in &self.water_bodies {
                    pass.set_bind_group(1, &body.bind_groups[body.waves.current()], &[]);
                    pass.set_vertex_buffer(0, body.vertex_buffer.slice(..));
                    pass.set_index_buffer(body.index_buffer.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..body.index_count, 0, 0..1);
                }
            }

            if let Some(sky) = self.sky_dome.as_ref().filter(|_| self.sky_visible) {
                pass.set_pipeline(self.sky_pipeline.pick(sky));
                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                pass.set_bind_group(1, &sky.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            self.particle_pipeline.draw(&mut pass, &self.uniform_buf.bind_group);
        }

        if let (true, Some(fx)) = (effects_drawn, self.effects.as_mut()) {
            // Depth is held read-only so the shader can read it for soft edges.
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("effects_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Load, store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: None,
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            fx.draw(&self.device, &mut pass, &self.uniform_buf.bind_group, &self.depth_texture);
        }

        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("overlay_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Load, store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Load, store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            if !overlay_verts.is_empty() {
                pass.set_pipeline(&self.overlay_pipeline.pipeline);
                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                pass.set_bind_group(1, &self.default_cuboid_lightmap.bind_group, &[]);
                pass.set_vertex_buffer(0, overlay_vb.slice(..));
                pass.set_index_buffer(overlay_ib.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..overlay_indices.len() as u32, 0, 0..1);
            }
        }

        self.queue.submit(Some(encoder.finish()));
    }

    fn make_depth(device: &Device, width: u32, height: u32) -> (Texture, TextureView) {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("depth"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&TextureViewDescriptor::default());
        (tex, view)
    }
}
