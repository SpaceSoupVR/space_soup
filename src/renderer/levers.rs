//! RUNTIME LEVERS: switch a renderer feature off -- or force one on -- on the
//! headset, while it runs, to see what it costs and what it does.
//!
//! # Why a file and not a build
//!
//! Every question of the form "is it the trace or the portals that costs the
//! millisecond?" or "does the seam go away without the probe blend?" used to
//! cost a build and a deploy per answer. A lever answers it in the time it
//! takes to push a few bytes:
//!
//! ```text
//! adb -s 2G0YC5ZG7706YV shell "echo '{\"probe_trace\": false}' > \
//!     /sdcard/Android/data/com.example.questapp/files/levers.json"
//! ```
//!
//! The app polls the file about once a second and hands every change to the
//! renderer, and each `PERF` line names the levers it was measured under, so a
//! number is never read against the wrong configuration. Delete the file (or
//! write `{}`) to go back to the shipped state.
//!
//! # One set of switches for people and for the A/B schedule
//!
//! `perf_ab` cycles the same switches on its own: each phase is these levers
//! with exactly one more thing off. So a lever and a phase can never disagree
//! about what "no portals" means, and every feature added here is measurable by
//! the schedule for free.
//!
//! Unknown names are an ERROR, not ignored: a lever misspelt in the middle of a
//! headset session would otherwise measure nothing and look like a finding.
//!
//! # A viewpoint is a lever too
//!
//! `bench` pins the camera to a named place (see `bench`), so the same file
//! that switches a feature off also says where the frame is measured from. A
//! host script walks the views and the A/B schedule with the headset on a
//! desk: `quest_app/bench.py`.

use serde::Deserialize;

/// Every switchable feature. The default is the shipped renderer.
///
/// `Clone`, not `Copy`: the bench viewpoint carries its name.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Levers {
    /// Reflection probes at all. Off: surfaces reflect the sky's harmonics.
    pub probes: bool,
    /// The exact reflection trace (rooms, doorways, proxies). Off: smooth
    /// surfaces fall back to the box projection.
    pub probe_trace: bool,
    /// What stands inside rooms -- the pillar, the lamps -- in the trace. Off:
    /// reflections pass through them to the walls.
    pub reflection_proxies: bool,
    /// The two-photograph blend across a room's cells. Off: every resident
    /// probe is its own room, so no fragment reads two.
    pub probe_blend: bool,
    /// Doorway portals: the blend across an opening, and the trace crossing
    /// into the next room.
    pub portals: bool,
    /// Every shadow pass and every shadow tap. Lights still shine, unshadowed.
    pub shadows: bool,
    /// The moving-objects sun map. The static map and the baked mask stay.
    pub sun_dynamic: bool,
    /// The player's crisp shadows from the lamps lighting them most: a
    /// characters-only tile of the shadow atlas each. Off, those lamps shadow
    /// the player by the capsules alone. See `shadow::MAX_CHARACTER_SHADOWS`.
    pub character_shadows: bool,
    /// The characters as capsules: their soft shadows, contact darkening and
    /// reflections. Off, the shaders see no characters (and the characters'
    /// shadow tiles go with them). See `uniforms::CapsuleUpload`.
    pub capsules: bool,
    /// The live light loop, the sky's sun included. Baked light and probes stay.
    pub direct_lights: bool,
    /// The STATIONARY lamps -- shaded live, shadowed from their baked masks.
    /// MEASUREMENT: off, their light is in no lightmap, so it is simply gone.
    pub stationary_lights: bool,
    /// Eye adaptation. Off: exposure is pinned at the scene's post setting.
    pub eye_adaptation: bool,
    /// The light loop skips a lamp that cannot reach the pixel -- past its
    /// range, or behind a wall by its baked mask -- before any of its maths.
    /// Lossless: off changes no pixel, only what the frame costs.
    pub light_culling: bool,
    /// Brush reflections from the half-resolution probe pass rather than
    /// traced per pixel. See `brush_pipeline::probe_pass`. Mono scene passes.
    pub half_res_reflections: bool,
    /// The brushes' depth drawn first in the scene pass, so nothing hidden
    /// behind a wall or under a floor is shaded. See
    /// `BrushPipeline::new_depth_prepass`. Lossless: off changes no pixel.
    pub depth_prepass: bool,
    /// From inside a closed room, whatever lies outside the building -- the
    /// terrain -- is drawn only where a doorway shows it. See `portal_cull`.
    /// Lossless for a closed room: off changes no pixel.
    pub portal_culling: bool,
    /// The single-eye probe pass leaves a traced hit's secondary lookups to
    /// a compute pass over just the texels that need them. See
    /// `probe_fixup`. Lossless: off changes no pixel.
    pub deferred_reflection_lookups: bool,
    /// The ground's probe reflection made in the half-resolution probe pass
    /// and read back, as the brushes' is, instead of traced per pixel in the
    /// scene pass. See `TerrainPipeline::new_probe_pass`. Single eye, with
    /// `deferred_reflection_lookups`.
    pub terrain_probe_pass: bool,
    /// Reflections that leave the building meet the ground: the trace over
    /// the ground map's heights. Off: they see only the sky. See
    /// `ground_map::trace`.
    pub ground_trace: bool,
    /// Fixed foveated rendering: how coarsely the edges of each eye's image
    /// are shaded, where the headset has it. See `foveation`.
    pub foveation: crate::renderer::foveation::FoveationLevel,
    /// APPLICATION SPACEWARP: motion vectors and depth with every frame, so
    /// the compositor can make every other one. ON by default since
    /// 2026-09-29, once its black frames were fixed: 36 frames a second are
    /// rendered, a 27.8 ms budget. Meshes carry their own motion (previous
    /// model, previous joints); cuboids, particles, water waves, reflections
    /// and highlights move only with the camera. It also turns sharpening off
    /// -- see `layer_settings::sharpening_for`. See `space_warp`.
    pub space_warp: bool,
    /// DIAGNOSIS ONLY: SpaceWarp variants, as bits, to find what makes the
    /// compositor's frames black (headset, 2026-09-29) without a rebuild per
    /// guess. 1 = the motion pass draws nothing (cleared vectors and depth);
    /// 2 = vectors written y DOWN (Vulkan's NDC; shipped is y up); 4 =
    /// vectors zeroed (depth still drawn);
    /// 8 = `appSpaceDeltaPose` the identity; 16 = `farZ` infinite; 64 = no
    /// layer settings (sharpening) on the projection layer; 128 = views drawn
    /// and submitted 5 cm right of the head, 256 = swaying +-5 cm -- head
    /// motion for the compositor to correct, on a headset lying on a desk;
    /// 512 = depth declared reversed; 1024 = depth declared 500 m .. 1 km;
    /// 2048 = the depth image's stencil stored rather than discarded;
    /// 4096 = the depth image cleared to 0 rather than 1;
    /// 16384 = reflections move with their surface, as before 2026-09-29
    /// (`space_warp`'s reflections section), to compare;
    /// 32768 = the brushes write what the reflection motion READ instead of
    /// motion -- reflected share, reach in metres, distance ratio;
    /// 65536 = with 8192, the images also saved raw to the app's files
    /// (`swdump.bin`, see `space_warp::Readback::read`).
    pub space_warp_debug: u32,
    /// DIAGNOSIS: both eyes' finished images of the next frame, raw, to the
    /// app's files as `eyecapture_<n>.bin`, each time this changes to a new
    /// nonzero `n` -- to see whether something differs between the eyes,
    /// which no screenshot shows (the system's is one view). Needs the
    /// `debug.spacesoup.eyecapture` property set to 1 before the app starts;
    /// see `xr::vulkan::eye_capture_enabled`. `quest_app/bench.py
    /// --eye-capture` drives it. Captures only: it changes nothing drawn, so
    /// it is not in `summary`.
    pub eye_capture: u32,
    /// The lamps' veils: each light source's glare, drawn where an HDR
    /// framebuffer would have bloomed it. See `glare`.
    pub glare: bool,
    /// How strong the veils are against `glare::VEIL_SHARE` of the CIE young
    /// eye's: 1 as shipped.
    pub glare_strength: f32,
    /// The characters mirrored in the floor the player stands on, in place of
    /// their capsules there. See `brush_pipeline::probe_pass::MIRROR_FORMAT`.
    /// OFF as shipped: its pass and blur levels cost about 1 ms at GPU level 5
    /// wherever a mirrored body could be in view -- which, with the Quest's
    /// tall field of view, is most views (2026-09-30). On once they cost less.
    pub floor_mirror: bool,
    /// Metres past which the terrain's layer normal maps fade out; 0 keeps
    /// them everywhere. See `terrain_pipeline` (`post_params.z`).
    pub terrain_detail_distance: f32,
    /// MEASUREMENT: block on the GPU at the end of every frame, as the
    /// renderer used to. Its wait is then exactly the GPU's time, which is
    /// what the A/B schedule attributes costs with; shipped, the CPU prepares
    /// the next frame while the GPU draws this one.
    pub gpu_sync: bool,
    /// Screen-space reflections. `None` leaves the in-headset switch alone.
    pub ssr: Option<bool>,
    /// Both eyes in one multiview scene pass. `None` leaves it as set.
    pub multiview: Option<bool>,
    /// MEASUREMENT ONLY -- the scene pass shades a quarter of the pixels. If
    /// the scene cost drops toward a quarter, the frame is fill-bound.
    pub half_viewport: bool,
    /// MEASUREMENT ONLY -- straight to the swapchain: no offscreen copy, no eye
    /// pass. Reflections that need the copy break.
    pub direct_path: bool,
    /// Cycle `perf_ab`'s phases, one per `PERF` window, on top of these levers.
    pub ab_cycle: bool,
    /// MEASUREMENT: the camera pinned to a named viewpoint instead of the
    /// head. The app moves the rig and the renderer pins the tracked head;
    /// see `bench`.
    pub bench: Option<crate::renderer::bench::BenchPose>,
}

impl Default for Levers {
    fn default() -> Self {
        Self {
            probes: true,
            probe_trace: true,
            reflection_proxies: true,
            probe_blend: true,
            portals: true,
            shadows: true,
            sun_dynamic: true,
            character_shadows: true,
            capsules: true,
            direct_lights: true,
            stationary_lights: true,
            eye_adaptation: true,
            light_culling: true,
            half_res_reflections: true,
            depth_prepass: true,
            portal_culling: true,
            deferred_reflection_lookups: true,
            terrain_probe_pass: true,
            ground_trace: true,
            foveation: crate::renderer::foveation::SHIPPED,
            space_warp: true,
            space_warp_debug: 0,
            eye_capture: 0,
            glare: true,
            glare_strength: 1.0,
            floor_mirror: false,
            terrain_detail_distance: 0.0,
            gpu_sync: false,
            ssr: None,
            multiview: None,
            half_viewport: false,
            direct_path: false,
            ab_cycle: false,
            bench: None,
        }
    }
}

impl Levers {
    /// Parse a lever file. `{}` -- or an empty file -- is the shipped state.
    pub fn parse(text: &str) -> Result<Self, String> {
        if text.trim().is_empty() {
            return Ok(Self::default());
        }
        let levers: Self = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if let Some(problem) = levers.bench.as_ref().and_then(|b| b.problem()) {
            return Err(problem);
        }
        Ok(levers)
    }

    /// These levers with the `perf_ab` phase's one extra switch applied.
    pub fn with_phase(self, phase: crate::renderer::perf_ab::Phase) -> Self {
        use crate::renderer::perf_ab::Phase;
        let mut l = self;
        match phase {
            Phase::Baseline => {}
            Phase::HalfViewport => l.half_viewport = true,
            Phase::DirectPath => l.direct_path = true,
            Phase::NoProbes => l.probes = false,
            Phase::NoShadows => l.shadows = false,
            Phase::NoSunDynamic => l.sun_dynamic = false,
            Phase::NoProbeBlend => l.probe_blend = false,
            Phase::NoPortals => l.portals = false,
            Phase::NoDirectLights => l.direct_lights = false,
            Phase::NoProbeTrace => l.probe_trace = false,
            Phase::NoProxies => l.reflection_proxies = false,
            Phase::NoStationary => l.stationary_lights = false,
            Phase::NoLightCulling => l.light_culling = false,
            Phase::FullResReflections => l.half_res_reflections = false,
            Phase::NoDepthPrepass => l.depth_prepass = false,
            Phase::NoPortalCulling => l.portal_culling = false,
            Phase::InlineReflectionLookups => l.deferred_reflection_lookups = false,
            Phase::TerrainPerPixelReflections => l.terrain_probe_pass = false,
            Phase::NoGroundTrace => l.ground_trace = false,
            Phase::NoFoveation => l.foveation = crate::renderer::foveation::FoveationLevel::Off,
            Phase::NoGlare => l.glare = false,
            Phase::FloorMirror => l.floor_mirror = true,
        }
        l
    }

    /// What differs from the shipped state, for the `PERF` line: `-` when
    /// nothing does.
    pub fn summary(&self) -> String {
        let d = Self::default();
        let mut out: Vec<String> = Vec::new();
        let mut flag = |name: &str, on: bool, default: bool| {
            if on != default {
                out.push(if on { name.to_string() } else { format!("no_{name}") });
            }
        };
        flag("probes", self.probes, d.probes);
        flag("probe_trace", self.probe_trace, d.probe_trace);
        flag("proxies", self.reflection_proxies, d.reflection_proxies);
        flag("probe_blend", self.probe_blend, d.probe_blend);
        flag("portals", self.portals, d.portals);
        flag("shadows", self.shadows, d.shadows);
        flag("sun_dynamic", self.sun_dynamic, d.sun_dynamic);
        flag("character_shadows", self.character_shadows, d.character_shadows);
        flag("capsules", self.capsules, d.capsules);
        flag("direct_lights", self.direct_lights, d.direct_lights);
        flag("stationary", self.stationary_lights, d.stationary_lights);
        flag("eye_adaptation", self.eye_adaptation, d.eye_adaptation);
        flag("light_culling", self.light_culling, d.light_culling);
        flag("half_res_reflections", self.half_res_reflections, d.half_res_reflections);
        flag("depth_prepass", self.depth_prepass, d.depth_prepass);
        flag("portal_culling", self.portal_culling, d.portal_culling);
        flag("deferred_lookups", self.deferred_reflection_lookups, d.deferred_reflection_lookups);
        flag("terrain_probe_pass", self.terrain_probe_pass, d.terrain_probe_pass);
        flag("ground_trace", self.ground_trace, d.ground_trace);
        flag("space_warp", self.space_warp, d.space_warp);
        flag("glare", self.glare, d.glare);
        flag("floor_mirror", self.floor_mirror, d.floor_mirror);
        flag("gpu_sync", self.gpu_sync, d.gpu_sync);
        flag("half_viewport", self.half_viewport, d.half_viewport);
        flag("direct_path", self.direct_path, d.direct_path);
        flag("ab_cycle", self.ab_cycle, d.ab_cycle);
        if self.foveation != d.foveation {
            out.push(format!("foveation={}", self.foveation.label()));
        }
        if self.space_warp_debug != 0 {
            out.push(format!("swdbg={}", self.space_warp_debug));
        }
        if self.glare && self.glare_strength != d.glare_strength {
            out.push(format!("glare_strength={}", self.glare_strength));
        }
        if self.terrain_detail_distance != d.terrain_detail_distance {
            out.push(format!("terrain_detail={}", self.terrain_detail_distance));
        }
        if let Some(on) = self.ssr {
            out.push(format!("ssr={}", if on { "on" } else { "off" }));
        }
        if let Some(on) = self.multiview {
            out.push(format!("multiview={}", if on { "on" } else { "off" }));
        }
        if let Some(b) = &self.bench {
            out.push(format!("bench={}", b.name));
        }
        if out.is_empty() {
            "-".to_string()
        } else {
            out.join(",")
        }
    }
}

/// A lever file, re-read when it changes.
pub struct LeverFile {
    path: std::path::PathBuf,
    /// The modification time and length last read; `None` before the first
    /// read and while the file does not exist.
    seen: Option<(std::time::SystemTime, u64)>,
}

impl LeverFile {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into(), seen: None }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// `Some` when the file changed since the last call: the levers it now
    /// holds, or why it could not be read. A file that disappears reads as the
    /// shipped state, so deleting it undoes every lever.
    pub fn poll(&mut self) -> Option<Result<Levers, String>> {
        match std::fs::metadata(&self.path) {
            Ok(m) => {
                let stamp = (m.modified().unwrap_or(std::time::UNIX_EPOCH), m.len());
                if self.seen == Some(stamp) {
                    return None;
                }
                self.seen = Some(stamp);
                Some(std::fs::read_to_string(&self.path).map_err(|e| e.to_string()).and_then(|t| Levers::parse(&t)))
            }
            Err(_) => {
                if self.seen.take().is_some() {
                    Some(Ok(Levers::default()))
                } else {
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::perf_ab::Phase;

    #[test]
    fn an_empty_file_is_the_shipped_renderer() {
        assert_eq!(Levers::parse("").unwrap(), Levers::default());
        assert_eq!(Levers::parse("{}").unwrap(), Levers::default());
        assert_eq!(Levers::default().summary(), "-");
    }

    #[test]
    fn a_lever_changes_only_itself() {
        let l = Levers::parse(r#"{"probe_trace": false, "ssr": true}"#).unwrap();
        assert!(!l.probe_trace && l.ssr == Some(true));
        assert_eq!(Levers { probe_trace: true, ssr: None, ..l.clone() }, Levers::default());
        assert_eq!(l.summary(), "no_probe_trace,ssr=on");
    }

    /// A misspelt lever measures nothing and would look like a finding.
    #[test]
    fn an_unknown_lever_is_an_error() {
        assert!(Levers::parse(r#"{"probe_tarce": false}"#).is_err());
    }

    /// The viewpoint rides in the same file and is named on every line.
    #[test]
    fn a_bench_view_is_read_and_named() {
        let l = Levers::parse(r#"{"bench": {"name": "hall_back", "eye": [-1.3, 1.6, -14.5], "at": [0.4, 2.4, 3.7]}, "ab_cycle": true}"#)
            .unwrap();
        let b = l.bench.as_ref().unwrap();
        assert_eq!((b.name.as_str(), b.eye, b.at), ("hall_back", [-1.3, 1.6, -14.5], [0.4, 2.4, 3.7]));
        assert_eq!(l.summary(), "ab_cycle,bench=hall_back");
    }

    /// A view that cannot be pinned -- a misspelt field, a name that would
    /// split the summary, no direction -- is refused like a misspelt lever.
    #[test]
    fn a_bench_view_that_cannot_be_used_is_an_error() {
        for bad in [
            r#"{"bench": {"name": "a", "eye": [0, 1, 0], "look_at": [0, 1, -1]}}"#,
            r#"{"bench": {"name": "a,b", "eye": [0, 1, 0], "at": [0, 1, -1]}}"#,
            r#"{"bench": {"name": "a", "eye": [0, 1, 0], "at": [0, 1, 0]}}"#,
            r#"{"bench": {"name": "a", "eye": [0, 1], "at": [0, 1, -1]}}"#,
        ] {
            assert!(Levers::parse(bad).is_err(), "accepted {bad}");
        }
    }

    /// Each phase is the levers with exactly one more thing switched, so the
    /// schedule and a person flipping the same lever measure the same thing.
    #[test]
    fn each_phase_is_one_lever() {
        for p in Phase::ALL {
            let l = Levers::default().with_phase(p);
            let changed = l.summary();
            if p == Phase::Baseline {
                assert_eq!(changed, "-");
            } else {
                assert!(!changed.contains(','), "{p:?} switched more than one lever: {changed}");
                assert_ne!(changed, "-", "{p:?} switched nothing");
            }
        }
    }

    #[test]
    fn the_file_is_reread_when_it_changes_and_forgotten_when_deleted() {
        let dir = std::env::temp_dir().join(format!("levers_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("levers.json");
        let _ = std::fs::remove_file(&path);
        let mut f = LeverFile::new(&path);
        assert!(f.poll().is_none(), "no file, nothing to report");
        std::fs::write(&path, r#"{"portals": false}"#).unwrap();
        assert_eq!(f.poll().unwrap().unwrap().portals, false);
        assert!(f.poll().is_none(), "unchanged, nothing to report");
        std::fs::write(&path, r#"{"portals": false, "shadows": false}"#).unwrap();
        let l = f.poll().unwrap().unwrap();
        assert!(!l.portals && !l.shadows);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(f.poll().unwrap().unwrap(), Levers::default(), "deleting the file undoes every lever");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
