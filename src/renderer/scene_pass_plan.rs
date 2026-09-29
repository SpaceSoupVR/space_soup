//! Whether a frame's scene pass writes straight into the swapchain.
//!
//! Its own module rather than living beside the XR renderer because that
//! renderer is `#[cfg(target_os = "android")]`: a decision compiled only for
//! the headset is a decision no test on a development machine ever runs, and
//! this one is pure arithmetic on a bool with nothing platform-specific in it.
/// Where a frame's scene pixels go, and therefore whether a second pass runs.
///
/// The scene used to be rendered into a private texture and then blitted into
/// the swapchain image by an eye pass. That offscreen copy exists for exactly
/// one reason: reflective solids and the mirror quad SAMPLE the finished scene,
/// and a pass cannot read the attachment it is writing. Nothing else wants it,
/// and no level shipped so far has either feature -- so in practice every frame
/// paid for a full-resolution store out of tile memory plus a second render
/// pass that read all of it back and wrote it again, per eye.
///
/// On a tile GPU the render pass is the expensive unit; this renderer measured
/// that once already, when collapsing three shadow passes into one atlas pass
/// returned more than culling 95% of the shadow geometry did.
///
/// It also gates foveation. `XR_FB_foveation` hangs a fragment density map on
/// the SWAPCHAIN image, so shading into a private texture and blitting leaves
/// the density map covering only the blit while the pass doing the real work
/// runs at full rate. Drawing into the swapchain directly is the prerequisite
/// for FFR ever being worth enabling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenePassPlan {
    /// Straight into the swapchain image. No eye pass follows.
    Direct,
    /// Into the offscreen target, because something will sample it back.
    ViaOffscreen,
    /// Both eyes in one MULTIVIEW pass, which draws into its own layered
    /// target, never the swapchain: the eye pass must copy each eye across,
    /// though nothing samples the scene back. Until 2026-09-29 this was
    /// `Direct` whenever SSR was off, so the copy never ran and the headset
    /// kept showing the last picture it had -- "multiview freezes the screen",
    /// working only while SSR was on.
    StereoCopy,
}

/// Whether screen-space reflections run on the headset.
///
/// OFF, measured rather than assumed. SSR is what forces the offscreen scene
/// copy for reflective materials, and on Quest 3 that path cost 20.1 ms of a
/// 60.6 ms frame (A/B, `perf_ab::Phase::DirectPath`): the eye pass redraws the
/// level's whole brush geometry to composite the reflection, the multisampled
/// depth has to be stored, and a mip chain is rebuilt per eye per frame. It is
/// also where the black band and black triangle on reflective walls came from
/// -- the screen-space frontier where rays leave the image. Meta's own Quest
/// guidance is to turn SSR off and reflect with probes, which this renderer
/// already bakes.
///
/// A named switch rather than deleted code, so a device with headroom can turn
/// it back on, and so the choice can later be surfaced as a quality setting.
/// A MIRROR is unaffected: a planar reflection is not screen-space and keeps
/// its own readback.
pub const XR_SCREEN_SPACE_REFLECTIONS: bool = false;

/// Whether this frame must render the scene into the offscreen copy.
///
/// One place for the rule, so the renderer and its test cannot disagree.
pub fn needs_scene_readback(
    has_mirror: bool,
    reflective_materials: bool,
    screen_space_reflections: bool,
) -> bool {
    has_mirror || (screen_space_reflections && reflective_materials)
}

impl ScenePassPlan {
    /// `needs_readback` is "something in this frame samples the rendered scene".
    pub fn for_frame(needs_readback: bool) -> Self {
        Self::for_frame_with(needs_readback, false)
    }

    /// [`Self::for_frame`], knowing whether the scene pass is the two-eye
    /// multiview one.
    pub fn for_frame_with(needs_readback: bool, stereo: bool) -> Self {
        if needs_readback {
            Self::ViaOffscreen
        } else if stereo {
            Self::StereoCopy
        } else {
            Self::Direct
        }
    }

    /// Whether the scene pass must resolve into the offscreen colour target.
    pub fn samples_scene_back(self) -> bool {
        matches!(self, Self::ViaOffscreen)
    }

    /// Whether the eye pass -- blit, reflective solids, mirror -- runs at all.
    ///
    /// The SAME condition as `samples_scene_back`, deliberately, and expressed
    /// as its own question so both call sites read as what they mean. They must
    /// never disagree: an eye pass that runs after a direct scene pass blits a
    /// stale offscreen texture over the frame that was just drawn.
    pub fn runs_eye_pass(self) -> bool {
        matches!(self, Self::ViaOffscreen | Self::StereoCopy)
    }
}

#[cfg(test)]
mod scene_pass_plan_tests {
    use super::ScenePassPlan;

    /// A multiview frame draws into its own layered target, so the eye pass
    /// must copy both eyes into the swapchain even with nothing to sample.
    #[test]
    fn a_multiview_frame_copies_its_eyes_across() {
        let plan = ScenePassPlan::for_frame_with(false, true);
        assert_eq!(plan, ScenePassPlan::StereoCopy);
        assert!(plan.runs_eye_pass(), "the multiview picture never reaches the swapchain");
        assert!(!plan.samples_scene_back(), "nothing samples it: no depth copy, no mips");
        assert_eq!(ScenePassPlan::for_frame_with(true, true), ScenePassPlan::ViaOffscreen);
    }

    #[test]
    fn a_plain_frame_renders_straight_into_the_swapchain() {
        let plan = ScenePassPlan::for_frame(false);
        assert_eq!(plan, ScenePassPlan::Direct);
        assert!(!plan.samples_scene_back());
        assert!(!plan.runs_eye_pass(), "nothing may overwrite a direct render");
    }

    #[test]
    fn a_frame_with_a_mirror_or_reflective_solid_keeps_the_offscreen_copy() {
        let plan = ScenePassPlan::for_frame(true);
        assert_eq!(plan, ScenePassPlan::ViaOffscreen);
        assert!(plan.samples_scene_back());
        assert!(plan.runs_eye_pass(), "the reflective draws live in the eye pass");
    }

    #[test]
    fn the_eye_pass_runs_exactly_when_the_scene_went_offscreen() {
        // The coupling itself, not either half. Rendering direct and then
        // running the eye pass blits a texture this frame never wrote, over
        // the top of the scene that was just drawn -- a black or frozen eye.
        // Rendering offscreen and skipping the eye pass never gets the image
        // into the swapchain at all.
        for needs_readback in [false, true] {
            let plan = ScenePassPlan::for_frame(needs_readback);
            assert_eq!(
                plan.runs_eye_pass(),
                plan.samples_scene_back(),
                "the two halves disagreed for needs_readback={needs_readback}",
            );
        }
    }

    #[test]
    fn with_ssr_off_reflective_materials_no_longer_force_the_offscreen_copy() {
        use super::needs_scene_readback;
        // The 20 ms. A polished floor alone must now take the direct path.
        assert!(!needs_scene_readback(false, true, false));
        assert_eq!(
            super::ScenePassPlan::for_frame(needs_scene_readback(false, true, false)),
            super::ScenePassPlan::Direct,
        );
    }

    #[test]
    fn a_mirror_still_reads_the_scene_back_whatever_ssr_is_set_to() {
        use super::needs_scene_readback;
        // A planar mirror is not screen-space; switching SSR off must not
        // quietly break the one feature that still needs the copy.
        assert!(needs_scene_readback(true, false, false));
        assert!(needs_scene_readback(true, true, false));
    }

    #[test]
    fn with_ssr_on_reflective_materials_still_take_the_offscreen_path() {
        // The switch must actually control something: turned back on, the old
        // behaviour returns.
        assert!(super::needs_scene_readback(false, true, true));
        assert!(!super::needs_scene_readback(false, false, true));
    }

    #[test]
    fn ssr_is_off_on_the_headset() {
        assert!(
            !super::XR_SCREEN_SPACE_REFLECTIONS,
            "XR_SCREEN_SPACE_REFLECTIONS is on: reflective materials force the \\
             offscreen path again, which measured 20 ms per frame on Quest 3.",
        );
    }

}
