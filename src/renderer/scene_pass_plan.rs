#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenePassPlan {

    Direct,

    ViaOffscreen,

    StereoCopy,
}

pub const XR_SCREEN_SPACE_REFLECTIONS: bool = false;

pub fn needs_scene_readback(
    has_mirror: bool,
    reflective_materials: bool,
    screen_space_reflections: bool,
) -> bool {
    has_mirror || (screen_space_reflections && reflective_materials)
}

impl ScenePassPlan {

    pub fn for_frame(needs_readback: bool) -> Self {
        Self::for_frame_with(needs_readback, false)
    }

    pub fn for_frame_with(needs_readback: bool, stereo: bool) -> Self {
        if needs_readback {
            Self::ViaOffscreen
        } else if stereo {
            Self::StereoCopy
        } else {
            Self::Direct
        }
    }

    pub fn samples_scene_back(self) -> bool {
        matches!(self, Self::ViaOffscreen)
    }

    pub fn runs_eye_pass(self) -> bool {
        matches!(self, Self::ViaOffscreen | Self::StereoCopy)
    }
}

#[cfg(test)]
mod scene_pass_plan_tests {
    use super::ScenePassPlan;

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

        assert!(!needs_scene_readback(false, true, false));
        assert_eq!(
            super::ScenePassPlan::for_frame(needs_scene_readback(false, true, false)),
            super::ScenePassPlan::Direct,
        );
    }

    #[test]
    fn a_mirror_still_reads_the_scene_back_whatever_ssr_is_set_to() {
        use super::needs_scene_readback;

        assert!(needs_scene_readback(true, false, false));
        assert!(needs_scene_readback(true, true, false));
    }

    #[test]
    fn with_ssr_on_reflective_materials_still_take_the_offscreen_path() {

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
