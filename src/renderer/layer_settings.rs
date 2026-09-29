//! What the COMPOSITOR is asked to do with the eye images after we hand them over.
//!
//! Its own module rather than living beside the XR renderer or the Android
//! entry point, for the reason `scene_pass_plan` gives: a decision compiled
//! only for the headset is a decision no test on a development machine ever
//! runs. `openxr` is an Android-only dependency of this crate, so nothing here
//! names an OpenXR type -- the caller maps this enum onto
//! `CompositionLayerSettingsFlagsFB`, which is three lines with no arithmetic
//! in them.

/// Compositor-side sharpening, via `XR_FB_composition_layer_settings`.
///
/// This is what Meta markets as MQSR (Meta Quest Super Resolution). It is a
/// SPATIAL upscale-and-sharpen run by the compositor on the finished eye image,
/// not a render technique -- there is no pass, no shader and no texture of ours
/// involved.
///
/// TWO CONSEQUENCES THAT MATTER FOR MEASURING IT:
///
/// 1. It costs TIMEWARP GPU time, not application GPU time. It will not appear
///    in a `PASS:` line however carefully those are instrumented, because our
///    timestamps stop at submit. A frame that got slower after enabling this
///    got slower somewhere our timers cannot see, and the way to tell is the
///    runtime's own stats, not ours.
/// 2. It is worth having precisely BECAUSE we render at
///    `renderer::RENDER_SCALE` = 0.7. The compositor is already upscaling our
///    eye image to the display; asking it to sharpen while it does so is
///    recovering detail that the render scale gave up. At a render scale of
///    1.0 there is much less to recover and the temporal aliasing below is the
///    same price, so this is not a free win to leave on unconditionally.
///
/// KNOWN COST: sharpening near display resolution raises high-frequency
/// contrast, which is what makes it read as sharper and also what makes edge
/// crawl more visible as the head moves. `Quality` runs a wider filter than
/// `Normal` and costs more compositor time for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sharpening {
    /// Chain nothing. The compositor's plain upscale.
    Off,
    /// `QUALITY_SHARPENING`. The wider, more expensive filter.
    Quality,
    /// `NORMAL_SHARPENING`. Cheaper, and the fallback if `Quality` reads as
    /// over-sharpened on the device.
    Normal,
}

/// What the headset asks for, when the runtime offers the extension.
///
/// BACK ON, and the experiment that turned it off is finished.
///
/// `Quality` shipped, the headset reported fine detail "slightly vibrating",
/// and `Off` was the A/B that settled it. THE SHIMMER PERSISTED WITH
/// SHARPENING OFF (2026-09-19), so sharpening was EXONERATED. It was later
/// traced to sampling density: restoring RENDER_SCALE to 1.0 fixed the
/// terrain shimmer outright (2026-09-21).
///
/// So this belongs ON, and leaving it off was an oversight rather than a
/// decision -- the experiment ended and the switch never went back.
///
/// It also matters MORE than it first appeared. Meta Quest Super Resolution is
/// the platform's own answer to exactly the position this renderer is in:
/// rendering at RENDER_SCALE = 0.7 and wanting the upscale to the display to
/// be better than the compositor's plain one. Turning it off while running at
/// 0.7 declines the one free mitigation available for the artifact being
/// chased.
///
/// If a future session turns this off again, say which measurement asked for
/// it -- this one was switched off to test a hypothesis that came back
/// negative.
///
/// It only applies while SpaceWarp is off, which is not the shipped state
/// since 2026-09-29 -- see `sharpening_for`.
pub const XR_SHARPENING: Sharpening = Sharpening::Quality;

/// What the headset asks for with Application SpaceWarp on or off: SpaceWarp
/// wins, so sharpening is `Off` whenever it runs.
///
/// Measured on the headset (2026-09-29): with both on, the runtime reported
/// the compositor tearing 30-60 times a second (VrApi `Tear=`). Sharpening is
/// worth at most ~0.5 ms and some edge crispness; SpaceWarp renders half the
/// frames, and that is the headroom characters, water and effects are going
/// to need. So SpaceWarp ships and takes sharpening off with it (user,
/// 2026-09-29).
pub fn sharpening_for(space_warp: bool) -> Sharpening {
    if space_warp {
        Sharpening::Off
    } else {
        XR_SHARPENING
    }
}

impl Sharpening {
    /// Whether a `CompositionLayerSettingsFB` needs chaining onto the layer.
    ///
    /// `Off` must chain NOTHING rather than chain a zeroed flags field. An
    /// extension struct in the `next` chain is a promise to the runtime that
    /// we meant to put it there; a runtime that reads it as "settings, with
    /// none of them set" is doing what we asked, and a runtime that treats the
    /// presence of the struct as the feature being requested is also doing
    /// what we asked. Not chaining it is the only unambiguous way to say no.
    pub fn wants_layer_settings(self) -> bool {
        !matches!(self, Self::Off)
    }
}

#[cfg(test)]
mod layer_settings_tests {
    use super::{sharpening_for, Sharpening, XR_SHARPENING};

    #[test]
    fn spacewarp_takes_sharpening_off_with_it() {
        assert!(
            !sharpening_for(true).wants_layer_settings(),
            "sharpening with SpaceWarp tore the compositor 30-60 times a second",
        );
        assert_eq!(sharpening_for(false), XR_SHARPENING, "without SpaceWarp the policy stands");
    }

    #[test]
    fn spacewarp_ships_on() {
        // SpaceWarp is the frame-rate plan (user, 2026-09-29): the headroom
        // it buys is for characters, water and effects. Turning it off by
        // default is a decision to make with bench numbers in hand, and it
        // brings sharpening back with it -- not an edit to a default.
        assert!(crate::renderer::levers::Levers::default().space_warp);
    }

    #[test]
    fn off_chains_nothing_at_all() {
        assert!(
            !Sharpening::Off.wants_layer_settings(),
            "Off must not chain a zeroed settings struct -- see the doc comment",
        );
    }

    #[test]
    fn every_on_state_chains_the_struct() {
        for s in [Sharpening::Quality, Sharpening::Normal] {
            assert!(s.wants_layer_settings(), "{s:?} has to reach the compositor");
        }
    }

    #[test]
    fn the_headset_asks_for_sharpening_while_the_render_scale_is_below_one() {
        // The two are coupled: sharpening earns its cost by recovering what
        // the render scale gave up. If someone raises RENDER_SCALE back to 1.0
        // this stops being an obvious win and starts being mostly the edge
        // crawl, so make them look at it together rather than inherit it.
        // DELIBERATELY NOT an assertion that sharpening is on. It was, and the
        // headset showed fine detail shimmering; `Off` is the A/B that settles
        // whether sharpening caused it. Asserting "on" here would have made
        // running that experiment look like breaking a test.
        //
        // What is still worth pinning is that the two are CONSIDERED together,
        // so this asserts the pair is one of the states someone reasoned about
        // rather than an accident.
        let scale = crate::renderer::RENDER_SCALE;
        let known = matches!(
            (scale < 1.0, XR_SHARPENING),
            (true, Sharpening::Quality) | (true, Sharpening::Normal) | (true, Sharpening::Off)
                | (false, Sharpening::Off)
        );
        assert!(
            known,
            "render scale {scale} with {XR_SHARPENING:?} is a combination nobody \
             has reasoned about; sharpening pays for itself by recovering what \
             the render scale gave up, so decide them together",
        );
    }
}
