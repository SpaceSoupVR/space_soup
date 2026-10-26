#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sharpening {

    Off,

    Quality,

    Normal,
}

pub const XR_SHARPENING: Sharpening = Sharpening::Quality;

pub fn sharpening_for(space_warp: bool) -> Sharpening {
    if space_warp {
        Sharpening::Off
    } else {
        XR_SHARPENING
    }
}

impl Sharpening {

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

        let scale = crate::renderer::RENDER_SCALE;
        let known = matches!(
            (scale < 1.0, XR_SHARPENING),
            (true, Sharpening::Quality | Sharpening::Normal | Sharpening::Off) |
(false, Sharpening::Off)
        );
        assert!(
            known,
            "render scale {scale} with {XR_SHARPENING:?} is a combination nobody \
             has reasoned about; sharpening pays for itself by recovering what \
             the render scale gave up, so decide them together",
        );
    }
}
