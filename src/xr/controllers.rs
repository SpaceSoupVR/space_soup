//! Quest Touch controller input — poses, triggers, squeeze, thumbsticks, buttons.

use log::info;
use openxr as xr;

pub struct ControllerState {
    pub l_trigger:     f32,
    pub r_trigger:     f32,
    pub l_squeeze:     f32,
    pub r_squeeze:     f32,
    pub l_stick:       xr::Vector2f,
    pub r_stick:       xr::Vector2f,
    pub l_stick_click: bool,
    pub r_stick_click: bool,
    pub btn_a:         bool,
    pub btn_b:         bool,
    pub btn_x:         bool,
    pub btn_y:         bool,
    pub btn_menu:      bool,
    pub l_grip_pose:   Option<xr::Posef>,
    pub r_grip_pose:   Option<xr::Posef>,
    pub l_aim_pose:    Option<xr::Posef>,
    pub r_aim_pose:    Option<xr::Posef>,
}

pub struct Controllers {
    action_set:             xr::ActionSet,
    left_trigger:           xr::Action<f32>,
    right_trigger:          xr::Action<f32>,
    left_squeeze:           xr::Action<f32>,
    right_squeeze:          xr::Action<f32>,
    left_thumbstick:        xr::Action<xr::Vector2f>,
    right_thumbstick:       xr::Action<xr::Vector2f>,
    left_thumbstick_click:  xr::Action<bool>,
    right_thumbstick_click: xr::Action<bool>,
    a_button:               xr::Action<bool>,
    b_button:               xr::Action<bool>,
    x_button:               xr::Action<bool>,
    y_button:               xr::Action<bool>,
    menu_button:            xr::Action<bool>,
    left_grip_space:        xr::Space,
    right_grip_space:       xr::Space,
    left_aim_space:         xr::Space,
    right_aim_space:        xr::Space,
    pub state:              ControllerState,
}

impl Controllers {
    pub fn new(
        instance: &xr::Instance,
        session:  &xr::Session<xr::Vulkan>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let action_set = instance.create_action_set("input", "Input", 0)?;

        let left_grip_pose:  xr::Action<xr::Posef> = action_set.create_action("left_grip_pose",  "Left Grip Pose",  &[])?;
        let right_grip_pose: xr::Action<xr::Posef> = action_set.create_action("right_grip_pose", "Right Grip Pose", &[])?;
        let left_aim_pose:   xr::Action<xr::Posef> = action_set.create_action("left_aim_pose",   "Left Aim Pose",   &[])?;
        let right_aim_pose:  xr::Action<xr::Posef> = action_set.create_action("right_aim_pose",  "Right Aim Pose",  &[])?;

        let left_trigger:           xr::Action<f32>          = action_set.create_action("left_trigger",           "Left Trigger",           &[])?;
        let right_trigger:          xr::Action<f32>          = action_set.create_action("right_trigger",          "Right Trigger",          &[])?;
        let left_squeeze:           xr::Action<f32>          = action_set.create_action("left_squeeze",           "Left Squeeze",           &[])?;
        let right_squeeze:          xr::Action<f32>          = action_set.create_action("right_squeeze",          "Right Squeeze",          &[])?;
        let left_thumbstick:        xr::Action<xr::Vector2f> = action_set.create_action("left_thumbstick",        "Left Thumbstick",        &[])?;
        let right_thumbstick:       xr::Action<xr::Vector2f> = action_set.create_action("right_thumbstick",       "Right Thumbstick",       &[])?;
        let left_thumbstick_click:  xr::Action<bool>         = action_set.create_action("left_thumbstick_click",  "Left Thumbstick Click",  &[])?;
        let right_thumbstick_click: xr::Action<bool>         = action_set.create_action("right_thumbstick_click", "Right Thumbstick Click", &[])?;
        let a_button:               xr::Action<bool>         = action_set.create_action("a_button",               "A Button",               &[])?;
        let b_button:               xr::Action<bool>         = action_set.create_action("b_button",               "B Button",               &[])?;
        let x_button:               xr::Action<bool>         = action_set.create_action("x_button",               "X Button",               &[])?;
        let y_button:               xr::Action<bool>         = action_set.create_action("y_button",               "Y Button",               &[])?;
        let menu_button:            xr::Action<bool>         = action_set.create_action("menu_button",            "Menu Button",            &[])?;

        let profile = instance.string_to_path("/interaction_profiles/oculus/touch_controller")?;
        instance.suggest_interaction_profile_bindings(profile, &[
            xr::Binding::new(&left_grip_pose,         instance.string_to_path("/user/hand/left/input/grip/pose")?),
            xr::Binding::new(&right_grip_pose,        instance.string_to_path("/user/hand/right/input/grip/pose")?),
            xr::Binding::new(&left_aim_pose,          instance.string_to_path("/user/hand/left/input/aim/pose")?),
            xr::Binding::new(&right_aim_pose,         instance.string_to_path("/user/hand/right/input/aim/pose")?),
            xr::Binding::new(&left_trigger,           instance.string_to_path("/user/hand/left/input/trigger/value")?),
            xr::Binding::new(&right_trigger,          instance.string_to_path("/user/hand/right/input/trigger/value")?),
            xr::Binding::new(&left_squeeze,           instance.string_to_path("/user/hand/left/input/squeeze/value")?),
            xr::Binding::new(&right_squeeze,          instance.string_to_path("/user/hand/right/input/squeeze/value")?),
            xr::Binding::new(&left_thumbstick,        instance.string_to_path("/user/hand/left/input/thumbstick")?),
            xr::Binding::new(&right_thumbstick,       instance.string_to_path("/user/hand/right/input/thumbstick")?),
            xr::Binding::new(&left_thumbstick_click,  instance.string_to_path("/user/hand/left/input/thumbstick/click")?),
            xr::Binding::new(&right_thumbstick_click, instance.string_to_path("/user/hand/right/input/thumbstick/click")?),
            xr::Binding::new(&a_button,               instance.string_to_path("/user/hand/right/input/a/click")?),
            xr::Binding::new(&b_button,               instance.string_to_path("/user/hand/right/input/b/click")?),
            xr::Binding::new(&x_button,               instance.string_to_path("/user/hand/left/input/x/click")?),
            xr::Binding::new(&y_button,               instance.string_to_path("/user/hand/left/input/y/click")?),
            xr::Binding::new(&menu_button,            instance.string_to_path("/user/hand/left/input/menu/click")?),
        ])?;

        session.attach_action_sets(&[&action_set])?;

        let id = xr::Posef::IDENTITY;
        let left_grip_space  = left_grip_pose.create_space(session.clone(), xr::Path::NULL, id)?;
        let right_grip_space = right_grip_pose.create_space(session.clone(), xr::Path::NULL, id)?;
        let left_aim_space   = left_aim_pose.create_space(session.clone(), xr::Path::NULL, id)?;
        let right_aim_space  = right_aim_pose.create_space(session.clone(), xr::Path::NULL, id)?;

        let state = ControllerState {
            l_trigger: 0.0, r_trigger: 0.0,
            l_squeeze: 0.0, r_squeeze: 0.0,
            l_stick: xr::Vector2f { x: 0.0, y: 0.0 },
            r_stick: xr::Vector2f { x: 0.0, y: 0.0 },
            l_stick_click: false, r_stick_click: false,
            btn_a: false, btn_b: false,
            btn_x: false, btn_y: false, btn_menu: false,
            l_grip_pose: None, r_grip_pose: None,
            l_aim_pose:  None, r_aim_pose:  None,
        };

        Ok(Self {
            action_set,
            left_trigger, right_trigger,
            left_squeeze, right_squeeze,
            left_thumbstick, right_thumbstick,
            left_thumbstick_click, right_thumbstick_click,
            a_button, b_button, x_button, y_button, menu_button,
            left_grip_space, right_grip_space,
            left_aim_space, right_aim_space,
            state,
        })
    }

    pub fn sync(
        &mut self,
        session: &xr::Session<xr::Vulkan>,
        stage:   &xr::Space,
        time:    xr::Time,
    ) -> Result<(), Box<dyn std::error::Error>> {
        session.sync_actions(&[xr::ActiveActionSet::new(&self.action_set)])?;

        self.state.l_trigger     = self.left_trigger.state(session, xr::Path::NULL)?.current_state;
        self.state.r_trigger     = self.right_trigger.state(session, xr::Path::NULL)?.current_state;
        self.state.l_squeeze     = self.left_squeeze.state(session, xr::Path::NULL)?.current_state;
        self.state.r_squeeze     = self.right_squeeze.state(session, xr::Path::NULL)?.current_state;
        self.state.l_stick       = self.left_thumbstick.state(session, xr::Path::NULL)?.current_state;
        self.state.r_stick       = self.right_thumbstick.state(session, xr::Path::NULL)?.current_state;
        self.state.l_stick_click = self.left_thumbstick_click.state(session, xr::Path::NULL)?.current_state;
        self.state.r_stick_click = self.right_thumbstick_click.state(session, xr::Path::NULL)?.current_state;
        self.state.btn_a         = self.a_button.state(session, xr::Path::NULL)?.current_state;
        self.state.btn_b         = self.b_button.state(session, xr::Path::NULL)?.current_state;
        self.state.btn_x         = self.x_button.state(session, xr::Path::NULL)?.current_state;
        self.state.btn_y         = self.y_button.state(session, xr::Path::NULL)?.current_state;
        self.state.btn_menu      = self.menu_button.state(session, xr::Path::NULL)?.current_state;

        let l_grip = self.left_grip_space.locate(stage, time)?;
        let r_grip = self.right_grip_space.locate(stage, time)?;
        let l_aim  = self.left_aim_space.locate(stage, time)?;
        let r_aim  = self.right_aim_space.locate(stage, time)?;

        self.state.l_grip_pose = l_grip.location_flags
            .contains(xr::SpaceLocationFlags::POSITION_VALID).then_some(l_grip.pose);
        self.state.r_grip_pose = r_grip.location_flags
            .contains(xr::SpaceLocationFlags::POSITION_VALID).then_some(r_grip.pose);
        self.state.l_aim_pose  = l_aim.location_flags
            .contains(xr::SpaceLocationFlags::POSITION_VALID).then_some(l_aim.pose);
        self.state.r_aim_pose  = r_aim.location_flags
            .contains(xr::SpaceLocationFlags::POSITION_VALID).then_some(r_aim.pose);

        Ok(())
    }

    pub fn log(&self) {
        let s = &self.state;
        info!("── Controllers ──────────────────────────────────────");
        info!("  L trigger={:.2} squeeze={:.2} stick=({:.2},{:.2}) click={}",
            s.l_trigger, s.l_squeeze, s.l_stick.x, s.l_stick.y, s.l_stick_click);
        info!("  R trigger={:.2} squeeze={:.2} stick=({:.2},{:.2}) click={}",
            s.r_trigger, s.r_squeeze, s.r_stick.x, s.r_stick.y, s.r_stick_click);
        info!("  Buttons: A={} B={} X={} Y={} Menu={}",
            s.btn_a, s.btn_b, s.btn_x, s.btn_y, s.btn_menu);
        if let Some(p) = s.l_grip_pose {
            info!("  L grip pos=({:.3},{:.3},{:.3}) rot=({:.3},{:.3},{:.3},{:.3})",
                p.position.x, p.position.y, p.position.z,
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
        }
        if let Some(p) = s.r_grip_pose {
            info!("  R grip pos=({:.3},{:.3},{:.3}) rot=({:.3},{:.3},{:.3},{:.3})",
                p.position.x, p.position.y, p.position.z,
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w);
        }
        if let Some(p) = s.l_aim_pose {
            info!("  L aim  pos=({:.3},{:.3},{:.3})", p.position.x, p.position.y, p.position.z);
        }
        if let Some(p) = s.r_aim_pose {
            info!("  R aim  pos=({:.3},{:.3},{:.3})", p.position.x, p.position.y, p.position.z);
        }
    }
}
