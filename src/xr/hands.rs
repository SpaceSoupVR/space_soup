//! Hand joint tracking via EXT_hand_tracking.

use log::info;
use openxr as xr;

use crate::xr::XrContext;

const JOINT_NAMES: [&str; 26] = [
    "Palm",        "Wrist",
    "Thumb_Meta",  "Thumb_Prox",  "Thumb_Dist",  "Thumb_Tip",
    "Index_Meta",  "Index_Prox",  "Index_Inter",  "Index_Dist",  "Index_Tip",
    "Middle_Meta", "Middle_Prox", "Middle_Inter", "Middle_Dist", "Middle_Tip",
    "Ring_Meta",   "Ring_Prox",   "Ring_Inter",   "Ring_Dist",   "Ring_Tip",
    "Little_Meta", "Little_Prox", "Little_Inter", "Little_Dist", "Little_Tip",
];

pub struct HandJoint {
    pub pose:   xr::Posef,
    pub radius: f32,
    pub valid:  bool,
}

pub struct HandTrackers {
    left:             Option<xr::HandTracker>,
    right:            Option<xr::HandTracker>,
    pub left_joints:  Vec<HandJoint>,
    pub right_joints: Vec<HandJoint>,
}

impl HandTrackers {
    pub fn new(
        xr:      &XrContext,
        session: &xr::Session<xr::Vulkan>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (left, right) = if xr.has_hand_tracking {
            (
                Some(session.create_hand_tracker(xr::Hand::LEFT)?),
                Some(session.create_hand_tracker(xr::Hand::RIGHT)?),
            )
        } else {
            info!("Hand tracking not available");
            (None, None)
        };
        Ok(Self { left, right, left_joints: Vec::new(), right_joints: Vec::new() })
    }

    pub fn sync(
        &mut self,
        stage: &xr::Space,
        time:  xr::Time,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.left_joints.clear();
        self.right_joints.clear();

        if let Some(ref tracker) = self.left {
            if let Ok(Some(joints)) = stage.locate_hand_joints(tracker, time) {
                for j in joints.iter() {
                    self.left_joints.push(HandJoint {
                        pose:   j.pose,
                        radius: j.radius,
                        valid:  j.location_flags.contains(xr::SpaceLocationFlags::POSITION_VALID),
                    });
                }
            }
        }
        if let Some(ref tracker) = self.right {
            if let Ok(Some(joints)) = stage.locate_hand_joints(tracker, time) {
                for j in joints.iter() {
                    self.right_joints.push(HandJoint {
                        pose:   j.pose,
                        radius: j.radius,
                        valid:  j.location_flags.contains(xr::SpaceLocationFlags::POSITION_VALID),
                    });
                }
            }
        }
        Ok(())
    }

    pub fn log(&self) {
        for (joints, side) in [(&self.left_joints, "L"), (&self.right_joints, "R")] {
            if joints.is_empty() { continue; }
            info!("  {side} hand joints:");
            for (i, j) in joints.iter().enumerate() {
                if j.valid {
                    let p    = j.pose.position;
                    let name = JOINT_NAMES.get(i).unwrap_or(&"Unknown");
                    info!("    [{name:14}] pos=({:.3},{:.3},{:.3}) r={:.4}",
                        p.x, p.y, p.z, j.radius);
                }
            }
        }
    }
}
