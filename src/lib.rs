pub mod renderer;

#[cfg(feature = "renderer")]
pub mod ui2d;

#[cfg(feature = "voxelize")]
pub mod voxelize;

#[cfg(not(target_os = "android"))]
pub mod canvas;

#[cfg(target_os = "android")]
pub mod xr;

pub use renderer::{Color3, Cuboid, CuboidStyle, Renderer};

#[cfg(not(target_os = "android"))]
pub use canvas::Canvas;

#[cfg(target_os = "android")]
pub use xr::{
    context::XrContext,
    controllers::{ControllerState, Controllers},
    hands::{HandJoint, HandTrackers},
    headset::Headset,
    vulkan::VkContext,
};
