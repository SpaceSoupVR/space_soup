pub mod renderer;

#[cfg(feature = "renderer")]
pub mod ui2d;

#[cfg(feature = "voxelize")]
pub mod voxelize;

pub mod perf_metrics_log;
pub mod perf_record;
pub mod thermal;

#[cfg(target_os = "android")]
pub mod xr;

pub use renderer::{Color3, Cuboid, CuboidShape, CuboidStyle, Renderer};

#[cfg(target_os = "android")]
pub use xr::{
    context::XrContext,
    controllers::{ControllerState, Controllers},
    hands::{HandJoint, HandTrackers},
    headset::Headset,
    vulkan::VkContext,
};

/// The game the tests that read shipped files open: `GAME_DIR` when set,
/// else artytest1, which ships test_room since the games were split one repo
/// each (`games/games.toml`). The workspace's `game/` is the older, unsplit
/// copy and no longer what the headset ships.
#[cfg(test)]
pub(crate) fn test_game_dir() -> std::path::PathBuf {
    std::env::var("GAME_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../games/artytest1"))
}
