//! Native macOS window chrome for wgpu windows.
//!
//! Two pieces, both optional and composable:
//!
//! * [`configure_macos_window`] — turns a plain titled window into Apple-style
//!   chrome: the content view extends to the window's top edge, the titlebar
//!   is transparent, and an empty unified `NSToolbar` gives the traffic
//!   lights a proper native strip (drag, double-click zoom, right-click menu
//!   all behave like any Mac app). Reserve [`MACOS_TITLEBAR_PT`] logical
//!   points of UI at the top for that strip.
//! * [`install_window_vibrancy`] — a frosted `NSVisualEffectView` installed
//!   *behind* the wgpu `CAMetalLayer`, so a transparent render (non-opaque
//!   surface alpha mode, clear color alpha below 1.0) composites over a blur
//!   of whatever is behind the window.

use objc2_app_kit::{
    NSAppearance, NSAppearanceCustomization, NSAppearanceNameAqua, NSAppearanceNameDarkAqua,
    NSAutoresizingMaskOptions, NSToolbar, NSView, NSVisualEffectBlendingMode,
    NSVisualEffectMaterial, NSVisualEffectState, NSVisualEffectView,
    NSWindowOrderingMode, NSWindowStyleMask, NSWindowTitleVisibility, NSWindowToolbarStyle,
};
use objc2_foundation::MainThreadMarker;
use wgpu::rwh::{HasWindowHandle, RawWindowHandle};

/// Logical height of the native titlebar strip that
/// [`configure_macos_window`] creates with `unified_titlebar`. Layouts should
/// keep their own chrome below this line.
pub const MACOS_TITLEBAR_PT: f32 = 52.0;

/// Backdrop material for [`install_window_vibrancy`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VibrancyMaterial {
    /// Dark, high-contrast HUD glass.
    HudWindow,
    /// The material of a standard window background.
    UnderWindowBackground,
    /// The lighter sidebar material.
    Sidebar,
}

impl VibrancyMaterial {
    fn to_ns(self) -> NSVisualEffectMaterial {
        match self {
            Self::HudWindow => NSVisualEffectMaterial::HUDWindow,
            Self::UnderWindowBackground => NSVisualEffectMaterial::UnderWindowBackground,
            Self::Sidebar => NSVisualEffectMaterial::Sidebar,
        }
    }
}

/// Forced window appearance, independent of the system setting. Vibrancy
/// materials, titlebar text and control tints all follow it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WindowAppearance {
    /// Follow the system appearance.
    #[default]
    System,
    Dark,
    Light,
}

/// How [`configure_macos_window`] should dress the window.
#[derive(Clone, Copy, Debug)]
pub struct MacosWindowChrome {
    /// Pin the window (and its vibrancy material) to an appearance.
    pub appearance: WindowAppearance,
    /// Extend the content view under a transparent titlebar and add an empty
    /// unified `NSToolbar`, so the traffic lights sit in a native
    /// [`MACOS_TITLEBAR_PT`]-tall strip with native drag/zoom/right-click.
    pub unified_titlebar: bool,
    /// Hide the window's title text in that strip.
    pub hide_title: bool,
    /// Frosted backdrop behind the render (see [`install_window_vibrancy`]).
    pub vibrancy: Option<VibrancyMaterial>,
}

impl Default for MacosWindowChrome {
    fn default() -> Self {
        Self {
            appearance: WindowAppearance::System,
            unified_titlebar: true,
            hide_title: false,
            vibrancy: Some(VibrancyMaterial::HudWindow),
        }
    }
}

fn appkit_view(window: &impl HasWindowHandle) -> Option<&NSView> {
    let handle = window.window_handle().ok()?;
    let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
        return None;
    };
    MainThreadMarker::new()?;
    // Safety: an AppKit handle's ns_view is an NSView, and we are on the main
    // thread; the caller keeps the window (and so the view) alive.
    Some(unsafe { appkit.ns_view.cast().as_ref() })
}

/// Applies [`MacosWindowChrome`] to a window created by any windowing crate
/// that exposes a raw AppKit handle (winit, tao, ...).
///
/// Call once after the window exists, on the main thread. For vibrancy to
/// show, create the window transparent and configure the wgpu surface with a
/// non-opaque alpha mode. Returns `false` when the handle is not an AppKit
/// window or this is not the main thread.
pub fn configure_macos_window(
    window: &impl HasWindowHandle,
    chrome: &MacosWindowChrome,
) -> bool {
    let Some(content) = appkit_view(window) else {
        return false;
    };
    let Some(ns_window) = content.window() else {
        return false;
    };
    // appkit_view established we're on the main thread.
    let mtm = MainThreadMarker::new().unwrap();

    let named_appearance = match chrome.appearance {
        WindowAppearance::System => None,
        WindowAppearance::Dark => Some(unsafe { NSAppearanceNameDarkAqua }),
        WindowAppearance::Light => Some(unsafe { NSAppearanceNameAqua }),
    };
    if let Some(name) = named_appearance {
        let appearance = NSAppearance::appearanceNamed(name);
        unsafe { ns_window.setAppearance(appearance.as_deref()) };
    }

    if chrome.unified_titlebar {
        ns_window.setTitlebarAppearsTransparent(true);
        ns_window.setStyleMask(ns_window.styleMask() | NSWindowStyleMask::FullSizeContentView);
        unsafe {
            let toolbar = NSToolbar::init(mtm.alloc());
            #[allow(deprecated)]
            toolbar.setShowsBaselineSeparator(false);
            ns_window.setToolbar(Some(&toolbar));
            ns_window.setToolbarStyle(NSWindowToolbarStyle::Unified);
        }
    }
    ns_window.setTitleVisibility(if chrome.hide_title {
        NSWindowTitleVisibility::NSWindowTitleHidden
    } else {
        NSWindowTitleVisibility::NSWindowTitleVisible
    });

    if let Some(material) = chrome.vibrancy {
        install_window_vibrancy(window, material);
    }
    true
}

/// Puts a full-window `NSVisualEffectView` behind the window's content view.
///
/// The effect view is added to the content view's *superview*, ordered below
/// the content view. Ordering matters: added as a subview of the content view
/// itself (what the `window-vibrancy` crate does) it sits in front of the
/// content view's `CAMetalLayer` and the blur covers the entire render.
///
/// Call once after the window exists, on the main thread, with the window
/// created transparent. Returns `false` when the handle is not an AppKit
/// window or this is not the main thread.
pub fn install_window_vibrancy(
    window: &impl HasWindowHandle,
    material: VibrancyMaterial,
) -> bool {
    let Some(content) = appkit_view(window) else {
        return false;
    };
    let mtm = MainThreadMarker::new().unwrap();

    unsafe {
        let Some(superview) = content.superview() else {
            return false;
        };

        let blur = NSVisualEffectView::initWithFrame(mtm.alloc(), content.frame());
        blur.setMaterial(material.to_ns());
        blur.setBlendingMode(NSVisualEffectBlendingMode::BehindWindow);
        blur.setState(NSVisualEffectState::FollowsWindowActiveState);
        blur.setAutoresizingMask(
            NSAutoresizingMaskOptions::NSViewWidthSizable
                | NSAutoresizingMaskOptions::NSViewHeightSizable,
        );
        superview.addSubview_positioned_relativeTo(
            &blur,
            NSWindowOrderingMode::NSWindowBelow,
            Some(content),
        );
    }
    true
}
