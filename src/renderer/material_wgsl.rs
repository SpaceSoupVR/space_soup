//! Projection shared by every shader that textures geometry with no UVs.
//!
//! Terrain and caves both have this problem and it has the same answer, so the
//! answer lives in one place. A heightfield has no natural UVs; a surface-nets
//! cave has none either, and never will -- the mesh is regenerated from a voxel
//! field every time it is edited, so there is nothing an unwrapper could keep
//! stable. Both take their texture coordinates from WORLD POSITION instead, and
//! choose the plane to project onto from the surface normal.
//!
//! BIPLANAR RATHER THAN TRIPLANAR
//!
//! Triplanar takes three samples per material and blends them by the normal's
//! three components. The smallest of those three always contributes least by
//! construction, so biplanar drops it and takes two. On a Quest that is a third
//! off the most expensive part of both shaders, and nobody has ever been able
//! to see it.
//!
//! WHY THIS IS GEOMETRY ONLY
//!
//! It returns axes and uvs and samples nothing. The two callers need different
//! things from the result -- terrain samples one array, a cave samples the same
//! array but also has to swizzle a tangent-space normal into world space per
//! axis -- and a shared function that also sampled would have to be handed its
//! texture by name, which is how a "shared" helper ends up with a parameter per
//! caller.

/// WGSL declaring `Biplanar`, `biplanar_uv` and `biplanar_axes`.
///
/// Axis ids are the axis PROJECTED ALONG, not the uv plane: 0 projects along x
/// (uv = world.zy), 1 along y (uv = world.xz), 2 along z (uv = world.xy). The
/// id is returned rather than only the uv because a normal map cannot be
/// unpacked without knowing which plane it came from.
pub fn wgsl_biplanar_block() -> &'static str {
    r#"
struct Biplanar {
    uv_major: vec2<f32>,
    uv_minor: vec2<f32>,
    axis_major: u32,
    axis_minor: u32,
    // The major axis's share of the blend. At worst 0.5, when two axes tie.
    w: f32,
}

fn biplanar_uv(axis: u32, world: vec3<f32>, r: f32) -> vec2<f32> {
    if (axis == 0u) { return world.zy / r; }
    if (axis == 1u) { return world.xz / r; }
    return world.xy / r;
}

// The two strongest axes of `n`, ranked without a branch per pixel.
fn biplanar_axes(world: vec3<f32>, n: vec3<f32>, repeat: f32) -> Biplanar {
    let r = max(repeat, 0.001);
    let a = abs(n);

    let ma = max(a.x, max(a.y, a.z));
    let mi = min(a.x, min(a.y, a.z));
    let me = a.x + a.y + a.z - ma - mi;

    var major: u32;
    var minor: u32;
    if (a.y == ma) {
        major = 1u;
        minor = select(2u, 0u, a.x >= a.z);
    } else if (a.x == ma) {
        major = 0u;
        minor = select(2u, 1u, a.z < a.y);
    } else {
        major = 2u;
        minor = select(0u, 1u, a.x < a.y);
    }

    var out: Biplanar;
    out.axis_major = major;
    out.axis_minor = minor;
    out.uv_major = biplanar_uv(major, world, r);
    out.uv_minor = biplanar_uv(minor, world, r);
    out.w = ma / max(ma + me, 0.001);
    return out;
}
"#
}

/// WGSL declaring `whiteout`, which folds a tangent-space normal sampled on one
/// projection plane into the geometric normal.
///
/// The "whiteout" blend perturbs the geometric normal rather than replacing it,
/// which is the only correct thing to do with untangented geometry: the map
/// describes bumps RELATIVE to a surface, and replacing would light every slope
/// as though it were facing the projection plane. A flat texel -- the (128,
/// 128, 255) fallback, unpacking to (0, 0, 1) -- returns the geometric normal
/// exactly, so an unauthored normal map costs its samples and changes nothing.
///
/// Per axis, because the tangent frame differs: on the y projection the map's u
/// runs along world x and its v along world z, on the x projection u runs along
/// z and v along y, and on the z projection u runs along x and v along y. Using
/// one swizzle for all three is the classic triplanar normal bug and it shows
/// up as lighting that reverses across the axis boundary.
pub fn wgsl_whiteout_block() -> &'static str {
    r#"
fn whiteout(axis: u32, tn: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    if (axis == 0u) { return vec3<f32>(abs(tn.z) * n.x, tn.y + n.y, tn.x + n.z); }
    if (axis == 1u) { return vec3<f32>(tn.x + n.x, abs(tn.z) * n.y, tn.y + n.z); }
    return vec3<f32>(tn.x + n.x, tn.y + n.y, abs(tn.z) * n.z);
}
"#
}
