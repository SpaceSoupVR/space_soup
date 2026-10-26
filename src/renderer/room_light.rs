use glam::Vec3;

use super::probe_stream::ProbeDesc;

pub type RoomLight = [[f32; 3]; 9];

pub const ROOM_EDGE: f32 = 0.5;

const NEAR: f32 = 0.25;

pub fn room_light_at(descs: &[ProbeDesc], p: Vec3) -> RoomLight {

    let mut rooms: Vec<(u32, Vec3, Vec3)> = Vec::new();
    for d in descs.iter().filter(|d| d.room_light.is_some()) {
        if !rooms.iter().any(|r| r.0 == d.volume) {
            rooms.push((d.volume, d.min, d.max));
        }
    }
    let size = |lo: Vec3, hi: Vec3| (hi - lo).max(Vec3::ZERO).element_product();
    rooms.sort_by(|a, b| size(a.1, a.2).total_cmp(&size(b.1, b.2)));
    let mut out = [[0.0f32; 3]; 9];
    let mut remaining = 1.0f32;
    for &(volume, lo, hi) in &rooms {

        let inside = (p - lo).min(hi - p).min_element();
        let w = smoothstep(-ROOM_EDGE, 0.0, inside);
        if w <= 0.0 {
            continue;
        }
        let take = w * remaining;
        add(&mut out, &photographs_at(descs, volume, p), take);
        remaining *= 1.0 - w;
        if remaining <= 1e-4 {
            break;
        }
    }

    if remaining > 0.5 {
        if let Some(d) = descs
            .iter()
            .filter(|d| d.room_light.is_some())
            .min_by(|a, b| {
                (a.centre - p)
                    .length_squared()
                    .total_cmp(&(b.centre - p).length_squared())
            })
        {
            add(&mut out, &d.room_light.unwrap_or_default(), remaining);
        }
    }
    out
}

fn photographs_at(descs: &[ProbeDesc], volume: u32, p: Vec3) -> RoomLight {
    let mut out = [[0.0f32; 3]; 9];
    let mut total = 0.0f32;
    for d in descs.iter().filter(|d| d.volume == volume) {
        let Some(light) = &d.room_light else { continue };
        let w = 1.0 / ((d.centre - p).length_squared() + NEAR);
        add(&mut out, light, w);
        total += w;
    }
    if total > 0.0 {
        for c in out.iter_mut().flatten() {
            *c /= total;
        }
    }
    out
}

fn add(out: &mut RoomLight, light: &RoomLight, w: f32) {
    for (o, l) in out.iter_mut().zip(light) {
        for c in 0..3 {
            o[c] += l[c] * w;
        }
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn turned_to_player(light: &RoomLight, yaw: f32) -> RoomLight {
    let (s, c) = yaw.sin_cos();
    let (r3, sc, ss, cc) = (3.0f32.sqrt(), s * c, s * s, c * c);
    let mut out = [[0.0f32; 3]; 9];
    for k in 0..3 {
        let l = |i: usize| light[i][k];
        out[0][k] = l(0);
        out[1][k] = l(1);
        out[2][k] = c * l(2) + s * l(3);
        out[3][k] = -s * l(2) + c * l(3);
        out[4][k] = c * l(4) - s * l(5);
        out[5][k] = s * l(4) + c * l(5);
        out[6][k] = (1.5 * cc - 0.5) * l(6) + r3 * sc * l(7) + 0.5 * r3 * ss * l(8);
        out[7][k] = -r3 * sc * l(6) + (cc - ss) * l(7) + sc * l(8);
        out[8][k] = 0.5 * r3 * ss * l(6) - sc * l(7) + f32::midpoint(1.0, cc) * l(8);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn basis(d: Vec3) -> [f32; 9] {
        let (x, y, z) = (d.x, d.y, d.z);
        [
            0.282_095,
            0.488_603 * y,
            0.488_603 * z,
            0.488_603 * x,
            1.092_548 * x * y,
            1.092_548 * y * z,
            0.315_392 * (3.0 * z * z - 1.0),
            1.092_548 * x * z,
            0.546_274 * (x * x - y * y),
        ]
    }

    fn along(light: &RoomLight, d: Vec3) -> f32 {
        basis(d.normalize())
            .iter()
            .zip(light)
            .map(|(b, l)| b * l[0])
            .sum()
    }

    fn to_world(d: Vec3, yaw: f32) -> Vec3 {
        let (s, c) = yaw.sin_cos();
        Vec3::new(c * d.x + s * d.z, d.y, -s * d.x + c * d.z)
    }

    fn sample_light(seed: u32) -> RoomLight {
        let mut x = seed.wrapping_mul(2_654_435_761);
        std::array::from_fn(|_| {
            std::array::from_fn(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x % 1000) as f32 / 500.0 - 1.0
            })
        })
    }

    #[test]
    fn turning_the_light_is_reading_it_along_the_turned_direction() {
        for seed in 1..6u32 {
            let light = sample_light(seed);
            for yaw in [
                0.0f32,
                0.3,
                1.0,
                std::f32::consts::FRAC_PI_2,
                2.5,
                -1.2,
                std::f32::consts::PI,
            ] {
                let turned = turned_to_player(&light, yaw);
                for d in [
                    Vec3::X,
                    Vec3::Y,
                    Vec3::Z,
                    Vec3::new(0.3, -0.5, 0.8),
                    Vec3::new(-0.7, 0.2, -0.4),
                    Vec3::new(0.1, 0.9, -0.3),
                ] {
                    let (got, want) = (
                        along(&turned, d),
                        along(&light, to_world(d.normalize(), yaw)),
                    );
                    assert!(
                        (got - want).abs() < 1e-4,
                        "seed {seed} yaw {yaw} along {d}: {got} vs {want}"
                    );
                }
            }
        }
    }

    fn room(volume: u32, lo: Vec3, hi: Vec3, centre: Vec3, level: f32) -> ProbeDesc {
        let mut light = [[0.0f32; 3]; 9];
        light[0] = [level / 0.282_095; 3];
        ProbeDesc {
            centre,
            min: lo,
            max: hi,
            volume,
            has_depth: true,
            room_light: Some(light),
        }
    }

    fn even(light: &RoomLight) -> f32 {
        light[0][0] * 0.282_095
    }

    #[test]
    fn a_model_takes_the_light_of_the_room_it_stands_in() {
        let outdoors = room(
            0,
            Vec3::splat(-50.0),
            Vec3::splat(50.0),
            Vec3::new(0.0, 2.0, 20.0),
            1.0,
        );
        let hall = room(
            1,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 4.0, 10.0),
            Vec3::new(5.0, 1.6, 5.0),
            0.2,
        );
        let hallway = room(
            2,
            Vec3::new(10.0, 0.0, 4.0),
            Vec3::new(20.0, 3.0, 6.0),
            Vec3::new(15.0, 1.6, 5.0),
            0.05,
        );
        let descs = [outdoors, hall, hallway];
        assert!(
            (even(&room_light_at(&descs, Vec3::new(5.0, 1.0, 5.0))) - 0.2).abs() < 1e-4,
            "in the hall"
        );
        assert!(
            (even(&room_light_at(&descs, Vec3::new(15.0, 1.0, 5.0))) - 0.05).abs() < 1e-4,
            "in the hallway"
        );
        assert!(
            (even(&room_light_at(&descs, Vec3::new(30.0, 1.0, 30.0))) - 1.0).abs() < 1e-4,
            "outdoors"
        );

        assert!(
            (even(&room_light_at(&descs, Vec3::new(0.05, 1.0, 9.95))) - 0.2).abs() < 1e-4,
            "by the hall's corner"
        );

        let mut last = f32::MAX;
        for i in 0..=20 {
            let x = 9.4 + i as f32 * 0.04;
            let e = even(&room_light_at(&descs, Vec3::new(x, 1.0, 5.0)));
            assert!(e <= last + 1e-5 && e > 0.0499 && e < 0.2001, "x {x}: {e}");
            last = e;
        }
        let mid = even(&room_light_at(&descs, Vec3::new(9.75, 1.0, 5.0)));
        assert!(
            mid > 0.08 && mid < 0.17,
            "a quarter metre short of the doorway, some of each: {mid}"
        );
    }

    #[test]
    fn a_rooms_photographs_blend_by_distance() {
        let a = room(
            1,
            Vec3::ZERO,
            Vec3::new(10.0, 4.0, 4.0),
            Vec3::new(1.0, 2.0, 2.0),
            0.4,
        );
        let b = room(
            1,
            Vec3::ZERO,
            Vec3::new(10.0, 4.0, 4.0),
            Vec3::new(9.0, 2.0, 2.0),
            0.0,
        );
        let near_a = even(&room_light_at(&[a, b], Vec3::new(1.2, 2.0, 2.0)));
        assert!(near_a > 0.35, "{near_a}");
        let half = even(&room_light_at(&[a, b], Vec3::new(5.0, 2.0, 2.0)));
        assert!((half - 0.2).abs() < 1e-4, "{half}");

        let old = ProbeDesc {
            room_light: None,
            ..a
        };
        assert_eq!(
            room_light_at(&[old], Vec3::new(1.0, 2.0, 2.0)),
            [[0.0; 3]; 9]
        );
    }
}
