use glam::Vec3;
use std::ops::Range;

pub const RUN_TRIANGLES: u32 = 96;

pub const JOIN_GAP_TRIANGLES: u32 = 768;

pub const LIFT_FROM: f32 = 3500.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    pub first: u32,
    pub count: u32,
    pub bounds: (Vec3, Vec3),
}

pub fn runs(position: impl Fn(u32) -> Vec3, indices: &[u32], reach: f32) -> Vec<Run> {
    let step = (RUN_TRIANGLES * 3) as usize;
    indices
        .chunks(step)
        .enumerate()
        .map(|(k, run)| {
            let (lo, hi) = run.iter().fold(
                (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
                |(lo, hi), &i| {
                    let p = position(i);
                    (lo.min(p), hi.max(p))
                },
            );
            Run { first: (k * step) as u32, count: run.len() as u32, bounds: (lo - Vec3::splat(reach), hi + Vec3::splat(reach)) }
        })
        .collect()
}

pub fn bounds_for(run: &Run, eyes: &[Vec3]) -> (Vec3, Vec3) {
    lifted(run.bounds, eyes)
}

pub fn lifted(bounds: (Vec3, Vec3), eyes: &[Vec3]) -> (Vec3, Vec3) {
    let (lo, mut hi) = bounds;
    for eye in eyes {
        let far = (lo.x - eye.x).abs().max((hi.x - eye.x).abs()).hypot((lo.z - eye.z).abs().max((hi.z - eye.z).abs()));
        if far >= LIFT_FROM {
            hi.y = hi.y.max(eye.y + (hi.y - lo.y));
        }
    }
    (lo, hi)
}

pub fn seen_ranges(runs: &[Run], mut seen: impl FnMut(usize, &Run) -> bool) -> Vec<Range<u32>> {
    let mut out: Vec<Range<u32>> = Vec::new();
    for (k, run) in runs.iter().enumerate() {
        if !seen(k, run) {
            continue;
        }
        match out.last_mut() {
            Some(last) if run.first <= last.end + JOIN_GAP_TRIANGLES * 3 => last.end = run.first + run.count,
            _ => out.push(run.first..run.first + run.count),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nx: u32, nz: u32) -> (Vec<Vec3>, Vec<u32>) {
        let mut verts = Vec::new();
        for iz in 0..=nz {
            for ix in 0..=nx {
                verts.push(Vec3::new(ix as f32, 0.0, iz as f32));
            }
        }
        let mut indices = Vec::new();
        for iz in 0..nz {
            for ix in 0..nx {
                let a = iz * (nx + 1) + ix;
                let (b, c) = (a + 1, a + nx + 1);
                indices.extend_from_slice(&[a, c, b, b, c, c + 1]);
            }
        }
        (verts, indices)
    }

    #[test]
    fn the_runs_are_the_buffer_in_its_own_order_and_hold_their_triangles() {
        let (verts, indices) = grid(100, 7);
        let runs = runs(|i| verts[i as usize], &indices, 0.5);
        let mut next = 0;
        for run in &runs {
            assert_eq!(run.first, next, "a gap or an overlap");
            assert_eq!(run.count % 3, 0);
            assert!(run.count <= RUN_TRIANGLES * 3);
            for &i in &indices[run.first as usize..(run.first + run.count) as usize] {
                let p = verts[i as usize];
                assert!(p.cmpge(run.bounds.0 + 0.49).all() && p.cmple(run.bounds.1 - 0.49).all(), "a vertex outside its run's box less the reach");
            }
            next += run.count;
        }
        assert_eq!(next as usize, indices.len());

        assert!(runs.iter().all(|r| r.bounds.1.z - r.bounds.0.z <= 3.0 && r.bounds.1.x - r.bounds.0.x <= 101.0));
        assert!(runs.len() >= 14);
    }

    #[test]
    fn what_is_drawn_is_every_seen_run_in_order_joined_over_short_gaps() {
        let (verts, indices) = grid(100, 40);
        let runs = runs(|i| verts[i as usize], &indices, 0.0);

        assert_eq!(seen_ranges(&runs, |_, _| true), vec![0..indices.len() as u32]);

        assert!(seen_ranges(&runs, |_, _| false).is_empty());

        let wedge = |_: usize, r: &Run| (r.bounds.1.x >= 90.0 && r.bounds.1.z <= 10.0) || r.bounds.0.z >= 30.0;
        let ranges = seen_ranges(&runs, wedge);
        assert_eq!(ranges.len(), 2, "{ranges:?}");
        assert!(ranges[0].end < ranges[1].start && ranges[1].end == indices.len() as u32);
        for (k, run) in runs.iter().enumerate() {
            let drawn = ranges.iter().any(|r| r.start <= run.first && run.first + run.count <= r.end);
            assert!(drawn || !wedge(k, run), "a seen run is not drawn");
        }

        assert!(ranges.windows(2).all(|w| w[0].end <= w[1].start));

        let far_apart = seen_ranges(&runs, |_, r| r.bounds.1.z <= 2.0 || r.bounds.0.z >= 38.0);
        assert_eq!(far_apart.len(), 2);
        assert!(far_apart[1].start - far_apart[0].end > JOIN_GAP_TRIANGLES * 3);
    }

    #[test]
    fn a_run_far_enough_out_to_be_lifted_stands_as_high_as_the_eye() {
        let near = Run { first: 0, count: 3, bounds: (Vec3::new(-10.0, -1.0, -10.0), Vec3::new(10.0, 1.0, 10.0)) };
        let far = Run { first: 3, count: 3, bounds: (Vec3::new(3990.0, -1.0, -10.0), Vec3::new(4100.0, 1.0, 10.0)) };
        let eyes = [Vec3::new(0.0, 30.0, 0.0), Vec3::new(0.1, 31.0, 0.0)];
        assert_eq!(bounds_for(&near, &eyes), near.bounds);
        assert_eq!(lifted(far.bounds, &eyes), bounds_for(&far, &eyes));
        let (lo, hi) = bounds_for(&far, &eyes);
        assert_eq!(lo, far.bounds.0);
        assert!(hi.y >= 31.0 && hi.x == 4100.0);
    }
}
