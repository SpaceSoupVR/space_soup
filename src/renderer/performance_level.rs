use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceLevel {

    PowerSavings,

    SustainedLow,

    SustainedHigh,

    Boost,
}

impl PerformanceLevel {
    pub fn label(self) -> &'static str {
        match self {
            PerformanceLevel::PowerSavings => "power_savings",
            PerformanceLevel::SustainedLow => "sustained_low",
            PerformanceLevel::SustainedHigh => "sustained_high",
            PerformanceLevel::Boost => "boost",
        }
    }

    pub fn raw(self) -> i32 {
        match self {
            PerformanceLevel::PowerSavings => 0,
            PerformanceLevel::SustainedLow => 25,
            PerformanceLevel::SustainedHigh => 50,
            PerformanceLevel::Boost => 75,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Domain {
    Cpu,
    Gpu,
}

impl Domain {
    pub const ALL: [Domain; 2] = [Domain::Cpu, Domain::Gpu];

    pub fn label(self) -> &'static str {
        match self {
            Domain::Cpu => "cpu",
            Domain::Gpu => "gpu",
        }
    }

    pub fn raw(self) -> i32 {
        match self {
            Domain::Cpu => 1,
            Domain::Gpu => 2,
        }
    }

    pub fn startup_level(self) -> PerformanceLevel {
        match self {
            Domain::Cpu => PerformanceLevel::SustainedLow,
            Domain::Gpu => PerformanceLevel::SustainedHigh,
        }
    }
}

pub fn requests(
    was: [Option<PerformanceLevel>; 2],
    now: [Option<PerformanceLevel>; 2],
) -> Vec<(Domain, PerformanceLevel)> {
    Domain::ALL
        .iter()
        .zip(was.iter().zip(now.iter()))
        .filter_map(|(&domain, (&was, &now))| match (was, now) {
            (a, b) if a == b => None,
            (_, Some(level)) => Some((domain, level)),
            (Some(_), None) => Some((domain, domain.startup_level())),
            (None, None) => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use PerformanceLevel::*;

    #[test]
    fn nothing_asked_calls_nothing() {
        assert!(requests([None, None], [None, None]).is_empty());
    }

    #[test]
    fn a_level_is_asked_when_it_changes_and_only_then() {
        assert_eq!(requests([None, None], [None, Some(Boost)]), vec![(Domain::Gpu, Boost)]);
        assert!(requests([None, Some(Boost)], [None, Some(Boost)]).is_empty());
        assert_eq!(
            requests([Some(SustainedLow), Some(Boost)], [Some(SustainedHigh), Some(Boost)]),
            vec![(Domain::Cpu, SustainedHigh)]
        );
    }

    #[test]
    fn a_level_no_longer_asked_goes_back_to_where_the_app_started() {
        assert_eq!(
            requests([Some(Boost), Some(PowerSavings)], [None, None]),
            vec![(Domain::Cpu, SustainedLow), (Domain::Gpu, SustainedHigh)]
        );
    }

    #[test]
    fn the_levels_and_domains_are_the_extensions_values() {
        let levels: Vec<i32> = [PowerSavings, SustainedLow, SustainedHigh, Boost].iter().map(|l| l.raw()).collect();
        assert_eq!(levels, vec![0, 25, 50, 75]);
        assert_eq!((Domain::Cpu.raw(), Domain::Gpu.raw()), (1, 2));
    }

    #[test]
    fn a_lever_names_a_level_as_its_label_does() {
        for level in [PowerSavings, SustainedLow, SustainedHigh, Boost] {
            let parsed: PerformanceLevel = serde_json::from_str(&format!("\"{}\"", level.label())).unwrap();
            assert_eq!(parsed, level);
        }
    }
}
