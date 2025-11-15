use csv::ReaderBuilder;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::time::Instant;

use chrono::{Datelike, TimeZone, Timelike, Utc, Weekday};
use chrono_tz::America::New_York;

use rand::thread_rng;
use rand_distr::{Distribution, Normal};

// --------------------------- Data types ---------------------------

#[derive(Debug, Deserialize)]
struct Bar {
    timestamp: u64, // Unix ms
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DayKey {
    year: i32,
    month: u32,
    day: u32,
}

// --------------------------- IO ---------------------------

fn load_bars(path: &str) -> csv::Result<Vec<Bar>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let mut bars = Vec::new();
    for rec in rdr.deserialize::<Bar>() {
        bars.push(rec?);
    }
    Ok(bars)
}

// --------------------------- Grouping helpers ---------------------------

// Keep ONLY Mon–Fri, 09:30–16:00 America/New_York (regular trading hours)
fn group_by_day_rth_weekdays(bars: &[Bar]) -> BTreeMap<DayKey, Vec<Bar>> {
    let rth_start = (9, 30, 0);
    let rth_end = (16, 0, 0);
    let mut map: BTreeMap<DayKey, Vec<Bar>> = BTreeMap::new();

    for b in bars {
        let dt_utc = match Utc.timestamp_millis_opt(b.timestamp as i64).single() {
            Some(t) => t,
            None => continue,
        };
        let dt_ny = dt_utc.with_timezone(&New_York);

        // Skip weekends
        let wd = dt_ny.weekday();
        if wd == Weekday::Sat || wd == Weekday::Sun {
            continue;
        }

        // Keep only RTH minutes
        let t = dt_ny.time();
        let in_rth =
            (t.hour(), t.minute(), t.second()) >= rth_start
                && (t.hour(), t.minute(), t.second()) < rth_end;
        if !in_rth {
            continue;
        }

        let key = DayKey {
            year: dt_ny.year(),
            month: dt_ny.month(),
            day: dt_ny.day(),
        };
        map.entry(key).or_default().push(Bar {
            timestamp: b.timestamp,
            open: b.open,
            high: b.high,
            low: b.low,
            close: b.close,
        });
    }

    for (_k, v) in map.iter_mut() {
        v.sort_by_key(|b| b.timestamp);
    }
    map
}

// --------------------------- Small utilities ---------------------------

fn moving_avg(x: &[f64], k: usize) -> Vec<f64> {
    if k <= 1 || x.is_empty() {
        return x.to_vec();
    }
    let mut out = vec![0.0; x.len()];
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += x[i];
        if i >= k {
            sum -= x[i - k];
        }
        out[i] = sum / ((i + 1).min(k) as f64);
    }
    out
}

// Estimate daily sigma from minute log-returns; clamp to sane daily range
fn estimate_sigma_from_day(day_bars: &[Bar]) -> f64 {
    if day_bars.len() < 2 {
        return 0.01;
    }
    let mut rets = Vec::with_capacity(day_bars.len() - 1);
    for w in day_bars.windows(2) {
        let p0 = w[0].close;
        let p1 = w[1].close;
        if p0 > 0.0 && p1.is_finite() && p0.is_finite() {
            let r = (p1 / p0).ln();
            if r.is_finite() {
                rets.push(r);
            }
        }
    }
    if rets.len() < 2 {
        return 0.01;
    }
    let mean = rets.iter().sum::<f64>() / (rets.len() as f64);
    let var = rets
        .iter()
        .map(|r| (r - mean).powi(2))
        .sum::<f64>()
        / ((rets.len() - 1) as f64);
    let std_step = var.max(0.0).sqrt();
    let n_steps = day_bars.len().max(1) as f64;
    let sigma = std_step * n_steps.sqrt();
    sigma.clamp(0.005, 0.05) // ~0.5%..5% daily
}

// Compute **day-specific drift μ** from open→close log return, clamp to ±5%/day.
fn estimate_mu_from_day(day_bars: &[Bar]) -> f64 {
    if day_bars.is_empty() {
        return 0.0;
    }
    let s0 = day_bars.first().unwrap().close;
    let s1 = day_bars.last().unwrap().close;
    if s0 <= 0.0 || !s0.is_finite() || !s1.is_finite() {
        return 0.0;
    }
    let mu = (s1 / s0).ln(); // log-return from first RTH close to last RTH close
    mu.clamp(-0.05, 0.05)    // cap to ±5% per day for sanity
}


// --------------------------- Cost schedule (DOLLAR units) ---------------------------

fn compute_cost_schedule(bars: &[Bar], cap_dollars: f64) -> Vec<f64> {
    // Tuned to make open/close meaningfully more expensive than mid-day.
    let alpha = 0.3_f64; // $ per $ of minute range (high-low)
    let floor = 0.01_f64; // $0.01 minimum
    let cap = cap_dollars.max(floor);
    let n = bars.len().max(1) as f64;
    let w = 6.0_f64; // edge width (minutes) for penalty bump
    let edge_weight = 1.0_f64 * cap; // up to full cap as edge add-on

    let mut raw: Vec<f64> = bars
        .iter()
        .enumerate()
        .map(|(t, b)| {
            let range = (b.high - b.low).max(0.0);
            let base = (floor + alpha * range).min(cap);
            // Edge penalty: Gaussian bumps at open & close
            let x = t as f64;
            let bump =
                (-(x / w).powi(2)).exp() + ( -((n - 1.0 - x) / w).powi(2)).exp(); // ~[0,2]
            let edge = edge_weight * (bump / 2.0); // normalize to ~[0, edge_weight]
            (base + edge).min(cap)
        })
        .collect();

    // Smooth a bit
    raw = moving_avg(&raw, 5);
    raw
}

// --------------------------- Our own GBM simulator ---------------------------

fn simulate_paths(
    mu: f64,
    sigma: f64,
    n_paths: usize,
    n_steps: usize,
    horizon: f64,
    s0: f64,
) -> Vec<Vec<f64>> {
    // Exact GBM:
    // S_{t+dt} = S_t * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z )
    let dt = horizon / (n_steps.saturating_sub(1).max(1) as f64);
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut rng = thread_rng();

    let mut paths = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let mut path = Vec::with_capacity(n_steps);
        let mut s = s0;
        path.push(s);
        for _ in 1..n_steps {
            let z = normal.sample(&mut rng);
            let step = drift + vol * z;
            s *= step.exp();
            path.push(s);
        }
        paths.push(path);
    }
    paths
}

fn average_effective_price_per_step(paths: &[Vec<f64>], costs: &[f64]) -> Vec<f64> {
    let n_steps = costs.len();
    let n_paths = paths.len() as f64;

    let mut sums = vec![0.0; n_steps];
    for path in paths {
        // assume path.len() == n_steps
        for (t, (&p, &c)) in path.iter().zip(costs.iter()).enumerate() {
            if p.is_finite() {
                sums[t] += p + c;
            }
        }
    }
    sums.into_iter().map(|s| s / n_paths).collect()
}

fn best_time_index(avg_effective: &[f64]) -> (usize, f64) {
    avg_effective
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, f64::NAN))
}



// --------------------------- Main ---------------------------

fn main() -> csv::Result<()> {
    let t0 = Instant::now();

    let csv_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "spy1day.csv".to_string());

    let bars = load_bars(&csv_path)?;
    if bars.is_empty() {
        eprintln!("No rows found in {}", csv_path);
        return Ok(());
    }

    let by_day = group_by_day_rth_weekdays(&bars);

    // Global config
    let horizon = 1.0;
    let n_paths_total: usize = 50_000; // lots of paths to reduce MC noise
    let batch_paths: usize = 1_000; // per-batch
    let cap_cost_dollars = 3.0; // max per-minute cost

    println!("date, best_time_ny, best_index, expected_effective_cost");

    let mut processed_days = 0usize;
    let mut skipped_days = 0usize;

    for (day, day_bars) in by_day {
        // Require a reasonably complete RTH day (typical ~390 minutes)
        if day_bars.len() < 300 {
            skipped_days += 1;
            continue;
        }

        // Per-day inputs
        let costs = compute_cost_schedule(&day_bars, cap_cost_dollars);
        let costs_len = costs.len();

        // Debug: min cost time
        let (min_cost_idx, min_cost_val) = costs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let ts_min = day_bars[min_cost_idx].timestamp;
        let dt_ny_min = New_York
            .timestamp_millis_opt(ts_min as i64)
            .single()
            .unwrap();
        eprintln!(
            "DEBUG {}-{:02}-{:02} min_cost @ {} idx={} cost={:.4}",
            day.year,
            day.month,
            day.day,
            dt_ny_min.format("%H:%M"),
            min_cost_idx,
            min_cost_val
        );

        let n_steps = costs_len;

        let sigma = estimate_sigma_from_day(&day_bars);
        let mu = estimate_mu_from_day(&day_bars); // *** NO DRIFT for this experiment ***
        let s0 = day_bars[0].close;

        // Monte Carlo: simulate n_paths_total paths with n_steps steps
        let mut done = 0usize;
        let mut running_sum = vec![0.0; n_steps];

        while done < n_paths_total {
            let this_batch = (n_paths_total - done).min(batch_paths);

            let paths = simulate_paths(mu, sigma, this_batch, n_steps, horizon, s0);

            let batch_avg = average_effective_price_per_step(&paths, &costs);

            for (t, v) in batch_avg.iter().enumerate() {
                running_sum[t] += v * (this_batch as f64);
            }

            done += this_batch;
        }

        let avg_eff: Vec<f64> = running_sum
            .into_iter()
            .map(|s| s / (n_paths_total as f64))
            .collect();

        let (best_idx, best_cost) = best_time_index(&avg_eff);

        // Map to NY time
        let best_idx_clamped = best_idx.min(day_bars.len().saturating_sub(1));
        let best_ts_ms = day_bars[best_idx_clamped].timestamp;
        let best_dt_ny = New_York
            .timestamp_millis_opt(best_ts_ms as i64)
            .single()
            .unwrap();

        println!(
            "{:04}-{:02}-{:02}, {}, {}, {:.4}",
            day.year,
            day.month,
            day.day,
            best_dt_ny.format("%H:%M"),
            best_idx,
            best_cost
        );

        processed_days += 1;
    }

    eprintln!(
        "Processed days: {}  (skipped: {})  total time: {:.3?}",
        processed_days,
        skipped_days,
        t0.elapsed()
    );

    Ok(())
}
