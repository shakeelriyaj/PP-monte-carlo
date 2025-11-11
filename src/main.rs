use csv::ReaderBuilder;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::time::Instant;

use chrono::{Datelike, TimeZone, Timelike, Utc};
use chrono_tz::America::New_York;
use stochastic_gbm::gbm::GeometricBrownianMotion;

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
struct DayKey { year: i32, month: u32, day: u32 }

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

// Only RTH minutes (09:30–16:00 America/New_York)
fn group_by_day_rth(bars: &[Bar]) -> BTreeMap<DayKey, Vec<Bar>> {
    let rth_start = (9, 30, 0);
    let rth_end = (16, 0, 0);
    let mut map: BTreeMap<DayKey, Vec<Bar>> = BTreeMap::new();

    for b in bars {
        let dt_utc = match Utc.timestamp_millis_opt(b.timestamp as i64).single() {
            Some(t) => t,
            None => continue,
        };
        let dt_ny = dt_utc.with_timezone(&New_York);
        let t = dt_ny.time();

        let in_rth = (t.hour(), t.minute(), t.second()) >= rth_start
            && (t.hour(), t.minute(), t.second()) < rth_end;
        if !in_rth { continue; }

        let key = DayKey { year: dt_ny.year(), month: dt_ny.month(), day: dt_ny.day() };
        map.entry(key).or_default().push(Bar {
            timestamp: b.timestamp, open: b.open, high: b.high, low: b.low, close: b.close
        });
    }
    for (_k, v) in map.iter_mut() {
        v.sort_by_key(|b| b.timestamp);
    }
    map
}

// --------------------------- Small utilities ---------------------------

fn moving_avg(x: &[f64], k: usize) -> Vec<f64> {
    if k <= 1 || x.is_empty() { return x.to_vec(); }
    let mut out = vec![0.0; x.len()];
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += x[i];
        if i >= k { sum -= x[i - k]; }
        out[i] = sum / ((i + 1).min(k) as f64);
    }
    out
}

// Estimate daily sigma from minute log-returns; clamp to sane range
fn estimate_sigma_from_day(day_bars: &[Bar]) -> f64 {
    if day_bars.len() < 2 { return 0.01; }
    let mut rets = Vec::with_capacity(day_bars.len() - 1);
    for w in day_bars.windows(2) {
        let p0 = w[0].close;
        let p1 = w[1].close;
        if p0 > 0.0 && p1.is_finite() && p0.is_finite() {
            let r = (p1 / p0).ln();
            if r.is_finite() { rets.push(r); }
        }
    }
    if rets.len() < 2 { return 0.01; }
    let mean = rets.iter().sum::<f64>() / (rets.len() as f64);
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / ((rets.len() - 1) as f64);
    let std_step = var.max(0.0).sqrt();
    let n_steps = day_bars.len().max(1) as f64;
    let sigma = std_step * n_steps.sqrt();
    sigma.clamp(0.005, 0.05) // ~0.5%..5% daily
}

// --------------------------- Cost schedule (DOLLAR units) ---------------------------
// IMPORTANT: no per-day normalization here. Use raw dollar ranges so days differ naturally.

fn compute_cost_schedule(bars: &[Bar], cap_dollars: f64) -> Vec<f64> {
    let alpha = 0.5_f64;       // scale from range to cost ($)
    let floor = 0.01_f64;      // $0.01 minimum cost per minute
    let cap = cap_dollars.max(floor);

    // raw dollar cost per minute
    let raw: Vec<f64> = bars.iter().map(|b| {
        let range = (b.high - b.low).max(0.0);
        let c = floor + alpha * range;
        c.min(cap)
    }).collect();

    // smooth (5-min MA)
    moving_avg(&raw, 5)
}

// --------------------------- Simulation (keep your names) ---------------------------

fn simulate_paths(
    mu: f64,
    sigma: f64,
    n_paths: usize,
    n_steps: usize,
    horizon: f64,
    s0: f64,
) -> Vec<Vec<f64>> {
    let gbm = GeometricBrownianMotion::new(mu, sigma, n_paths, n_steps, horizon, s0);
    gbm.simulate()
}

fn align_costs_to_steps(costs: &[f64], n_steps: usize) -> Vec<f64> {
    if costs.is_empty() { return vec![0.0; n_steps]; }
    if costs.len() == n_steps { return costs.to_vec(); }
    (0..n_steps).map(|t| {
        let idx = ((t as f64) * (costs.len() as f64 - 1.0) / (n_steps as f64 - 1.0)).round() as usize;
        costs[idx.min(costs.len() - 1)]
    }).collect()
}

fn average_effective_price_per_step(paths: &[Vec<f64>], costs: &[f64]) -> Vec<f64> {
    let n_steps = costs.len();
    let n_paths = paths.len() as f64;

    let mut sums = vec![0.0; n_steps];
    // Only require finite (no “outlier” rejection that could zero everything)
    for path in paths {
        for (t, (&p, &c)) in path.iter().zip(costs.iter()).enumerate() {
            if p.is_finite() { sums[t] += p + c; }
        }
    }
    sums.into_iter().map(|s| s / n_paths).collect()
}

fn best_time_index(avg_effective: &[f64]) -> (usize, f64) {
    avg_effective.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i, v)).unwrap_or((0, f64::NAN))
}

// --------------------------- Main: ALL days ---------------------------

fn main() -> csv::Result<()> {
    let t0 = Instant::now();

    let csv_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "spyususd-m1-bid-2022-01-01-2025-03-19.csv".to_string());

    let bars = load_bars(&csv_path)?;
    if bars.is_empty() {
        eprintln!("No rows found in {}", csv_path);
        return Ok(());
    }

    let by_day = group_by_day_rth(&bars);

    // Global config
    let horizon = 1.0;
    let n_paths_total: usize = 200; // paths per day (increase later)
    let batch_paths: usize = 50;    // per-batch
    let cap_cost_dollars = 0.50;

    println!("date, best_time_ny, best_index, expected_effective_cost");

    let mut processed_days = 0usize;
    let mut skipped_days = 0usize;

    for (day, day_bars) in by_day {
        // Require a reasonably complete RTH day
        if day_bars.len() < 300 {
            skipped_days += 1;
            continue;
        }

        // Per-day inputs
        let costs = compute_cost_schedule(&day_bars, cap_cost_dollars);
        let costs_len = costs.len();
        let gbm_steps = costs_len.saturating_sub(1);

        let sigma = estimate_sigma_from_day(&day_bars);
        let mu = 0.0;
        let s0 = day_bars[0].close;

        // Batch simulate + average
        let mut done = 0usize;
        let mut running_sum = vec![0.0; costs_len];

        while done < n_paths_total {
            let this_batch = (n_paths_total - done).min(batch_paths);
            let paths = simulate_paths(mu, sigma, this_batch, gbm_steps, horizon, s0);

            let costs_aligned = align_costs_to_steps(&costs, paths[0].len());
            let batch_avg = average_effective_price_per_step(&paths, &costs_aligned);

            for (t, v) in batch_avg.iter().enumerate() {
                running_sum[t] += v * (this_batch as f64);
            }
            done += this_batch;
        }

        let avg_eff: Vec<f64> = running_sum.into_iter()
            .map(|s| s / (n_paths_total as f64))
            .collect();

        let (best_idx, best_cost) = best_time_index(&avg_eff);

        // Map to NY time
        let best_idx_clamped = best_idx.min(day_bars.len().saturating_sub(1));
        let best_ts_ms = day_bars[best_idx_clamped].timestamp;
        let best_dt_ny = New_York.timestamp_millis_opt(best_ts_ms as i64).single().unwrap();

        println!(
            "{:04}-{:02}-{:02}, {}, {}, {:.4}",
            day.year, day.month, day.day,
            best_dt_ny.format("%H:%M"),
            best_idx, best_cost
        );

        processed_days += 1;
    }

    eprintln!(
        "Processed days: {}  (skipped: {})  total time: {:.3?}",
        processed_days, skipped_days, t0.elapsed()
    );

    Ok(())
}
