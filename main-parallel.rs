use csv::ReaderBuilder;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::time::Instant;

use chrono::{Datelike, TimeZone, Timelike, Utc, Weekday};
use chrono_tz::America::New_York;

use rand::thread_rng;
use rand_distr::{Distribution, Normal};

use rayon::prelude::*;

#[derive(Debug, Deserialize)]
struct Bar {
    timestamp : u64,
    open : f64,
    high : f64,
    low : f64,
    close: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct DayKey {
    year : i32,
    month : u32,
    day : u32
}

// Now we need to take the file in and create a vector for 
// each row of data that we read. This will return a vector
// of bars for each row that it reads from the file.
fn load_bars(path : &str) -> csv::Result<Vec<Bar>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let mut bars = Vec::new();

    for data in reader.deserialize::<Bar>() {
        bars.push(data?);
    }
    Ok(bars)
}

// We want to use only the regular trading hours, so we can change it into NY time. 
fn sort_regular_trading_day( bars : &[Bar]) -> BTreeMap<DayKey, Vec<Bar>> {
    let trading_day_start = (9,30,0);
    let trading_day_end = (16,0,0);
    let mut map : BTreeMap<DayKey, Vec<Bar>> = BTreeMap::new();

    for minute in bars {
        let dt_utc = match Utc.timestamp_millis_opt(minute.timestamp as i64).single() {
            Some(t) => t,
            None => continue,
        };
        let dt_ny = dt_utc.with_timezone(&New_York);

        //now we need to skip the weekends.
        let weekday = dt_ny.weekday();
        if weekday == Weekday::Sat || weekday == Weekday::Sun {
            continue;
        }

        // now we need to filter out only the regular trading hours minutes
        let t = dt_ny.time();
        let in_trading_hours = (t.hour(), t.minute(), t.second()) >= trading_day_start
            && (t.hour(), t.minute(), t.second()) < trading_day_end;
        if !in_trading_hours {
            continue;
        }

        let key = DayKey {
            year : dt_ny.year(),
            month : dt_ny.month(),
            day : dt_ny.day(),
        };

        map.entry(key).or_default().push(Bar {
            timestamp : minute.timestamp,
            open : minute.open,
            high : minute.high,
            low : minute.low,
            close : minute.close,
        });
    }

    for (_k, v) in map.iter_mut() {
        v.sort_by_key(|minute| minute.timestamp);
    }
    map
}

//So now we have an ordered series of trading minutes per trading day.
// All we have to do is compute a simple moving average over size k.
// This will smooth out later minute costs.

fn moving_avg( x : &[f64], k : usize) -> Vec<f64> {
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

//Now we need our volitility/sigma value. We will use the minute log returns
// This is the minute to minute log returns ln(p1/p0) using the close pricing. 
// It will calculate the sample variance and std per minute. 
// We assume the daily volatility is minute std * sqrt(# of minutes)
// Sigma is between 0.5% and 5% per day
fn estimate_sigma_from_day(trading_day : &[Bar]) -> f64 {
    if trading_day.len() < 2 {
        return 0.01;
    }
    let mut returns = Vec::with_capacity(trading_day.len() - 1);
    for window in trading_day.windows(2) {
        let first_price = window[0].close;
        let second_price = window[1].close;
        if first_price > 0.0 && second_price.is_finite() && first_price.is_finite() {
            let log_return = (second_price / first_price).ln();
            if log_return.is_finite() {
                returns.push(log_return);
            }
        }
    }

    if returns.len() < 2 {
        return 0.01;
    }

    let mean = returns.iter().sum::<f64>() / (returns.len() as f64);

    let variance = returns
        .iter()
        .map(|log_return| (log_return - mean).powi(2))
        .sum::<f64>()
        / ((returns.len() - 1) as f64);
    
    let std_step = variance.max(0.0).sqrt();
    let n_steps = trading_day.len().max(1) as f64;
    let sigma = std_step * n_steps.sqrt();
    sigma.clamp(0.005, 0.05)

}

// okay now we need to compute the drift that goes into the monte carlo.
// this is taken from the data that we give in the csv, along with sigma
// We take the log return from the first trading close to the last, ln(s1/s0)
// this is the daily drift, 
fn estimate_mu_from_day(trading_day : &[Bar]) -> f64 {
    if trading_day.is_empty() {
        return 0.0;
    }
    let first_price = trading_day.first().unwrap().close;
    let last_price = trading_day.last().unwrap().close;
    if first_price <= 0.0 || !first_price.is_finite() || !last_price.is_finite() {
        return 0.0;
    }
    let mu = (last_price / first_price).ln();
    mu.clamp(-0.05, 0.05)
}

// We now compute the per minute cost in dollars for trading at that minute.
// It depends on the volatility, which is base = floor + alpha * (high - low)
// We also add in a penalty for open and close buys simply because that corrects the model.

fn compute_trading_cost( bars : &[Bar], cap_dollars : f64) -> Vec<f64> {
    let alpha = 0.3_f64;
    let floor = 0.01_f64;
    let cap = cap_dollars.max(floor);
    let minutes = bars.len().max(1) as f64;
    let min_buy_time = 6.0_f64;
    let edge_weight = 1.0_f64 * cap;
//raw is simply the initial, unsmoothed vector of per-minute trading costs. 
// It represents the raw cost curve before applying the moving average smoothing at the end of the function.
    let mut raw : Vec<f64> = bars
        .iter()
        .enumerate()
        .map(|(t, minute)| {
            let range = (minute.high - minute.low).max(0.0);
            let base = (floor + alpha * range).min(cap);
            let x = t as f64;
            let bump =
                (-(x / min_buy_time).powi(2)).exp() + ( -((minutes - 1.0 - x) / min_buy_time).powi(2)).exp();
            let edge = edge_weight * (bump / 2.0);
            (base + edge).min(cap)
        }).collect();
    
    raw = moving_avg(&raw, 5);
    raw
}


// Now we need our own GBM simulator. 
// 1. mu (drift)
// A float representing the expected return over the day.
// Example: if the stock tends to rise 0.1% per day → mu = 0.001.
// 2. sigma (volatility)
// Daily volatility estimate.
// Example: if daily vol = 2% → sigma = 0.02.
// 3. n_paths
// How many simulated price paths to generate (e.g. 1,000).
// 4. n_steps
// Number of time steps per path (usually number of minutes in day).
// Typical value ≈ 390.
// 5. horizon
// Total time span of simulation.
// Usually 1.0 = one day.
// 6. s0
// Starting price — the close price of the first trading minute.
//


//into_par_iter() creates a parallel iterator, splitting the range across worker threads.
// Each worker thread:
// gets its own thread_rng() (safe and independent),
// generates one full path.
// previously, simulate_paths loops through 0..n_paths serially.
fn simulate_paths(
    mu : f64,
    sigma : f64,
    n_paths : usize,
    n_steps : usize,
    horizon : f64,
    s0 : f64,
) -> Vec<Vec<f64>> {
    let dt = horizon / (n_steps.saturating_sub(1).max(1) as f64);
    let drift = (mu - 0.5 * sigma * sigma) * dt;
    let vol = sigma * dt.sqrt();

    let normal = Normal::new(0.0, 1.0).unwrap();

    (0..n_paths)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let mut path = Vec::with_capacity(n_steps);
            let mut s = s0;
            path.push(s);
            for _ in 1..n_steps {
                let z = normal.sample(&mut rng);
                let step = drift + vol * z;
                s *= step.exp();
                path.push(s);
            }
            path
        })
        .collect()
    }


// This function computes the expected effective cost of trading at each minute of the day.
// Effective cost = simulated price at minute t + execution cost at minute t
// Then it averages this value across all Monte Carlo paths.
//input made by the monte carlo paths = [
//   [500.0, 500.3, 500.1, ...],   // path 0
//   [500.0, 499.9, 499.7, ...],   // path 1
//   [500.0, 500.4, 500.8, ...],   // path 2
//   ...
// ]
//input vector giving costs per minute = [cost_0, cost_1, cost_2, ...]


fn average_effective_price_per_step(paths : &[Vec<f64>], costs : &[f64]) -> Vec<f64> {
    let n_steps = costs.len();
    let n_paths = paths.len() as f64;

    let mut sums = vec![0.0; n_steps];
    for path in paths {
        for (t, (&p, &c)) in path.iter().zip(costs.iter()).enumerate() {
            if p.is_finite() {
                sums[t] += p + c;
            }
        }
    }
    sums.into_iter().map(|s| s / n_paths).collect()
}

// This function finds The minute index where the expected effective cost is lowest
// (i.e., the best time of day to trade)
// Along with the actual value of that minimum cost
// So given a vector like:
// avg_effective = [502.3, 501.8, 500.9, 500.5, 501.1, ...]
// It returns:
// (best_index, best_value) = (3, 500.5)
// Because 500.5 is the lowest number.

fn best_time_index(avg_effective_price : &[f64]) -> (usize, f64) {
    avg_effective_price
        .iter()
        .enumerate() // convert to an iterator and add indices, so now we can find index and value
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap()) //a and b are pairs like (index, value) a.1 is the value (the second element) partial_cmp compares floating point numbers
        .map(|(i, &v)| (i,v))//convert to a tuple, with min index and value
        .unwrap_or((0, f64::NAN))
}

// cool now we can run our main.
fn main() -> csv::Result<()> {
    let current_time = Instant::now();
    let csv_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "spyususd-m1-bid-2022-01-01-2025-03-19.csv".to_string());

    let bars = load_bars(&csv_path)?;
    if bars.is_empty() {
        eprintln!("No rows found in {}", csv_path);
        return Ok(());
    }

    let by_day = sort_regular_trading_day(&bars);
    // take in all the data, and put them into regular trading days
    // simulate 50,000 paths per day, in batches of 1000 for performance
    // the max per minute cost is 3.0.

    let horizon = 1.0;
    let n_paths_total : usize = 50000;
    let batch_paths : usize = 1000;
    let cap_cost_dollars = 3.0;
    println!("date, best_time_ny, best_index, expected_effective_cost");

    let mut processed_days = 0usize;
    let mut skipped_days = 0usize;

    for (day, minute) in by_day {
        if minute.len() < 300 {
            skipped_days += 1;
            continue;
        }
        let costs = compute_trading_cost(&minute, cap_cost_dollars);
        let costs_len = costs.len();

        let (min_cost_idx, min_cost_val) = costs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let ts_min = minute[min_cost_idx].timestamp;
        let dt_ny_min = New_York
            .timestamp_millis_opt(ts_min as i64)
            .single()
            .unwrap();
        // eprintln!(
        //     "DEBUG {}-{:02}-{:02} min_cost @ {} idx={} cost={:.4}",
        //     day.year,
        //     day.month,
        //     day.day,
        //     dt_ny_min.format("%H:%M"),
        //     min_cost_idx,
        //     min_cost_val
        // );

        let n_steps = costs_len;
        let sigma = estimate_sigma_from_day(&minute);
        let mu = estimate_mu_from_day(&minute);
        let starting_price = minute[0].close;

        let mut done = 0usize;
        let mut running_sum = vec![0.0; n_steps];

        while done < n_paths_total {
            let this_batch = (n_paths_total - done).min(batch_paths);
            let paths = simulate_paths(mu, sigma, this_batch, n_steps, horizon, starting_price);
            let batch_avg = average_effective_price_per_step(&paths, &costs);

            for (t,v) in batch_avg.iter().enumerate() {
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
        let best_idx_clamped = best_idx.min(minute.len().saturating_sub(1));
        let best_ts_ms = minute[best_idx_clamped].timestamp;
        let best_dt_ny = New_York
            .timestamp_millis_opt(best_ts_ms as i64)
            .single()
            .unwrap();

        // println!(
        //     "{:04}-{:02}-{:02}, {}, {}, {:.4}",
        //     day.year,
        //     day.month,
        //     day.day,
        //     best_dt_ny.format("%H:%M"),
        //     best_idx,
        //     best_cost
        // );
        
        // processed_days += 1;

      
        // eprintln!(
        //     "Processed days: {}  (skipped: {})  total time: {:.3?}",
        //     processed_days,
        //     skipped_days,
        //     current_time.elapsed()
        // );

    }
    Ok(())

}
