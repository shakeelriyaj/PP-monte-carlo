use stochastic_gbm::gbm::GeometricBrownianMotion;
use csv::ReaderBuilder;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Bar {
    timestamp: u64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
}

fn load_bars(path : &str) -> csv::Result<Vec<Bar>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let mut bars = Vec::new();
    for rec in rdr.deserialize::<Bar>() {
        bars.push(rec?);
    }
    Ok(bars)
}

fn compute_cost_schedule(bars: &[Bar], max_cost_dollars: f64) -> Vec<f64> {
    let mut raw: Vec<f64> = bars
        .iter()
        .map(|b| ((b.high - b.low) / b.close).max(0.0))
        .collect();

    let max_raw = raw
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(1e-12);

    for c in &mut raw {
        *c = *c / max_raw * max_cost_dollars;
    }
    raw
}

fn simulate_paths(
    // - Drift (mu): 0.2
    // - Volatility (sigma): 0.4
    // - Number of paths: 50
    // - Number of steps: 200
    // - Time horizon: 1.0
    // - Initial asset value (S_0): 500.0

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
    if costs.is_empty() {
        return vec![0.0; n_steps];
    }
    if costs.len() == n_steps {
        return costs.to_vec();
    }
    (0..n_steps)
        .map(|t| {
            let idx = ((t as f64) * (costs.len() as f64 - 1.0) / (n_steps as f64 - 1.0)).round()
                as usize;
            costs[idx.min(costs.len() - 1)]
        })
        .collect()
}

fn average_effective_price_per_step(paths: &[Vec<f64>], costs: &[f64]) -> Vec<f64> {
    let n_steps = costs.len();
    let mut sums = vec![0.0; n_steps];
    for path in paths {
        for (t, (&p, &c)) in path.iter().zip(costs.iter()).enumerate() {
            sums[t] += p + c;
        }
    }
    for s in &mut sums { *s /= paths.len() as f64; }
    sums
}

fn best_time_index(avg_effective: &[f64]) -> (usize, f64) {
    avg_effective
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, f64::NAN))
}



//spyususd-m1-bid-2022-01-01-2025-03-19.csv
fn main() -> csv::Result<()> {
    let bars = load_bars("spyususd-m1-bid-2022-01-01-2025-03-19.csv")?;
    if bars.is_empty() {
        eprintln!("No rows found in csv file");
        return Ok(());
    }

    let data_costs = compute_cost_schedule(&bars, 0.50);

    let mu = 0.2;
    let sigma = 0.4;
    let n_paths = 10000;
    let n_steps = 200;
    let horizon = 1.0;
    let s0 = bars[0].open;

    let costs_len = data_costs.len();
    let gbm_steps = costs_len.saturating_sub(1);

    let paths = simulate_paths(mu, sigma, n_paths, gbm_steps, horizon, s0);

    let path_len = paths[0].len();
    assert_eq!(path_len, costs_len, "GBM path len != costs length");

    let costs = align_costs_to_steps(&data_costs, path_len);

    let avg_eff = average_effective_price_per_step(&paths, &costs);

    let (best_idx, best_cost) = best_time_index(&avg_eff);

    println!("Best time index (0..{}): {}", n_steps - 1, best_idx);
    println!("Expected effective cocst at best time: {:.4}", best_cost);

    println!(
        "Min/Max effective: {:.4} / {:.4}",
        avg_eff
            .iter()
            .fold(f64::INFINITY, |m, &x| m.min(x)),
        avg_eff
            .iter()
            .fold(f64::NEG_INFINITY, |m, &x| m.max(x))
    );

    Ok(())
}

