# Monte Carlo Intraday Optimal Trade Timing

This project identifies the **best time of day to execute a trade** in SPY using historical minute-level OHLC data. It combines a deterministic execution cost model with a **Monte Carlo simulation** of Geometric Brownian Motion (GBM) to estimate the expected effective cost of trading at each minute. The optimal execution time is the minute with the lowest expected effective cost.

---

## Overview

Executing a large trade intraday involves both **price risk** and **execution cost**, which vary throughout the trading session. Using 1173 days of historical SPY minute bars, this project:

- Estimates drift (μ) and volatility (σ) from intraday log returns  
- Builds a smoothed execution cost curve based on market conditions  
- Simulates tens of thousands of GBM paths per day  
- Computes expected effective cost across all minutes  
- Selects the minute with the **minimum expected cost**

---

## Features

- Load & preprocess intraday minute-level OHLC data (CSV)
- Convert timestamps to New York time & filter regular trading hours  
- Compute daily μ and σ  
- Smooth execution cost curve using microstructure-based modeling  
- Simulate **50,000+** Monte Carlo paths per day  
- Parallelized simulation and aggregation using Rayon  
- Output optimal trade time per day

---

## Data Structures

### Inputs
- CSV of intraday OHLC bars  
- Hyperparameters: number of paths, batch size, cost cap  
- Stored as:  
  ```rust
  Vec<Bar>

### Outputs
Outputs For each trading day: 
- Date 
- Best execution time (NY) 
- Best minute index 
- Expected effective cost 
- Represented as:
    struct DayResult {
        date: DayKey,
        best_time_ny: DateTime<Tz>,
        best_index: usize,
        expected_effective_cost: f64,
    }

## Parallel Programming Component


The Monte Carlo simulation dominates runtime. Parallelism is applied using Rayon:
- Path Generation:
    Each thread independently creates full GBM paths using its own RNG:
    (0..n_paths).into_par_iter()
- Cost Aggregation:
    Parallelization over time steps ensures no shared state or locking:
    (0..n_steps).into_par_iter()

## Issues Addressed

- Randomness: Each thread uses its own thread_rng() to ensure independent, thread-safe sampling.

- Data Races: Parallelizing over time steps avoids shared mutable state and eliminates lock contention.

## Evaluation

Both serial and parallel implementations were run 10 times each.

Single-Day Dataset

Serial: 1.234 s

Parallel: 0.238 s

Speedup: 5.18×

Full Dataset (1,173 Days)

Serial: 894.497 s

Parallel: 168.9 s

Speedup: 5.29×

Parallelizing reduces runtime by ~5× across both small and large datasets.

## Build & Run

  ```rust
cargo build --release

./target/release/monte spy1day.csv

rm -f times.txt
for i in {1..10}; do
  /usr/bin/time -f "%e" ./target/release/monte spy1day.csv 2>> times.txt
done

awk '{sum+=$1} END {print "Average:", sum/NR}' times.txt
```

