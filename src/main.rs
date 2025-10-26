use stochastic_gbm::gbm::GeometricBrownianMotion;
fn main() {
    // - Drift (mu): 0.2
    // - Volatility (sigma): 0.4
    // - Number of paths: 50
    // - Number of steps: 200
    // - Time horizon: 1.0
    // - Initial asset value (S_0): 500.0
    let gbm = GeometricBrownianMotion::new(0.2, 0.4, 50, 200, 1.0, 500.0);
    let paths = gbm.simulate();
    for path in paths {
        println!("{:?}", path);
    }
}
