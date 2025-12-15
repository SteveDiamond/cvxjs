use clarabel::algebra::*;
use clarabel::solver::*;

fn main() {
    // min x s.t. x >= 1
    // Standard form: min x s.t. -x + s = -1, s >= 0
    // A = [[-1]], b = [-1], K = Nonneg(1)

    let n = 1;
    let m = 1;

    // P = 0 (zero matrix)
    let p = CscMatrix::new(n, n, vec![0, 0], vec![], vec![]);

    // q = [1]
    let q = vec![1.0];

    // A = [[-1]]
    let a = CscMatrix::new(m, n, vec![0, 1], vec![0], vec![-1.0]);

    // b = [-1]
    let b = vec![-1.0];

    // Cones
    let cones = vec![NonnegativeConeT(1)];

    let settings = DefaultSettingsBuilder::default()
        .verbose(true)
        .build()
        .unwrap();

    println!("Creating solver...");
    let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings);
    println!("Solver created, solving...");
    solver.solve();
    println!("Solved!");

    println!("Status: {:?}", solver.solution.status);
    println!("Obj val: {}", solver.solution.obj_val);
    println!("x: {:?}", solver.solution.x);
}
