use criterion::{criterion_group, criterion_main, Criterion, BatchSize};

use game_of_life_avx::*;

fn create_universe() -> M256 {
    let universe_with_glider = vec![
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0,
        0, 1, 0, 1, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ];

    M256::from(&universe_with_glider)
}

fn bench_glider(universe: &mut M256) {
    for _ in 0..1024 {
        unsafe {
            universe.step();
        }
    }
}

fn bench(c: &mut Criterion) {
    let universe = create_universe();

    c.bench_function("glider_1000_steps", move |b| {
        b.iter_batched(|| universe.clone(), |mut universe| bench_glider(&mut universe), BatchSize::SmallInput)
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
