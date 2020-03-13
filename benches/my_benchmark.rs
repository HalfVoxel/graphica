use criterion::{black_box, criterion_group, criterion_main, Criterion};
//use mycrate::fibonacci;
use lyon::math::*;
use std::time::Instant;
use wgpu_example::main::PathData;

fn bench_iter(data: &mut PathData) {
    let mut k = point(0.0, 0.0);
    //let t1 = Instant::now();
    // let mut best = None;

    for p in data.iter_sub_paths() {
        // assert_eq!(p.iter_points().count(), 4);
        for point in p.iter_points() {
            // dbg!(point.control_before());
            // dbg!(point.position());
            // dbg!(point.control_after());
            // dbg!(point.next().control_before());
            k += point.position().to_vector();
            k += point.control_before().to_vector();
            //best = Some(point);
        }
    }
    // let id = best.unwrap().index();
    // data.remove(id);
    //println!("Test1: {:#?}", t1.elapsed());
    black_box(k);
    //dbg!(k);
}

fn bench_iter_beziers(data: &PathData) {
    let mut k = point(0.0, 0.0);
    //let t1 = Instant::now();
    // let mut best = None;
    for p in data.iter_sub_paths() {
        for a in p.iter_beziers() {
            k += a.position().to_vector();
            k += a.next().position().to_vector();
        }
    }
    black_box(k);
}

fn bench_iter2(data: &PathData) {
    let mut k = point(0.0, 0.0);
    //let t1 = Instant::now();
    for r in &data.sub_paths {
        for i in r.range.clone().step_by(3) {
            let point = data.points[i];
            k += point.to_vector();
            let j = if i == r.range.start { r.range.start } else { i - 1 };
            k += point.to_vector() + data.points[j].to_vector();
        }
    }
    //println!("Test2: {:#?}", t1.elapsed());
    black_box(k);
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut data = PathData::new();
    for _ in 0..1000 {
        data.add_circle(point(0.0, 0.0), 5.0);
    }
    assert_eq!(data.iter_sub_paths().count(), 1000);

    c.bench_function("bench_iter", |b| b.iter(|| bench_iter(&mut data)));
    c.bench_function("bench_iter2", |b| b.iter(|| bench_iter2(&data)));
    c.bench_function("bench_iter_beziers", |b| b.iter(|| bench_iter_beziers(&data)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
