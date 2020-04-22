use criterion::{black_box, criterion_group, criterion_main, Criterion};
//use mycrate::fibonacci;
use graphica::geometry_utilities::sqr_distance_bezier_point;
use graphica::geometry_utilities::sqr_distance_bezier_point2;
use graphica::geometry_utilities::sqr_distance_bezier_point_binary;
use graphica::geometry_utilities::sqr_distance_bezier_point_simd;
use lyon::math::*;
use rand::prelude::*;
use rand::{Rng, SeedableRng};
use std::time::Instant;

use graphica::geometry_utilities::ParamCurvePointAtDistance;
use kurbo::CubicBez;
use kurbo::ParamCurve;
use kurbo::ParamCurveArclen;
use kurbo::Point as KurboPoint;

// use wgpu_example::main::PathData;

// fn bench_iter(data: &mut PathData) {
//     let mut k = point(0.0, 0.0);
//     //let t1 = Instant::now();
//     // let mut best = None;

//     for p in data.iter_sub_paths() {
//         // assert_eq!(p.iter_points().count(), 4);
//         for point in p.iter_points() {
//             // dbg!(point.control_before());
//             // dbg!(point.position());
//             // dbg!(point.control_after());
//             // dbg!(point.next().control_before());
//             k += point.position().to_vector();
//             k += point.control_before().to_vector();
//             //best = Some(point);
//         }
//     }
//     // let id = best.unwrap().index();
//     // data.remove(id);
//     //println!("Test1: {:#?}", t1.elapsed());
//     black_box(k);
//     //dbg!(k);
// }

// fn bench_iter_beziers(data: &PathData) {
//     let mut k = point(0.0, 0.0);
//     //let t1 = Instant::now();
//     // let mut best = None;
//     for p in data.iter_sub_paths() {
//         for a in p.iter_beziers() {
//             k += a.position().to_vector();
//             k += a.next().position().to_vector();
//         }
//     }
//     black_box(k);
// }

// fn bench_iter2(data: &PathData) {
//     let mut k = point(0.0, 0.0);
//     //let t1 = Instant::now();
//     for r in &data.sub_paths {
//         for i in r.range.clone().step_by(3) {
//             let point = data.points[i];
//             k += point.to_vector();
//             let j = if i == r.range.start { r.range.start } else { i - 1 };
//             k += point.to_vector() + data.points[j].to_vector();
//         }
//     }
//     //println!("Test2: {:#?}", t1.elapsed());
//     black_box(k);
// }

fn bench3() {
    fn approx_parabola_integral(x: f32) -> f32 {
        let d = 0.67f32;
        let quarter = 0.25f32;
        x / (1.0f32 - d + (d.powi(4) + quarter * x * x).powf(quarter))
    }

    let k = approx_parabola_integral(black_box(0.5));
    black_box(k);
}

fn sqr_distance_bezier_point_bench_simd(points: &Vec<Point>) {
    for i in 0..100 {
        let k = sqr_distance_bezier_point_simd(points[i], points[i + 1], points[i + 2], points[i + 3], points[i + 4]);
        black_box(k);
    }
}

fn sqr_distance_bezier_point_bench_old(points: &Vec<Point>) {
    for i in 0..100 {
        let k = sqr_distance_bezier_point(points[i], points[i + 1], points[i + 2], points[i + 3], points[i + 4]);
        black_box(k);
    }
}

fn sqr_distance_bezier_point_bench_naive(points: &Vec<Point>) {
    for i in 0..100 {
        let k = sqr_distance_bezier_point2(points[i], points[i + 1], points[i + 2], points[i + 3], points[i + 4]);
        black_box(k);
    }
}

fn sqr_distance_bezier_point_bench_binary(points: &Vec<Point>) {
    for i in 0..100 {
        let k = sqr_distance_bezier_point_binary(points[i], points[i + 1], points[i + 2], points[i + 3], points[i + 4]);
        black_box(k);
    }
}

fn bezier_point_at_distance(bez: &CubicBez) {
    for i in 0..10 {
        let k = bez.point_at_distance(i as f64 * 1.0, 0.01);
        black_box(k);
    }
}

fn bezier_arclen(bez: &CubicBez) {
    for i in 0..10 {
        let k = bez.arclen(0.1);
        black_box(k);
    }
}

fn bezier_move_forward_distance(p0: &Point, p1: &Point, p2: &Point, p3: &Point) {
    for i in 0..10 {
        let k =
            graphica::geometry_utilities::bezier_move_forward_distance(*p0, *p1, *p2, *p3, 0.0, i as f32 * 1.0, 1.0);
        black_box(k);
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    // let mut data = PathData::new();
    // for _ in 0..1000 {
    //     data.add_circle(point(0.0, 0.0), 5.0);
    // }
    // assert_eq!(data.iter_sub_paths().count(), 1000);

    // c.bench_function("bench_iter", |b| b.iter(|| bench_iter(&mut data)));
    // c.bench_function("bench_iter2", |b| b.iter(|| bench_iter2(&data)));
    // c.bench_function("bench_iter_beziers", |b| b.iter(|| bench_iter_beziers(&data)));

    // c.bench_function("bench_parabola_approx", |b| b.iter(|| bench3()));
    let seed: &[u8] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::seed_from_u64(0);
    let mut points = vec![];
    for _ in 0..100 {
        for _ in 0..5 {
            points.push(point(rng.gen(), rng.gen()));
        }
    }
    c.bench_function("sqr_distance_bezier_point_bench_binary", |b| {
        b.iter(|| sqr_distance_bezier_point_bench_binary(&points))
    });
    c.bench_function("sqr_distance_bezier_point_bench_simd", |b| {
        b.iter(|| sqr_distance_bezier_point_bench_simd(&points))
    });
    c.bench_function("sqr_distance_bezier_point_bench_old", |b| {
        b.iter(|| sqr_distance_bezier_point_bench_old(&points))
    });
    c.bench_function("sqr_distance_bezier_point_bench_naive", |b| {
        b.iter(|| sqr_distance_bezier_point_bench_naive(&points))
    });

    let p0 = KurboPoint::new(0.0, 10.0);
    let p1 = KurboPoint::new(10.0, 30.0);
    let p2 = KurboPoint::new(5.0, 20.0);
    let p3 = KurboPoint::new(0.0, 20.0);
    let bezier = CubicBez::new(p0, p1, p2, p3);

    let c0 = point(0.0, 10.0);
    let c1 = point(10.0, 30.0);
    let c2 = point(5.0, 20.0);
    let c3 = point(0.0, 20.0);
    c.bench_function("bezier_point_at_distance", |b| {
        b.iter(|| bezier_point_at_distance(&bezier))
    });
    c.bench_function("bezier_arclen", |b| b.iter(|| bezier_arclen(&bezier)));
    c.bench_function("bezier_move_forward_distance", |b| {
        b.iter(|| bezier_move_forward_distance(&c0, &c1, &c2, &c3))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
