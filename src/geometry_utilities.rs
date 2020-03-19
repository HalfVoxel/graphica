use crate::path::ImmutablePathPoint;
use euclid;
use lyon::math::*;
use rand::Rng;
use types::*;
use lazy_static::lazy_static;
use packed_simd::*;
use packed_simd::f32x4;
use std::convert::TryInto;

pub mod types {
    pub struct ScreenSpace;
    pub struct CanvasSpace;
    pub type ScreenPoint = euclid::Point2D<f32, ScreenSpace>;
    pub type ScreenVector = euclid::Vector2D<f32, ScreenSpace>;
    pub type CanvasPoint = euclid::Point2D<f32, CanvasSpace>;
    pub type CanvasVector = euclid::Vector2D<f32, CanvasSpace>;
    pub type CanvasLength = euclid::Length<f32, CanvasSpace>;
    pub type ScreenLength = euclid::Length<f32, ScreenSpace>;
    pub type CanvasRect = euclid::Rect<f32, CanvasSpace>;
}

pub enum VectorFieldPrimitive {
    Curl {
        center: CanvasPoint,
        strength: f32,
        radius: f32,
    },
    Linear {
        direction: CanvasVector,
        strength: f32,
    },
}

pub struct VectorField {
    pub primitives: Vec<VectorFieldPrimitive>,
}

pub fn evalute_cubic_bezier<U>(
    p0: euclid::Point2D<f32, U>,
    p1: euclid::Point2D<f32, U>,
    p2: euclid::Point2D<f32, U>,
    p3: euclid::Point2D<f32, U>,
    t: f32,
) -> euclid::Point2D<f32, U> {
    let p0 = p0.to_untyped().to_vector();
    let p1 = p1.to_untyped().to_vector();
    let p2 = p2.to_untyped().to_vector();
    let p3 = p3.to_untyped().to_vector();
    let t1 = 1.0 - t;
    let t2 = t1 * t1;
    let t3 = t1 * t1 * t1;
    (p0 * t3 + p1 * (3.0 * t2 * t) + p2 * (3.0 * t1 * t * t) + p3 * (t * t * t))
        .to_point()
        .cast_unit()
}

pub fn evalute_cubic_bezier_derivative<U>(
    p0: euclid::Point2D<f32, U>,
    p1: euclid::Point2D<f32, U>,
    p2: euclid::Point2D<f32, U>,
    p3: euclid::Point2D<f32, U>,
    t: f32,
) -> euclid::Vector2D<f32, U> {
    let p0 = p0.to_untyped().to_vector();
    let p1 = p1.to_untyped().to_vector();
    let p2 = p2.to_untyped().to_vector();
    let p3 = p3.to_untyped().to_vector();
    let t1 = 1.0 - t;
    let t2 = t1 * t1;
    let t3 = t1 * t1 * t1;
    ((p1 - p0) * (3.0*t2) + (p2 - p1)*(6.0*t1*t) + (p3 - p2)*(3.0*t*t))
        .cast_unit()
}

#[repr(C, align(16))]
struct WeightLookup {
    data: [f32;4*(2187+1)],
}

#[repr(C, align(16))]
struct WeightLookupBinary {
    data: [f32;4*(4*1024+1)],
    data_derivative: [f32;4*(4*1024+1)],
}
lazy_static! {
    #[repr(C, align(16))]
    static ref WEIGHT_LOOKUP: WeightLookup = initialize_weights();
}

lazy_static! {
    #[repr(C, align(16))]
    static ref WEIGHT_LOOKUP_BINARY: WeightLookupBinary = initialize_weights_binary();
}

fn initialize_weights() -> WeightLookup {
    let mut res = [0.0;4*(2187+1)];
    let step = 1.0 / 2187.0;
    for i in 0..2187+1 {
        let t = (i as f32) * step;
        let t1 = 1.0 - t;
        let t2 = t1 * t1;
        let t3 = t1 * t1 * t1;

        let a = t3;
        let b = 3.0 * t2 * t;
        let c = 3.0 * t1 * t * t;
        let d = t * t * t;
        res[4*i] = a;
        res[4*i+1] = b;
        res[4*i+2] = c;
        res[4*i+3] = d;
    }
    WeightLookup { data: res }
}

fn initialize_weights_binary() -> WeightLookupBinary {
    let mut res = [0.0;4*(4*1024+1)];
    let mut res2 = [0.0;4*(4*1024+1)];
    let step = 1.0 / (4.0*1024.0);
    for i in 0..(4*1024)+1 {
        let t = (i as f32) * step;
        let t1 = 1.0 - t;
        let t2 = t1 * t1;
        let t3 = t1 * t1 * t1;

        let a = t3;
        let b = 3.0 * t2 * t;
        let c = 3.0 * t1 * t * t;
        let d = t * t * t;
        res[4*i] = a;
        res[4*i+1] = b;
        res[4*i+2] = c;
        res[4*i+3] = d;

        // ((p1 - p0) * (3.0*t2) + (p2 - p1)*(6.0*t1*t) + (p3 - p2)*(3.0*t*t))
        let ap = -3.0*t2;
        let bp = 3.0*t2 - 6.0*t1*t;
        let cp = 6.0*t1*t - 3.0*t*t;
        let dp = 3.0*t*t;
        res2[4*i] = ap;
        res2[4*i+1] = bp;
        res2[4*i+2] = cp;
        res2[4*i+3] = dp;
    }
    WeightLookupBinary { data: res, data_derivative: res2 }
}

fn evaluate_cubic_bezier(
    p0: Vector,
    p1: Vector,
    p2: Vector,
    p3: Vector,
    weights: &[f32;4],
) -> Vector {
    p0 * weights[0] + p1 * weights[1] + p2 * weights[2] + p3 * weights[3]
}

fn binary_search_distance(
    p0: Vector,
    p1: Vector,
    p2: Vector,
    p3: Vector,
    p: Vector,
    mut start: u32,
    mut end: u32
) -> (f32, f32) {
    let weights = &WEIGHT_LOOKUP_BINARY.data;
    let weights2 = &WEIGHT_LOOKUP_BINARY.data_derivative;
    for _ in 0..10 {
        let mid = (start + end) / 2;
        let dp = evaluate_cubic_bezier(p0, p1, p2, p3, weights2[4*mid as usize..4*mid as usize+4].try_into().unwrap());
        let curve_p = evaluate_cubic_bezier(p0, p1, p2, p3, weights[4*mid as usize..4*mid as usize+4].try_into().unwrap());
        if (p - curve_p).dot(dp) > 0.0 {
            start = mid;
        } else {
            end = mid;
        }
    }
    let final_t = ((start + end) as f32) * 0.5 * (1.0 / 4096.0);
    let final_dist = (evalute_cubic_bezier(p0.to_point(), p1.to_point(), p2.to_point(), p3.to_point(), final_t) - p.to_point()).square_length();
    (final_dist, final_t)
}

fn ternary_search_distance(
    p0: Vector,
    p1: Vector,
    p2: Vector,
    p3: Vector,
    p: Vector,
) -> (f32, f32) {
    let mut a: u32 = 0;
    let mut b: u32 = 2187;
    let weights = &WEIGHT_LOOKUP.data;
    for _ in 0..7 {
        // 3.powi(7) == 2187
        let mid1 = (2*a + b)/3;
        let mid2 = (a + 2*b)/3;
        let d1 = (evaluate_cubic_bezier(p0, p1, p2, p3, weights[4*mid1 as usize..4*mid1 as usize+4].try_into().unwrap()) - p).square_length();
        let d2 = (evaluate_cubic_bezier(p0, p1, p2, p3, weights[4*mid2 as usize..4*mid2 as usize+4].try_into().unwrap()) - p).square_length();
        if d1 < d2 {
            b = mid2;
        } else {
            a = mid1;
        }
    }

    let pa = evaluate_cubic_bezier(p0, p1, p2, p3, weights[4*a as usize..4*a as usize+4].try_into().unwrap());
    let pb = evaluate_cubic_bezier(p0, p1, p2, p3, weights[4*b as usize..4*b as usize+4].try_into().unwrap());
    let t = closest_point_on_segment_t(pa, pb, p);

    let final_t = ((a as f32) + (b - a) as f32 * t) / 2187.0;
    let final_dist = (pa + (pb - pa)*t - p).square_length();
    (final_dist, final_t)
}

fn reduce_xs(v: f32x8) -> f32x2 {
    // 0 1 2 3 4 5 6 7
    // 0 1 2 3 + 4 5 6 7
    let v1: f32x4 = shuffle!(v, [0, 1, 2, 3]);
    let v2: f32x4 = shuffle!(v, [4, 5, 6, 7]);
    let v : f32x4 = v1 + v2;
    // 0 1 2 3
    // 0 1 + 2 3
    let v1: f32x2 = shuffle!(v, [0, 1]);
    let v2: f32x2 = shuffle!(v, [2, 3]);
    let v = v1 + v2;
    v
}

fn evaluate_cubic_bezier_simd(
    points: f32x8,
    weights: f32x4,
) -> Vector {
    let weights = shuffle!(weights, [0, 0, 1, 1, 2, 2, 3, 3]);
    let v = points * weights;
    unsafe {
        let k = reduce_xs(v);
        return vector(k.extract_unchecked(0),k.extract_unchecked(1));
    }
}

pub fn ternary_search_distance_simd(
    p0: Vector,
    p1: Vector,
    p2: Vector,
    p3: Vector,
    p: Vector,
) -> (f32, f32) {
    let mut a : u32 = 0;
    let mut b : u32 = 2187;
    let weights = &WEIGHT_LOOKUP.data;
    let points = f32x8::new(p0.x, p0.y, p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);

    for _ in 0..7 {
        // 3.powi(7) == 2187
        let mid1 = (2*a + b)/3;
        let mid2 = (a + 2*b)/3;
        // SAFETY: the WeightLookup struct has an alignment of 16, which is the same as f32x4
        let weights1 = unsafe { f32x4::from_slice_aligned_unchecked(&weights[4*mid1 as usize..4*mid1 as usize+4]) };
        let d1 = (evaluate_cubic_bezier_simd(points, weights1) - p).square_length();
        let weights2 = unsafe { f32x4::from_slice_aligned_unchecked(&weights[4*mid2 as usize..4*mid2 as usize+4]) };
        let d2 = (evaluate_cubic_bezier_simd(points, weights2) - p).square_length();
        if d1 < d2 {
            b = mid2;
        } else {
            a = mid1;
        }
    }

    let weights_a = unsafe { f32x4::from_slice_aligned_unchecked(&weights[4*a as usize..4*a as usize+4]) };
    let weights_b = unsafe { f32x4::from_slice_aligned_unchecked(&weights[4*b as usize..4*b as usize+4]) };
    let pa = evaluate_cubic_bezier_simd(points, weights_a);
    let pb = evaluate_cubic_bezier_simd(points, weights_b);
    // let dist = (evaluate_cubic_bezier_simd(points, a) - p).square_length();
    let t = closest_point_on_segment_t(pa, pb, p);

    let final_t = ((a as f32) + (b - a) as f32 * t) / 2187.0;
    let final_dist = (pa + (pb - pa)*t - p).square_length();
    (final_dist, final_t)
}

pub fn split(
    p0: Vector,
    p1: Vector,
    p2: Vector,
    p3: Vector,
    t: f32) -> ((Vector, Vector, Vector, Vector), (Vector, Vector, Vector, Vector)) {
        let ctrl1a = p0 + (p1 - p0) * t;
        let ctrl2a = p1 + (p2 - p1) * t;
        let ctrl1aa = ctrl1a + (ctrl2a - ctrl1a) * t;
        let ctrl3a = p2 + (p3 - p2) * t;
        let ctrl2aa = ctrl2a + (ctrl3a - ctrl2a) * t;
        let ctrl1aaa = ctrl1aa + (ctrl2aa - ctrl1aa) * t;
        let to = p3;

        (
            (p0, ctrl1a, ctrl1aa, ctrl1aaa),
            (ctrl1aaa, ctrl2aa, ctrl3a, to),
        )
}

fn remap01(v: f32, to_range: (f32,f32)) -> f32 {
    to_range.0 + v * (to_range.1-to_range.0)
}

fn closest_point_on_segment_t(
    start: Vector,
    end: Vector,
    point: Vector,
) -> f32 {
    let dir = end - start;
    let sqr_length = dir.square_length();

    if sqr_length <= 0.0000001 {
        0.0
    } else {
        let factor = (point - start).dot(dir) / sqr_length;
        factor.max(0.0).min(1.0)
    }
}

pub fn sqr_distance_bezier_point_lower_bound<U>(
    p0: euclid::Point2D<f32, U>,
    p1: euclid::Point2D<f32, U>,
    p2: euclid::Point2D<f32, U>,
    p3: euclid::Point2D<f32, U>,
    p: euclid::Point2D<f32, U>,
) -> f32 {
    // Calculate bounding box of the control points
    // The curve will always lie inside the bounding box
    let xmin = p0.x.min(p1.x).min(p2.x).min(p3.x);
    let ymin = p0.y.min(p1.y).min(p2.y).min(p3.y);
    let xmax = p0.x.max(p1.x).max(p2.x).max(p3.x);
    let ymax = p0.y.max(p1.y).max(p2.y).max(p3.y);
    // Clamp #p to the bounding box
    let p2 = point(p.x.min(xmax).max(xmin), p.y.min(ymax).max(ymin));
    // Distance between the clamped point and the original point.
    // In order words the shortest distance between #p and the bounding box.
    (p - p2).square_length()
}

pub fn sqr_distance_bezier_point<U>(
    p0: euclid::Point2D<f32, U>,
    p1: euclid::Point2D<f32, U>,
    p2: euclid::Point2D<f32, U>,
    p3: euclid::Point2D<f32, U>,
    p: euclid::Point2D<f32, U>,
) -> (f32, euclid::Point2D<f32, U>) {
    let p0 = p0.to_untyped().to_vector();
    let p1 = p1.to_untyped().to_vector();
    let p2 = p2.to_untyped().to_vector();
    let p3 = p3.to_untyped().to_vector();
    let p = p.to_untyped().to_vector();

    // Split into 4
    let (a1, a2) = split(p0, p1, p2, p3, 0.5);
    let (c1, c2) = split(a1.0, a1.1, a1.2, a1.3, 0.5);
    let (c3, c4) = split(a2.0, a2.1, a2.2, a2.3, 0.5);

    let (d1, t1) = ternary_search_distance(c1.0, c1.1, c1.2, c1.3, p);
    let (d2, t2) = ternary_search_distance(c2.0, c2.1, c2.2, c2.3, p);
    let (d3, t3) = ternary_search_distance(c3.0, c3.1, c3.2, c3.3, p);
    let (d4, t4) = ternary_search_distance(c4.0, c4.1, c4.2, c4.3, p);

    let mut d = d1;
    let mut t = remap01(t1, (0.0, 0.25));
    if d2 < d {
        d = d2;
        t = remap01(t2, (0.25, 0.5));
    }
    if d3 < d {
        d = d3;
        t = remap01(t3, (0.5, 0.75));
    }
    if d4 < d {
        d = d4;
        t = remap01(t4, (0.75, 1.0));
    }

    (d, evalute_cubic_bezier::<U>(p0.to_point().cast_unit(), p1.to_point().cast_unit(), p2.to_point().cast_unit(), p3.to_point().cast_unit(), t))
}

pub fn sqr_distance_bezier_point_binary<U>(
    p0: euclid::Point2D<f32, U>,
    p1: euclid::Point2D<f32, U>,
    p2: euclid::Point2D<f32, U>,
    p3: euclid::Point2D<f32, U>,
    p: euclid::Point2D<f32, U>,
) -> (f32, euclid::Point2D<f32, U>) {
    let p0 = p0.to_untyped().to_vector();
    let p1 = p1.to_untyped().to_vector();
    let p2 = p2.to_untyped().to_vector();
    let p3 = p3.to_untyped().to_vector();
    let p = p.to_untyped().to_vector();

    const OVERLAP: u32 = 128;
    let (d1, t1) = binary_search_distance(p0, p1, p2, p3, p, 0, 1024 + OVERLAP);
    let (d2, t2) = binary_search_distance(p0, p1, p2, p3, p, 1024 - OVERLAP, 2048 + OVERLAP);
    let (d3, t3) = binary_search_distance(p0, p1, p2, p3, p, 2048 - OVERLAP, 3072 + OVERLAP);
    let (d4, t4) = binary_search_distance(p0, p1, p2, p3, p, 3072 - OVERLAP, 4096);

    let mut d = d1;
    let mut t = t1;
    if d2 < d {
        d = d2;
        t = t2;
    }
    if d3 < d {
        d = d3;
        t = t3;
    }
    if d4 < d {
        d = d4;
        t = t4;
    }

    (d, evalute_cubic_bezier::<U>(p0.to_point().cast_unit(), p1.to_point().cast_unit(), p2.to_point().cast_unit(), p3.to_point().cast_unit(), t))
}

pub fn sqr_distance_bezier_point_simd<U>(
    p0: euclid::Point2D<f32, U>,
    p1: euclid::Point2D<f32, U>,
    p2: euclid::Point2D<f32, U>,
    p3: euclid::Point2D<f32, U>,
    p: euclid::Point2D<f32, U>,
) -> (f32, euclid::Point2D<f32, U>) {
    let p0 = p0.to_untyped().to_vector();
    let p1 = p1.to_untyped().to_vector();
    let p2 = p2.to_untyped().to_vector();
    let p3 = p3.to_untyped().to_vector();
    let p = p.to_untyped().to_vector();

    // Split into 4
    let (a1, a2) = split(p0, p1, p2, p3, 0.5);
    let (c1, c2) = split(a1.0, a1.1, a1.2, a1.3, 0.5);
    let (c3, c4) = split(a2.0, a2.1, a2.2, a2.3, 0.5);

    let (d1, t1) = ternary_search_distance_simd(c1.0, c1.1, c1.2, c1.3, p);
    let (d2, t2) = ternary_search_distance_simd(c2.0, c2.1, c2.2, c2.3, p);
    let (d3, t3) = ternary_search_distance_simd(c3.0, c3.1, c3.2, c3.3, p);
    let (d4, t4) = ternary_search_distance_simd(c4.0, c4.1, c4.2, c4.3, p);

    let mut d = d1;
    let mut t = remap01(t1, (0.0, 0.25));
    if d2 < d {
        d = d2;
        t = remap01(t2, (0.25, 0.5));
    }
    if d3 < d {
        d = d3;
        t = remap01(t3, (0.5, 0.75));
    }
    if d4 < d {
        d = d4;
        t = remap01(t4, (0.75, 1.0));
    }

    (d, evalute_cubic_bezier::<U>(p0.to_point().cast_unit(), p1.to_point().cast_unit(), p2.to_point().cast_unit(), p3.to_point().cast_unit(), t))
}

pub fn sqr_distance_bezier_point2<U>(
    p0: euclid::Point2D<f32, U>,
    p1: euclid::Point2D<f32, U>,
    p2: euclid::Point2D<f32, U>,
    p3: euclid::Point2D<f32, U>,
    p: euclid::Point2D<f32, U>,
) -> (f32, euclid::Point2D<f32, U>) {
    let mut closest = euclid::Point2D::new(0.0, 0.0);
    let mut closest_dist = std::f32::INFINITY;
    for i in 0..100 {
        let t = i as f32 / 100.0;
        let bezier_point = evalute_cubic_bezier(p0, p1, p2, p3, t);
        let dist = (bezier_point - p).square_length();
        if dist < closest_dist {
            closest_dist = dist;
            closest = bezier_point;
        }
    }
    (closest_dist, closest)
}

pub fn closest_point_on_segment<U>(
    start: euclid::Point2D<f32, U>,
    end: euclid::Point2D<f32, U>,
    point: euclid::Point2D<f32, U>,
) -> euclid::Point2D<f32, U> {
    let dir = end - start;
    let sqr_length = dir.square_length();

    if sqr_length <= 0.000001 {
        start
    } else {
        let factor = (point - start).dot(dir) / sqr_length;
        start + dir * factor.max(0.0).min(1.0)
    }
}

pub fn square_distance_segment_point<U>(
    a: euclid::Point2D<f32, U>,
    b: euclid::Point2D<f32, U>,
    p: euclid::Point2D<f32, U>,
) -> f32 {
    (closest_point_on_segment(a, b, p) - p).square_length()
}

impl VectorField {
    pub fn sample(&self, point: CanvasPoint) -> Option<CanvasVector> {
        self.sample_time(0.0, point)
    }

    pub fn is_solenoid_field() -> bool {
        // https://en.wikipedia.org/wiki/Solenoidal_vector_field
        true
    }

    fn sample_time(&self, _time: f32, point: CanvasPoint) -> Option<CanvasVector> {
        let mut result = CanvasVector::new(0.0, 0.0);
        for primitive in &self.primitives {
            match primitive {
                &VectorFieldPrimitive::Curl {
                    center,
                    strength,
                    radius,
                } => {
                    let dir = point - center;
                    let dist = dir.length();
                    if dist > 0.0001 && dist < radius {
                        // This has a divergence of zero
                        // Producing a solenoid field: https://en.wikipedia.org/wiki/Solenoidal_vector_field
                        let force = CanvasVector::new(-dir.y, dir.x) * (1.0 / dist * (1.0 - dist / radius) * strength);
                        result += force;
                    }
                }
                &VectorFieldPrimitive::Linear { direction, strength } => {
                    // This has a divergence of zero
                    // Producing a solenoid field: https://en.wikipedia.org/wiki/Solenoidal_vector_field
                    result += direction.normalize() * strength
                }
            }
        }
        Some(result)
    }

    fn normalize_safe(v: CanvasVector) -> CanvasVector {
        let len = v.length();
        if len < 0.0001 {
            CanvasVector::new(0.0, 0.0)
        } else {
            v / len
        }
    }

    pub fn trace(&self, mut point: CanvasPoint) -> (Vec<CanvasPoint>, bool) {
        // Use 4th order runge-kutta
        let dt = 5.0;
        let mut result = vec![point];
        for i in 0..100 {
            let t = i as f32 * dt;
            let k1 = Self::normalize_safe(self.sample_time(t, point).unwrap()) * dt;
            let k2 = Self::normalize_safe(self.sample_time(t + dt * 0.5, point + k1 * 0.5).unwrap()) * dt;
            let k3 = Self::normalize_safe(self.sample_time(t + dt * 0.5, point + k2 * 0.5).unwrap()) * dt;
            let k4 = Self::normalize_safe(self.sample_time(t + dt, point + k3).unwrap()) * dt;

            let prev = point;
            point += k1 * (1.0 / 6.0) + k2 * (2.0 / 6.0) + k3 * (2.0 / 6.0) + k4 * (1.0 / 6.0);

            if result.len() > 2 {
                let &first = result.first().unwrap();
                let dist = square_distance_segment_point(prev, point, first);
                if dist <= dt * dt * 1.01 * 1.01 {
                    // Looped back on itself
                    return (result, true);
                }
            }
            result.push(point);
        }
        (result, false)
    }

    pub fn trace_with_clearance(&self, mut point: CanvasPoint, clearance: f32) -> Vec<CanvasPoint> {
        // Use 4th order runge-kutta
        let dt = 5.0;
        let mut result = vec![point];
        for i in 0..100 {
            let t = i as f32 * dt;
            let k1 = Self::normalize_safe(self.sample_time(t, point).unwrap()) * dt;
            let k2 = Self::normalize_safe(self.sample_time(t + dt * 0.5, point + k1 * 0.5).unwrap()) * dt;
            let k3 = Self::normalize_safe(self.sample_time(t + dt * 0.5, point + k2 * 0.5).unwrap()) * dt;
            let k4 = Self::normalize_safe(self.sample_time(t + dt, point + k3).unwrap()) * dt;

            point += k1 * (1.0 / 6.0) + k2 * (2.0 / 6.0) + k3 * (2.0 / 6.0) + k4 * (1.0 / 6.0);
            result.push(point);
        }
        result
    }
}

struct PathClearanceGrid<'a, U> {
    clearance: f32,
    bounds: euclid::Rect<f32, U>,
    grid: Vec<Option<ImmutablePathPoint<'a>>>,
}

impl<'a, U> PathClearanceGrid<'a, U> {
    fn new(clearance: f32, bounds: euclid::Rect<f32, U>) -> PathClearanceGrid<'a, U> {
        let cell_size = clearance * 2.0;
        PathClearanceGrid {
            clearance,
            bounds,
            grid: vec![
                None;
                (bounds.width() / cell_size).ceil() as usize * (bounds.height() / cell_size).ceil() as usize
            ],
        }
    }

    fn add(point: ImmutablePathPoint<'a>) {
        // point.prev()
    }
}

pub fn poisson_disc_sampling<U>(
    bounds: euclid::Rect<f32, U>,
    radius: f32,
    rng: &mut impl Rng,
) -> Vec<euclid::Point2D<f32, U>> {
    let radius_squared = radius * radius;
    let cell_size = radius * (0.5f32).sqrt();
    let grid_width = (bounds.width() / cell_size).ceil() as usize;
    let grid_height = (bounds.height() / cell_size).ceil() as usize;

    const K: u32 = 4;
    const EPSILON: f32 = 0.001;

    let mut grid: Vec<i32> = vec![0; grid_width * grid_height];
    let mut queue = Vec::new();
    let mut result: Vec<euclid::Point2D<f32, U>> = Vec::new();

    let to_grid_coord = |p| {
        let relative: euclid::Vector2D<f32, U> = (p - bounds.min()) * (1.0 / cell_size);
        let x0 = relative.x as i32;
        let y0 = relative.y as i32;
        (x0, y0)
    };

    let initial_point = point(
        rng.gen_range(bounds.min_x(), bounds.max_x()),
        rng.gen_range(bounds.min_y(), bounds.max_y()),
    );
    queue.push(initial_point);
    result.push(initial_point);
    let initial_grid_coord = to_grid_coord(initial_point);
    grid[(initial_grid_coord.1 * grid_width as i32 + initial_grid_coord.0) as usize] = 0;

    'outer: while !queue.is_empty() {
        let i = rng.gen_range(0, queue.len());
        let p = queue[i];

        let starting_fraction = rng.gen_range(0.0, 1.0);
        'inner: for j in 0..K {
            let angle = 2.0 * std::f32::consts::PI * (starting_fraction + j as f32 / K as f32);
            let r = radius + EPSILON;
            let np: euclid::Point2D<f32, U> = p + vector(r * angle.cos(), r * angle.sin());

            if !bounds.contains(np) {
                continue;
            }

            let (x0, y0) = to_grid_coord(np);
            for y in (y0 - 1).max(0)..(y0 + 2).min(grid_height as i32 - 1) {
                for x in (x0 - 1).max(0)..(x0 + 2).min(grid_width as i32 - 1) {
                    let cell_index = y * grid_width as i32 + x;
                    if (result[grid[cell_index as usize] as usize] - np).square_length() < radius_squared {
                        continue 'inner;
                    }
                }
            }

            let cell_index = y0 as usize * grid_width + x0 as usize;
            grid[cell_index] = result.len() as i32;
            queue.push(np);
            result.push(np);
            continue 'outer;
        }

        queue.swap_remove(i);
    }
    result
}
