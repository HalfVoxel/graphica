use euclid;
use lyon::math::*;
use crate::something::{ImmutablePathPoint, PathPoint};
use rand::Rng;
use types::*;

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
    Curl { center: CanvasPoint, strength: f32, radius: f32 },
    Linear { direction: CanvasVector, strength: f32 },
}

pub struct VectorField {
    pub primitives: Vec<VectorFieldPrimitive>,
}

pub fn evalute_cubic_bezier<U>(p0: euclid::Point2D<f32,U>, p1: euclid::Point2D<f32,U>, p2: euclid::Point2D<f32,U>, p3: euclid::Point2D<f32,U>, t: f32) -> euclid::Point2D<f32,U> {
    let p0 = p0.to_untyped().to_vector();
    let p1 = p1.to_untyped().to_vector();
    let p2 = p2.to_untyped().to_vector();
    let p3 = p3.to_untyped().to_vector();
    let t1 = 1.0-t;
    let t2 = t1*t1;
    let t3 = t1*t1*t1;
    (p0*t3 + p1*(3.0*t2*t) + p2*(3.0*t1*t*t) + p3*(t*t*t)).to_point().cast_unit()
}

#[inline(never)]
pub fn sqr_distance_bezier_point<U>(p0: euclid::Point2D<f32,U>, p1: euclid::Point2D<f32,U>, p2: euclid::Point2D<f32,U>, p3: euclid::Point2D<f32,U>, p: euclid::Point2D<f32,U>) -> (f32, euclid::Point2D<f32,U>) {
    let mut closest = euclid::Point2D::new(0.0,0.0);
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

pub fn closest_point_on_segment<U> (start: euclid::Point2D<f32,U>, end: euclid::Point2D<f32,U>, point: euclid::Point2D<f32,U>) -> euclid::Point2D<f32,U> {
    let dir = end - start;
    let sqr_length = dir.square_length();

    if sqr_length <= 0.000001 {
        start
    } else {
        let factor = (point - start).dot(dir) / sqr_length;
        start + dir*factor.max(0.0).min(1.0)
    }
}

pub fn square_distance_segment_point<U>(a: euclid::Point2D<f32, U>, b: euclid::Point2D<f32, U>, p: euclid::Point2D<f32, U>) -> f32 {
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

    fn sample_time(&self, time: f32, point: CanvasPoint) -> Option<CanvasVector> {
        let mut result = CanvasVector::new(0.0, 0.0);
        for primitive in &self.primitives {
            match primitive {
                &VectorFieldPrimitive::Curl { center, strength, radius } => {
                    let dir = point - center;
                    let dist = dir.length();
                    if dist > 0.0001 && dist < radius {
                        // This has a divergence of zero
                        // Producing a solenoid field: https://en.wikipedia.org/wiki/Solenoidal_vector_field
                        let force = CanvasVector::new(-dir.y, dir.x) * (1.0 / dist * (1.0 - dist/radius) * strength);
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
            let k2 = Self::normalize_safe(self.sample_time(t + dt*0.5, point + k1 * 0.5).unwrap()) * dt;
            let k3 = Self::normalize_safe(self.sample_time(t + dt*0.5, point + k2 * 0.5).unwrap()) * dt;
            let k4 = Self::normalize_safe(self.sample_time(t + dt, point + k3).unwrap()) * dt;
            
            let prev = point;
            point += k1*(1.0/6.0) + k2*(2.0/6.0) + k3*(2.0/6.0) + k4*(1.0/6.0);

            if result.len() > 2 {
                let &first = result.first().unwrap();
                let dist = square_distance_segment_point(prev, point, first);
                if dist <= dt * dt * 1.01*1.01 {
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
            let k2 = Self::normalize_safe(self.sample_time(t + dt*0.5, point + k1 * 0.5).unwrap()) * dt;
            let k3 = Self::normalize_safe(self.sample_time(t + dt*0.5, point + k2 * 0.5).unwrap()) * dt;
            let k4 = Self::normalize_safe(self.sample_time(t + dt, point + k3).unwrap()) * dt;

            point += k1*(1.0/6.0) + k2*(2.0/6.0) + k3*(2.0/6.0) + k4*(1.0/6.0);
            result.push(point);
        }
        result
    }
}

struct PathClearanceGrid<'a,U> {
    clearance: f32,
    bounds: euclid::Rect<f32,U>,
    grid: Vec<Option<ImmutablePathPoint<'a>>>,
}

impl<'a,U> PathClearanceGrid<'a,U> {
    fn new(clearance: f32, bounds: euclid::Rect<f32,U>) -> PathClearanceGrid<'a, U> {
        let cell_size = clearance*2.0;
        PathClearanceGrid {
            clearance,
            bounds,
            grid: vec![None;(bounds.width() / cell_size).ceil() as usize * (bounds.height() / cell_size).ceil() as usize]
        }
    }

    fn add(point: ImmutablePathPoint<'a>) {
        // point.prev()
    }
}

pub fn poisson_disc_sampling<U>(bounds: euclid::Rect<f32,U>, radius: f32, rng: &mut impl Rng) -> Vec<euclid::Point2D<f32,U>> {
    let radius_squared = radius*radius;
    let cell_size = radius * (0.5f32).sqrt();
    let grid_width = (bounds.width() / cell_size).ceil() as usize;
    let grid_height = (bounds.height() / cell_size).ceil() as usize;

    const k: u32 = 4;
    const epsilon: f32 = 0.001;

    let mut grid: Vec<i32> = vec![0; grid_width*grid_height];
    let mut queue = Vec::new();
    let mut result: Vec<euclid::Point2D<f32,U>> = Vec::new();

    let to_grid_coord = |p| {
        let relative: euclid::Vector2D<f32,U> = (p - bounds.min()) * (1.0 / cell_size);
        let x0 = relative.x as i32;
        let y0 = relative.y as i32;
        (x0, y0)
    };

    let initial_point = point(rng.gen_range(bounds.min_x(), bounds.max_x()), rng.gen_range(bounds.min_y(), bounds.max_y()));
    queue.push(initial_point);
    result.push(initial_point);
    let initial_grid_coord = to_grid_coord(initial_point);
    grid[(initial_grid_coord.1*grid_width as i32 + initial_grid_coord.0) as usize] = 0;

    'outer: while !queue.is_empty() {
        let i = rng.gen_range(0, queue.len());
        let p = queue[i];

        let starting_fraction = rng.gen_range(0.0, 1.0);
        'inner: for j in 0..k {
            let angle = 2.0 * std::f32::consts::PI * (starting_fraction + j as f32 / k as f32);
            let r = radius + epsilon;
            let np : euclid::Point2D<f32,U> = p + vector(r * angle.cos(), r * angle.sin());

            if !bounds.contains(np) {
                continue
            }

            let (x0, y0) = to_grid_coord(np);
            for y in (y0 - 1).max(0)..(y0 + 2).min(grid_height as i32 - 1) {
                for x in (x0 - 1).max(0)..(x0+2).min(grid_width as i32 - 1) {
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