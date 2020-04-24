use crate::geometry_utilities::types::*;
use lyon::math::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hasher};

#[derive(Clone)]
pub struct SubPathData {
    pub range: std::ops::Range<usize>,
    closed: bool,
}

pub struct PathData {
    /// Points
    /// [point1, ctrl1_0, ctrl1_1, ..., pointN, ctrlN_0, ctrlN_1]
    pub points: Vec<CanvasPoint>,
    pub sub_paths: Vec<SubPathData>,
    pub path_index: u32,
    pub version: u32,
    last_hash: std::cell::Cell<(u32, u64)>,
    in_path: bool,
}

#[derive(Clone)]
pub struct ImmutablePathPoint<'a> {
    pub(crate) index: usize,
    pub(crate) sub_path: usize,
    pub(crate) data: &'a PathData,
}

pub struct MutablePathPoint<'a> {
    pub(crate) index: usize,
    pub(crate) sub_path: usize,
    pub(crate) data: &'a mut PathData,
}

pub struct ImmutableControlPoint<'a> {
    pub(crate) index: usize,
    pub(crate) sub_path: usize,
    pub(crate) data: &'a PathData,
}

pub struct MutableControlPoint<'a> {
    pub(crate) index: usize,
    pub(crate) sub_path: usize,
    pub(crate) data: &'a mut PathData,
}

impl<'a> PathPoint<'a> for MutablePathPoint<'a> {
    fn index(&self) -> usize {
        self.index
    }

    fn sub_path(&self) -> usize {
        self.sub_path
    }

    fn data(&'a self) -> &'a PathData {
        self.data
    }
}

impl<'a> ImmutableControlPoint<'a> {
    pub fn vertex(&'a self) -> ImmutablePathPoint<'a> {
        if self.index == self.data.sub_paths[self.sub_path].range.end - 1 {
            ImmutablePathPoint {
                data: self.data,
                sub_path: self.sub_path,
                index: self.data.sub_paths[self.sub_path].range.start,
            }
        } else {
            ImmutablePathPoint {
                data: self.data,
                sub_path: self.sub_path,
                // If index is 1 mod 3 then the vertex is one behind us, if we are at 2 mod 3 then the vertex is on the next index
                // 0 mod 3 is impossible since only vertices are at 0 mod 3
                index: ((self.index + 1) / 3) * 3,
            }
        }
    }

    pub fn position(&self) -> CanvasPoint {
        self.data.points[self.index] + self.vertex().position().to_vector()
    }
}

impl<'a> MutableControlPoint<'a> {
    // TODO: Code duplication
    pub fn vertex(&'a self) -> ImmutablePathPoint<'a> {
        if self.index == self.data.sub_paths[self.sub_path].range.end - 1 {
            ImmutablePathPoint {
                data: self.data,
                sub_path: self.sub_path,
                index: self.data.sub_paths[self.sub_path].range.start,
            }
        } else {
            ImmutablePathPoint {
                data: self.data,
                sub_path: self.sub_path,
                // If index is 1 mod 3 then the vertex is one behind us, if we are at 2 mod 3 then the vertex is on the next index
                // 0 mod 3 is impossible since only vertices are at 0 mod 3
                index: ((self.index + 1) / 3) * 3,
            }
        }
    }

    pub fn position(&self) -> CanvasPoint {
        self.data.points[self.index] + self.vertex().position().to_vector()
    }

    pub fn set_position(&mut self, value: CanvasPoint) {
        self.data.points[self.index] = value - self.vertex().position().to_vector()
    }
}

pub trait PathPoint<'a> {
    fn index(&self) -> usize;
    fn sub_path(&self) -> usize;
    fn data(&'a self) -> &'a PathData;

    fn position(&'a self) -> CanvasPoint {
        self.data().points[self.index()]
    }

    fn control_after(&'a self) -> CanvasPoint {
        self.position() + self.data().points[control_after_index(self.index())].to_vector()
    }

    fn control_before(&'a self) -> CanvasPoint {
        self.position()
            + self.data().points[control_before_index(self.data(), self.sub_path(), self.index())].to_vector()
    }

    fn point_type(&'a self) -> PointType {
        self.data().point_type(self.index() as i32)
    }
}

impl<'a> PathPoint<'a> for ImmutablePathPoint<'a> {
    fn index(&self) -> usize {
        self.index
    }

    fn sub_path(&self) -> usize {
        self.sub_path
    }

    fn data(&'a self) -> &'a PathData {
        self.data
    }
}

pub(crate) fn prev_index(data: &PathData, sub_path: usize, index: usize) -> Option<usize> {
    if index == data.sub_paths[sub_path].range.start {
        if !data.sub_paths[sub_path].closed {
            None
        } else {
            Some(data.sub_paths[sub_path].range.end - 3)
        }
    } else {
        Some(index - 3)
    }
}

pub(crate) fn next_index(data: &PathData, sub_path: usize, index: usize) -> Option<usize> {
    if index + 3 >= data.sub_paths[sub_path].range.end {
        if !data.sub_paths[sub_path].closed {
            None
        } else {
            Some(data.sub_paths[sub_path].range.start)
        }
    } else {
        Some(index + 3)
    }
}

pub(crate) fn control_before_index(data: &PathData, sub_path: usize, index: usize) -> usize {
    if index == data.sub_paths[sub_path].range.start {
        data.sub_paths[sub_path].range.end - 1
    } else {
        index - 1
    }
}

pub(crate) fn control_after_index(index: usize) -> usize {
    index + 1
}

impl<'b, 'a: 'b> ImmutablePathPoint<'a> {
    pub fn prev(&'a self) -> Option<ImmutablePathPoint<'b>> {
        debug_assert!(self.data().point_type(self.index() as i32) == PointType::Point);
        prev_index(self.data, self.sub_path, self.index).map(|new_index| ImmutablePathPoint {
            index: new_index,
            sub_path: self.sub_path,
            data: self.data,
        })
    }

    pub fn next(&'a self) -> Option<ImmutablePathPoint<'b>> {
        debug_assert!(self.data().point_type(self.index() as i32) == PointType::Point);
        next_index(self.data, self.sub_path, self.index).map(|new_index| ImmutablePathPoint {
            index: new_index,
            sub_path: self.sub_path,
            data: self.data,
        })
    }
}

impl<'b, 'a: 'b> MutablePathPoint<'a> {
    pub fn set_position(&'a mut self, value: CanvasPoint) {
        self.data.points[self.index] = value;
    }

    pub fn set_control_after(&mut self, value: CanvasPoint) {
        self.data.points[control_after_index(self.index)] = (value - self.position()).to_point();
    }

    pub fn set_control_before(&mut self, value: CanvasPoint) {
        let index = control_before_index(self.data, self.sub_path, self.index);
        self.data.points[index] = (value - self.position()).to_point();
    }

    pub fn prev(&'a self) -> Option<ImmutablePathPoint<'b>> {
        debug_assert!(self.data().point_type(self.index() as i32) == PointType::Point);
        prev_index(self.data, self.sub_path, self.index).map(|new_index| ImmutablePathPoint {
            index: new_index,
            sub_path: self.sub_path,
            data: self.data,
        })
    }

    pub fn next(&'a self) -> Option<ImmutablePathPoint<'b>> {
        debug_assert!(self.data().point_type(self.index() as i32) == PointType::Point);
        next_index(self.data, self.sub_path, self.index).map(|new_index| ImmutablePathPoint {
            index: new_index,
            sub_path: self.sub_path,
            data: self.data,
        })
    }

    pub fn prev_mut(self) -> Option<MutablePathPoint<'b>> {
        debug_assert!(self.data().point_type(self.index() as i32) == PointType::Point);
        if let Some(new_index) = prev_index(self.data, self.sub_path, self.index) {
            Some(MutablePathPoint {
                index: new_index,
                sub_path: self.sub_path,
                data: self.data,
            })
        } else {
            None
        }
    }
}

pub enum ControlPointDirection {
    Before,
    After,
}

pub struct SubPath<'a> {
    data: &'a PathData,
    index: usize,
}

impl<'a, 'b: 'a> SubPath<'b> {
    pub fn first(&'a self) -> ImmutablePathPoint<'b> {
        ImmutablePathPoint {
            index: self.data.sub_paths[self.index].range.start,
            data: self.data,
            sub_path: self.index,
        }
    }

    pub fn closed(&self) -> bool {
        self.data.sub_paths[self.index].closed
    }

    fn iter_point_indices(&'a self) -> impl Iterator<Item = usize> {
        self.data.sub_paths[self.index].range.clone().step_by(3)
    }

    pub fn iter_points(&'a self) -> impl Iterator<Item = ImmutablePathPoint<'b>> {
        let data = self.data;
        let index = self.index;
        self.iter_point_indices().map(move |i| ImmutablePathPoint {
            index: i,
            data: data,
            sub_path: index,
        })
    }

    pub fn iter_beziers(&'a self) -> impl Iterator<Item = ImmutablePathPoint<'b>> {
        let data = self.data;
        let index = self.index;
        let mut range = self.data.sub_paths[self.index].range.clone();
        if !self.closed() {
            range.end -= 3;
        }

        range.step_by(3).map(move |i| ImmutablePathPoint {
            index: i,
            data: data,
            sub_path: index,
        })
    }
}
/*impl<'a> Iterator for PathIterator<'a> {
    type item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        self.index += 1;
        if self.index < self.data.sub_paths[self.sub_path].end {
            Some(PathPoint {
                index: self.index,
                data: self.data,
                sub_path: 0,
            })
        } else {
            None
        }
    }
}*/

#[derive(PartialEq)]
pub enum PointType {
    Point,
    ControlPoint,
}

#[derive(Copy, Clone)]
pub struct BorderRadii {
    pub top_left: f32,
    pub top_right: f32,
    pub bottom_left: f32,
    pub bottom_right: f32,
}

impl BorderRadii {
    pub fn new_uniform(radius: f32) -> BorderRadii {
        BorderRadii {
            top_left: radius,
            top_right: radius,
            bottom_left: radius,
            bottom_right: radius,
        }
    }
}

impl PathData {
    pub fn new() -> PathData {
        PathData {
            points: vec![],
            sub_paths: vec![],
            in_path: false,
            path_index: 0,
            version: 0,
            last_hash: std::cell::Cell::new((0, 0)),
        }
    }

    pub fn copy_from(&mut self, other: &PathData) {
        self.points = other.points.clone();
        self.sub_paths = other.sub_paths.clone();
        self.in_path = other.in_path;
        self.dirty();
    }

    pub fn hash(&self) -> u64 {
        if self.last_hash.get().0 == self.version {
            self.last_hash.get().1
        } else {
            let mut h = DefaultHasher::new();
            for p in &self.points {
                h.write_u32(p.x.to_bits());
                h.write_u32(p.y.to_bits());
            }
            let h = h.finish();
            self.last_hash.set((self.version, h));
            h
        }
    }

    pub fn dirty(&mut self) {
        self.version += 1;
    }

    pub fn remove<'a>(&'a mut self, _index: usize) {}

    pub fn iter_sub_paths<'a>(&'a self) -> impl Iterator<Item = SubPath<'a>> {
        (0..self.sub_paths.len()).map(move |i| SubPath { data: self, index: i })
    }

    pub fn iter_points<'a>(&'a self) -> impl Iterator<Item = ImmutablePathPoint<'a>> {
        self.iter_sub_paths().flat_map(|sp| sp.iter_points())
    }

    fn find_sub_path(&self, index: i32) -> usize {
        for (i, sp) in self.sub_paths.iter().enumerate() {
            if sp.range.contains(&(index as usize)) {
                return i;
            }
        }
        panic!("No sub path contains index {:?}", index);
    }

    pub fn point_type(&self, index: i32) -> PointType {
        match index % 3 {
            0 => PointType::Point,
            _ => PointType::ControlPoint,
        }
    }

    pub fn point<'a>(&'a self, index: i32) -> ImmutablePathPoint<'a> {
        ImmutablePathPoint {
            data: self,
            sub_path: self.find_sub_path(index),
            index: index as usize,
        }
    }

    pub fn point_mut<'a>(&'a mut self, index: i32) -> MutablePathPoint<'a> {
        let sub_path = self.find_sub_path(index);
        MutablePathPoint {
            data: self,
            sub_path,
            index: index as usize,
        }
    }

    pub fn control_point<'a>(&'a self, index: i32) -> ImmutableControlPoint<'a> {
        ImmutableControlPoint {
            data: self,
            sub_path: self.find_sub_path(index),
            index: index as usize,
        }
    }

    pub fn control_point_mut<'a>(&'a mut self, index: i32) -> MutableControlPoint<'a> {
        let sub_path = self.find_sub_path(index);
        MutableControlPoint {
            data: self,
            sub_path,
            index: index as usize,
        }
    }

    // pub fn set_point(&mut self, index: i32, point: CanvasPoint) {
    //     match self.point_type(index) {
    //         PointType::Point => *self.point_mut(index).set_position(point),
    //         PointType::ControlPoint => *self.point_mut(index) = (point - self.point(index - (index%3)).position()).to_point(),
    //     }
    // }

    /*pub fn control_point(&self, index: i32, offset: ControlPointDirection) -> CanvasPoint {
        let mut ctrl_index = index * 3;
        let data_len = self.points.len() as i32;
        ctrl_index = match offset {
            ControlPointDirection::Before => ctrl_index - 1,
            ControlPointDirection::After => ctrl_index + 1,
        };
        assert!(ctrl_index >= -1 && ctrl_index <= data_len);
        ctrl_index = (ctrl_index + data_len) % data_len;
        return self.point(index % self.len()) + self.points[ctrl_index as usize].to_vector()
    }

    pub fn next_control(&self, index: i32) -> CanvasPoint {
        self.control_point(index, ControlPointDirection::After)
    }

    pub fn previous_control(&self, index: i32) -> CanvasPoint {
        self.control_point(index, ControlPointDirection::Before)
    }*/

    pub fn len(&self) -> i32 {
        self.points.len() as i32 / 3
    }

    pub fn clear(&mut self) {
        self.points.clear();
        self.sub_paths.clear();
        self.in_path = false;
    }

    pub fn current(&self) -> Option<ImmutablePathPoint> {
        if self.points.len() > 0 && self.in_path {
            Some(self.point(self.points.len() as i32 - 3))
        } else {
            None
        }
    }

    pub fn line_to(&mut self, pt: CanvasPoint) -> i32 {
        self.start_if_necessary();
        self.points.push(pt);
        self.points.push(point(0.0, 0.0));
        self.points.push(point(0.0, 0.0));
        self.extend_current();

        self.points.len() as i32 - 3
    }

    pub fn move_to(&mut self, pt: CanvasPoint) -> i32 {
        self.end();
        self.line_to(pt)
    }

    pub fn start_if_necessary(&mut self) {
        if !self.in_path {
            self.start();
        }
    }

    fn start(&mut self) {
        self.end();
        self.sub_paths.push(SubPathData {
            range: self.points.len()..self.points.len(),
            closed: false,
        });
        self.in_path = true;
    }

    pub fn close(&mut self) {
        assert!(self.in_path);
        self.end();
        self.sub_paths.last_mut().unwrap().closed = true;
    }

    pub fn end(&mut self) {
        self.extend_current();
        self.in_path = false;
    }

    pub fn extend_current(&mut self) {
        if self.in_path {
            self.sub_paths.last_mut().unwrap().range.end = self.points.len();
        }
    }

    pub fn add_circle(&mut self, center: CanvasPoint, radius: f32) {
        self.start();
        lyon::geom::Arc::circle(center.to_untyped(), radius).for_each_cubic_bezier(&mut |&bezier| {
            self.points.push(bezier.from.cast_unit::<CanvasSpace>());
            self.points
                .push((bezier.ctrl1 - bezier.from).to_point().cast_unit::<CanvasSpace>());
            self.points
                .push((bezier.ctrl2 - bezier.to).to_point().cast_unit::<CanvasSpace>());
        });
        self.close();
    }

    pub fn add_rounded_rect(&mut self, rect: &Rect, mut radii: BorderRadii) {
        fn clamp(r1: &mut f32, r2: &mut f32, max: f32) {
            if *r1 + *r2 > max {
                let delta = (*r1 + *r2 - max) * 0.5;
                *r1 -= delta;
                *r2 -= delta;
            }
        }

        let min_wh = rect.width().min(rect.height());
        radii.bottom_left = radii.bottom_left.min(min_wh);
        radii.bottom_right = radii.bottom_right.min(min_wh);
        radii.top_left = radii.top_left.min(min_wh);
        radii.top_right = radii.top_right.min(min_wh);

        clamp(&mut radii.bottom_left, &mut radii.bottom_right, rect.width());
        clamp(&mut radii.top_left, &mut radii.top_right, rect.width());
        clamp(&mut radii.top_left, &mut radii.bottom_left, rect.height());
        clamp(&mut radii.top_right, &mut radii.bottom_right, rect.height());

        fn corner_arc(corner: Point, radius: f32, start_angle: Angle, offset: Vector) -> lyon::geom::Arc<f32> {
            lyon::geom::Arc {
                center: corner + offset * radius,
                radii: vector(radius, radius),
                start_angle: start_angle,
                sweep_angle: Angle::frac_pi_2(),
                x_rotation: Angle::zero(),
            }
        }

        let arcs = &[
            corner_arc(
                point(rect.min_x(), rect.min_y()),
                radii.top_left,
                Angle::degrees(180.0),
                vector(1.0, 1.0),
            ),
            corner_arc(
                point(rect.max_x(), rect.min_y()),
                radii.top_right,
                Angle::degrees(270.0),
                vector(-1.0, 1.0),
            ),
            corner_arc(
                point(rect.max_x(), rect.max_y()),
                radii.bottom_right,
                Angle::degrees(0.0),
                vector(-1.0, -1.0),
            ),
            corner_arc(
                point(rect.min_x(), rect.max_y()),
                radii.bottom_left,
                Angle::degrees(90.0),
                vector(1.0, -1.0),
            ),
        ];

        self.start();
        for arc in arcs {
            if arc.radii.x > 0.0 {
                arc.for_each_cubic_bezier(&mut |&bezier| {
                    self.points.push(bezier.from.cast_unit::<CanvasSpace>());
                    self.points
                        .push((bezier.ctrl1 - bezier.from).to_point().cast_unit::<CanvasSpace>());
                    self.points
                        .push((bezier.ctrl2 - bezier.to).to_point().cast_unit::<CanvasSpace>());
                });
            }
            self.line_to(arc.to().cast_unit());
        }
        self.close();
    }

    pub fn build(&self, builder: &mut lyon::path::Builder) {
        if self.len() == 0 {
            return;
        }

        for sub_path in self.iter_sub_paths() {
            builder.move_to(sub_path.first().position().to_untyped());
            for a in sub_path.iter_beziers() {
                let b = a.next().unwrap();
                builder.cubic_bezier_to(
                    a.control_after().to_untyped(),
                    b.control_before().to_untyped(),
                    b.position().to_untyped(),
                );
            }
            if sub_path.closed() {
                builder.close();
            }
        }
        /*let count = if self.closed { self.len() as i32 } else { self.len() as i32 - 1 };
        for i in 0..count {
            // TODO: Performance
            let next = (i+1) % (self.len() as i32);
            builder.cubic_bezier_to(
                self.next_control(i),
                self.previous_control(next),
                self.point(next)
            );
        }

        if self.closed {
            builder.close();
        }*/
    }
}

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_sanity() {
        let mut data = PathData::new();
        data.line_to(point(1.0, 2.0));
        assert_eq!(data.point(0).position(), point(1.0, 2.0));
    }

    #[test]
    fn test_iter() {
        let mut data = PathData::new();
        for _ in 0..1000 {
            data.add_circle(point(0.0, 0.0), 5.0);
        }
        assert_eq!(data.iter_sub_paths().count(), 1000);
        let mut k = point(0.0, 0.0);
        for _ in 0..10 {
            for p in data.iter_sub_paths() {
                assert_eq!(p.iter_points().count(), 4);
                for point in p.iter_points() {
                    k += point.position().to_vector();
                }
            }
        }
        dbg!(k);
    }
}
