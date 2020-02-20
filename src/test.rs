type Point = i32;

struct SubPathData {
    range: std::ops::Range<usize>,
    closed: bool,
}

struct PathData {
    /// Points
    /// [point1, ctrl1_0, ctrl1_1, ..., pointN, ctrlN_0, ctrlN_1]
    points: Vec<Point>,
    sub_paths: Vec<SubPathData>,
    in_path: bool,
}

struct PathPoint<'a> {
    index: usize,
    sub_path: usize,
    data: &'a PathData,
}

impl<'a> PathPoint<'a> {
    fn position(&self) -> Point {
        self.data.points[self.index]
    }

    fn control_after(&self) -> Point {
        self.position() + self.data.points[self.index + 1]
    }

    fn control_before(&self) -> Point {
        if self.index == self.data.sub_paths[self.sub_path].range.start {
            self.position() + self.data.points[self.data.sub_paths[self.sub_path].range.end - 1]
        } else {
            self.position() + self.data.points[self.index - 1]
        }
    }

    fn next (&'a self) -> PathPoint<'a> {
        PathPoint {
            index: if self.index + 3 >= self.data.sub_paths[self.sub_path].range.end { self.data.sub_paths[self.sub_path].range.start } else { self.index + 3 },
            sub_path: self.sub_path,
            data: self.data,
        }
    }
}

struct SubPath<'a> {
    data: &'a PathData,
    index: usize,
}

impl<'a> SubPath<'a> {
    fn iter_points(&'a self) -> impl Iterator<Item=PathPoint<'a>> {
        self.data.sub_paths[self.index].range.clone().step_by(3).map(move|i| PathPoint { index: i, data: self.data, sub_path: self.index })
    }
}

impl PathData {
    fn new() -> PathData {
        PathData {
            points: vec![],
            sub_paths: vec![],
            in_path: false,
        }
    }

    fn iter_sub_paths<'a> (&'a self) -> impl Iterator<Item=SubPath<'a>> {
        (0..self.sub_paths.len()).map(move|i| SubPath { data: self, index: i })
    }
}
