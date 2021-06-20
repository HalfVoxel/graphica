use crate::{geometry_utilities::types::CanvasPoint, path::PathData};

pub fn catmull_rom_smooth(points: &[CanvasPoint]) -> PathData {
    let mut result = PathData::default();
    match points.len() {
        0 | 1 => {
            // emit nothing
        }
        2 => {
            result.move_to(points[0]);
            result.line_to(points[1]);
        }
        _ => {
            let catmull_rom_to =
                |path: &mut PathData, p0: CanvasPoint, p1: CanvasPoint, p2: CanvasPoint, p3: CanvasPoint| {
                    let p0 = p0.to_vector();
                    let p1 = p1.to_vector();
                    let p2 = p2.to_vector();
                    let p3 = p3.to_vector();
                    let _c0 = p1;
                    let c1 = (-p0 + p1 * 6.0 + p2 * 1.0) * (1.0 / 6.0);
                    let c2 = (p1 + p2 * 6.0 - p3) * (1.0 / 6.0);
                    let c3 = p2;
                    let vertex = path.line_to(c3.to_point());
                    path.point_mut(vertex)
                        .prev_mut()
                        .unwrap()
                        .set_control_after(c1.to_point());
                    path.point_mut(vertex).set_control_before(c2.to_point());
                };
            // count >= 3
            let count = points.len();
            result.move_to(points[0]);

            // Draw first curve, this is special because the first two control points are the same
            catmull_rom_to(&mut result, points[0], points[0], points[1], points[2]);
            for i in 0..count - 3 {
                catmull_rom_to(&mut result, points[i], points[i + 1], points[i + 2], points[i + 3]);
            }
            // Draw last curve
            catmull_rom_to(
                &mut result,
                points[count - 3],
                points[count - 2],
                points[count - 1],
                points[count - 1],
            );
            result.end();
        }
    }

    result
}
