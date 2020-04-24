use crate::canvas::CanvasView;
use crate::geometry_utilities::types::*;
use crate::geometry_utilities::{
    poisson_disc_sampling, sqr_distance_bezier_point_binary, sqr_distance_bezier_point_lower_bound, VectorField,
    VectorFieldPrimitive,
};
use crate::input::*;
use crate::main::Document;
use crate::path::*;
use crate::path_collection::{
    ControlPointReference, MutableReferenceResolver, PathCollection, PathReference, ReferenceResolver,
    SelectionReference, VertexReference,
};
use crate::toolbar::ToolType;
use lyon::math::*;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashSet;

#[derive(Eq, Clone)]
pub struct Selection {
    /// Items in the selection.
    /// Should not contain any duplicates
    pub items: Vec<SelectionReference>,
}

impl PartialEq for Selection {
    fn eq(&self, other: &Self) -> bool {
        if self.items.len() != other.items.len() {
            return false;
        }

        let mut refs = HashSet::new();
        for item in &self.items {
            // Insert everything into the set
            // Also assert to make sure there are no duplicates in the selection
            // as this will break the algorithm
            assert!(refs.insert(item));
        }

        for item in &other.items {
            if !refs.contains(item) {
                return false;
            }
        }

        // The selections have the same length and self contains everything in other.
        // This means they are identical since selections do not contain duplicates
        true
    }
}

pub struct ClosestInSelection {
    distance: CanvasLength,
    point: CanvasPoint,
    closest: ClosestItemInSelection,
}
pub enum ClosestItemInSelection {
    Curve { start: VertexReference },
    ControlPoint(ControlPointReference),
}

impl Selection {
    // pub fn distance_to(&self, paths: &PathCollection, point: CanvasPoint) -> Option<ClosestInSelection> {
    //     let mut min_dist = std::f32::INFINITY;
    //     let mut closest_point = None;
    //     let mut closest_ref = None;
    //     let mut point_set = HashSet::new();
    //     for vertex in &self.items {
    //         if let SelectionReference::VertexReference(vertex) = vertex {
    //             point_set.insert(vertex);
    //         }
    //     }
    //     for vertex_ref in &self.items {
    //         match vertex_ref {
    //             SelectionReference::ControlPointReference(ctrl_ref) => {
    //                 let vertex = paths.resolve(ctrl_ref);
    //                 let dist = (vertex.position() - point).square_length();
    //                 if dist < min_dist {
    //                     min_dist = dist;
    //                     closest_point = Some(vertex.position());
    //                     closest_ref = Some(vertex_ref.clone());
    //                 }
    //             }
    //             SelectionReference::VertexReference(vertex_ref2) => {
    //                 let vertex = paths.resolve(vertex_ref2);
    //                 if vertex_ref2
    //                     .next(paths)
    //                     .filter(|next| point_set.contains(&vertex_ref2))
    //                     .is_some()
    //                 {
    //                     let (dist, closest_on_curve) = sqr_distance_bezier_point(
    //                         vertex.position(),
    //                         vertex.control_after(),
    //                         vertex.next().unwrap().control_before(),
    //                         vertex.next().unwrap().position(),
    //                         point,
    //                     );
    //                     if dist < min_dist {
    //                         min_dist = dist;
    //                         closest_point = Some(closest_on_curve);
    //                         closest_ref = Some(vertex_ref.clone());
    //                     }
    //                 } else {
    //                     let dist = (vertex.position() - point).square_length();
    //                     if dist < min_dist {
    //                         min_dist = dist;
    //                         closest_point = Some(vertex.position());
    //                         closest_ref = Some(vertex_ref.clone());
    //                     }
    //                 }
    //             }
    //         }
    //     }
    //     if let Some(p) = closest_ref {
    //         match p {
    //             SelectionReference::VertexReference(vertex_ref2) => Some(ClosestInSelection {
    //                 distance: CanvasLength::new(min_dist.sqrt()),
    //                 point: closest_point.unwrap(),
    //                 closest: ClosestItemInSelection::Curve { start: vertex_ref2 },
    //             }),
    //             SelectionReference::ControlPointReference(ctrl_ref) => Some(ClosestInSelection {
    //                 distance: CanvasLength::new(min_dist.sqrt()),
    //                 point: closest_point.unwrap(),
    //                 closest: ClosestItemInSelection::ControlPoint(ctrl_ref),
    //             }),
    //         }
    //     } else {
    //         None
    //     }
    // }

    pub fn distance_to_curve(
        &self,
        paths: &PathCollection,
        point: CanvasPoint,
    ) -> Option<(VertexReference, CanvasPoint, CanvasLength)> {
        let mut min_dist = std::f32::INFINITY;
        let mut closest_point = None;
        let mut closest_ref = None;
        let mut point_set = HashSet::new();
        for vertex in &self.items {
            if let SelectionReference::VertexReference(vertex) = vertex {
                point_set.insert(vertex);
            }
        }
        for vertex_ref in &self.items {
            match vertex_ref {
                SelectionReference::VertexReference(vertex_ref2) => {
                    let vertex = paths.resolve(vertex_ref2);
                    if vertex_ref2
                        .next(paths)
                        .filter(|_next| point_set.contains(&vertex_ref2))
                        .is_some()
                    {
                        let a = vertex.position();
                        let b = vertex.control_after();
                        let c = vertex.next().unwrap().control_before();
                        let d = vertex.next().unwrap().position();
                        if sqr_distance_bezier_point_lower_bound(a, b, c, d, point) < min_dist {
                            let (dist, closest_on_curve) = sqr_distance_bezier_point_binary(a, b, c, d, point);
                            if dist < min_dist {
                                min_dist = dist;
                                closest_point = Some(closest_on_curve);
                                closest_ref = Some(vertex_ref2.clone());
                            }
                        }
                    } else {
                        let dist = (vertex.position() - point).square_length();
                        if dist < min_dist {
                            min_dist = dist;
                            closest_point = Some(vertex.position());
                            closest_ref = Some(vertex_ref2.clone());
                        }
                    }
                }
                _ => {}
            }
        }

        if let Some(closest_point) = closest_point {
            Some((closest_ref.unwrap(), closest_point, CanvasLength::new(min_dist.sqrt())))
        } else {
            None
        }
    }
}

fn smooth_vertices(selection: &Selection, paths: &mut PathCollection) {
    for vertex in &selection.items {
        if let SelectionReference::VertexReference(vertex) = vertex {
            let mut vertex = paths.resolve_mut(vertex);
            let pos = vertex.position();
            let prev = vertex.prev();
            let next = vertex.next();
            let dir = if prev.is_some() && next.is_some() {
                next.unwrap().position() - prev.unwrap().position()
            } else if let Some(prev) = prev {
                pos - prev.position()
            } else if let Some(next) = next {
                next.position() - pos
            } else {
                continue;
            };

            vertex.set_control_before(pos - dir * 0.25);
            vertex.set_control_after(pos + dir * 0.25);
        }
    }
}

fn drag_at_point(
    selection: &Selection,
    paths: &PathCollection,
    point: CanvasPoint,
    distance_threshold: CanvasLength,
) -> Option<Selection> {
    let selected_vertices: Vec<VertexReference> = selection
        .items
        .iter()
        .filter_map(|v| {
            if let SelectionReference::VertexReference(v_ref) = v {
                Some(*v_ref)
            } else {
                None
            }
        })
        .collect();

    let distance_to_controls = selected_vertices
        .iter()
        .flat_map(|v| vec![v.control_before(&paths), v.control_after(&paths)])
        .map(|v| (v, (paths.resolve(&v).position() - point).length()))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // view.screen_to_canvas_point(capture.mouse_start)
    let closest_vertex = selected_vertices
        .iter()
        .map(|v| (v, (paths.resolve(v).position() - point).length()))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let distance_to_curve = selection.distance_to_curve(&paths, point);

    // Figure out which item is the best to drag
    let mut best_score = CanvasLength::new(std::f32::INFINITY);
    // Lower weights means they are prioritized higher when picking
    const VERTEX_WEIGHT: f32 = 0.5;
    const CONTROL_WEIGHT: f32 = 0.7;
    const CURVE_WEIGHT: f32 = 1.0;

    if let Some(closest_vertex) = &closest_vertex {
        let dist = CanvasLength::new(closest_vertex.1);
        let score = dist * VERTEX_WEIGHT;
        if score < best_score && dist < distance_threshold {
            best_score = score;
        }
    }
    if let Some(distance_to_controls) = &distance_to_controls {
        let dist = CanvasLength::new(distance_to_controls.1);
        let score = dist * CONTROL_WEIGHT;
        if score < best_score && dist < distance_threshold {
            best_score = score;
        }
    }
    if let Some(distance_to_curve) = &distance_to_curve {
        let score = distance_to_curve.2 * CURVE_WEIGHT;
        if score < best_score && distance_to_curve.2 < distance_threshold {
            best_score = score;
        }
    }

    // Figure out which item had the best score and then return that result
    if let Some(closest_vertex) = &closest_vertex {
        let dist = CanvasLength::new(closest_vertex.1);
        let score = dist * VERTEX_WEIGHT;
        if score == best_score {
            // The mouse started at a vertex, this means the user probably wants to drag the existing selection.
            return Some(selection.clone());
        }
    }

    if let Some(distance_to_controls) = &distance_to_controls {
        let dist = CanvasLength::new(distance_to_controls.1);
        let score = dist * CONTROL_WEIGHT;
        if score == best_score {
            return Some(Selection {
                items: vec![distance_to_controls.0.into()],
            });
        }
    }

    if let Some(distance_to_curve) = &distance_to_curve {
        let score = distance_to_curve.2 * CURVE_WEIGHT;
        if score == best_score {
            // The mouse started at a selected curve, this means the user probably wants to drag the existing selection.
            return Some(selection.clone());
        }
    }

    // Nothing was close enough
    None
}

enum SelectState {
    Down(CapturedClick, Selection),
    DragSelect(CapturedClick),
    Dragging(CapturedClick, Selection, Vec<CanvasPoint>),
}

pub struct PathEditor {
    ui_path: PathReference,
    selected: Option<Selection>,
    select_state: Option<SelectState>,
    vector_field: VectorField,
}

impl PathEditor {
    pub fn new(document: &mut Document) -> PathEditor {
        PathEditor {
            ui_path: document.paths.push(PathData::new()),
            selected: None,
            select_state: None,
            vector_field: VectorField {
                primitives: vec![
                    VectorFieldPrimitive::Curl {
                        center: point(0.0, 0.0),
                        strength: 1.0,
                        radius: 500.0,
                    },
                    VectorFieldPrimitive::Curl {
                        center: point(0.0, 50.0),
                        strength: 1.0,
                        radius: 500.0,
                    },
                    VectorFieldPrimitive::Curl {
                        center: point(100.0, 50.0),
                        strength: 1.0,
                        radius: 500.0,
                    },
                    VectorFieldPrimitive::Curl {
                        center: point(200.0, 300.0),
                        strength: 1.0,
                        radius: 2000.0,
                    },
                    VectorFieldPrimitive::Linear {
                        direction: vector(1.0, 1.0),
                        strength: 1.1,
                    },
                ],
            },
        }
    }

    fn update_ui(
        &mut self,
        ui_document: &mut Document,
        document: &mut Document,
        view: &CanvasView,
        input: &InputManager,
        tool: &ToolType,
    ) {
        let ui_path = ui_document.paths.resolve_path_mut(&self.ui_path);
        ui_path.clear();
        ui_path.add_rounded_rect(
            &rect(100.0, 100.0, 200.0, 100.0),
            BorderRadii {
                top_left: 0.0,
                top_right: 1000.0,
                bottom_right: 50.0,
                bottom_left: 50.0,
            },
        );

        let mouse_pos_canvas = view.screen_to_canvas_point(input.mouse_position);

        match tool {
            ToolType::Pencil => {
                // Emulate adding a line_to
                let path = document.paths.last_mut().unwrap();
                if let Some(start) = path.current() {
                    let i = ui_path.move_to(view.canvas_to_screen_point(start.position()).cast_unit());
                    ui_path.line_to(view.canvas_to_screen_point(mouse_pos_canvas).cast_unit());
                    ui_path
                        .point_mut(i)
                        .set_control_after(view.canvas_to_screen_point(start.control_after()).cast_unit());
                    ui_path.end();
                }
            }
            ToolType::Select => {
                ui_path.add_circle(input.mouse_position.cast_unit(), 3.0);

                // Hover visualization for dragging
                if let Some(selected) = &self.selected {
                    let drag = drag_at_point(
                        selected,
                        &document.paths,
                        mouse_pos_canvas,
                        ScreenLength::new(5.0) * view.screen_to_canvas_scale(),
                    );
                    if let Some(drag) = drag {
                        for item in &drag.items {
                            if let SelectionReference::VertexReference(vertex) = item {
                                let vertex = document.paths.resolve(vertex);
                                ui_path.add_circle(view.canvas_to_screen_point(vertex.position()).cast_unit(), 4.0);
                            }
                            if let SelectionReference::ControlPointReference(vertex) = item {
                                let vertex = document.paths.resolve(vertex);
                                ui_path.add_circle(view.canvas_to_screen_point(vertex.position()).cast_unit(), 2.0);
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        if let Some(selected) = &self.selected {
            // let closest = selected.distance_to_curve(&document.paths, mouse_pos_canvas).unwrap();
            // ui_path.add_circle(view.canvas_to_screen_point(closest.1).cast_unit(), 3.0);

            for vertex in &selected.items {
                if let SelectionReference::VertexReference(vertex) = vertex {
                    let vertex = document.paths.resolve(vertex);
                    ui_path.add_circle(view.canvas_to_screen_point(vertex.position()).cast_unit(), 5.0);

                    if vertex.control_before() != vertex.position() {
                        ui_path.add_circle(view.canvas_to_screen_point(vertex.control_before()).cast_unit(), 3.0);
                    }
                    if vertex.control_after() != vertex.position() {
                        ui_path.add_circle(view.canvas_to_screen_point(vertex.control_after()).cast_unit(), 3.0);
                    }

                    if vertex.control_before() != vertex.position() {
                        ui_path.move_to(view.canvas_to_screen_point(vertex.control_before()).cast_unit());
                        ui_path.line_to(view.canvas_to_screen_point(vertex.position()).cast_unit());
                    }
                    if vertex.control_after() != vertex.position() {
                        ui_path.move_to(view.canvas_to_screen_point(vertex.control_after()).cast_unit());
                        ui_path.line_to(view.canvas_to_screen_point(vertex.position()).cast_unit());
                    }
                }
            }
        }
        if let Some(SelectState::DragSelect(capture)) = &self.select_state {
            let start = capture.mouse_start;
            let end = input.mouse_position;
            ui_path.move_to(point(start.x, start.y));
            ui_path.line_to(point(end.x, start.y));
            ui_path.line_to(point(end.x, end.y));
            ui_path.line_to(point(start.x, end.y));
            ui_path.line_to(point(start.x, start.y));
            ui_path.close();
        }

        // let everything = self.select_everything();
        // if let Some((_, closest_point)) = everything.distance_to(mouse_pos) {
        //     dbg!(closest_point);
        //     ui_path.move_to(mouse_pos);
        //     ui_path.line_to(closest_point);
        //     ui_path.end();
        // }

        if document.paths.len() == 0 {
            document.paths.push(PathData::new());
        }

        let path = &mut document.paths.paths[0];
        if path.len() == 0 {
            path.clear();
            for p in &self.vector_field.primitives {
                match p {
                    &VectorFieldPrimitive::Curl { center, .. } => {
                        path.add_circle(center, 1.0);
                    }
                    &VectorFieldPrimitive::Linear { .. } => {}
                }
            }

            let mut rng: StdRng = SeedableRng::seed_from_u64(0);
            let samples = poisson_disc_sampling(rect(-100.0, -100.0, 300.0, 300.0), 80.0, &mut rng);
            for (_i, &p) in samples.iter().enumerate() {
                // if let VectorFieldPrimitive::Linear { ref mut strength, .. } = self.vector_field.primitives.last_mut().unwrap() {
                // *strength = i as f32;
                // }
                path.move_to(p);
                let (vertices, closed) = self.vector_field.trace(p);
                for &p in vertices.iter().skip(1) {
                    path.line_to(p);
                }
                if closed {
                    path.close();
                } else {
                    path.end();
                }
            }
            // if let VectorFieldPrimitive::Linear { ref mut strength, .. } = self.vector_field.primitives.last_mut().unwrap() {
            // *strength = 2.0;
            // }
        }
        // let d = f32::sin(0.01 * (input.frame_count as f32));
        // for x in 0..100 {
        //     for y in 0..100 {
        //         let p = (point(x as f32, y as f32) - vector(50.0, 50.0)) * 4.0;
        //         let dir = self.vector_field.sample(p).unwrap();
        //         path.move_to(p);
        //         path.line_to(p + dir.normalize() * 5.0 * d);
        //         path.end();
        //     }
        // }
    }

    fn update_selection(&mut self, document: &mut Document, view: &CanvasView, input: &mut InputManager) {
        self.select_state = match self.select_state.take() {
            None => {
                if let Some(mut capture) = input.capture_click_batch(winit::event::MouseButton::Left) {
                    for (path_index, path) in document.paths.iter().enumerate() {
                        for point in path.iter_points() {
                            let reference = VertexReference::new(path_index as u32, point.index() as u32);
                            capture.add(
                                CircleShape {
                                    center: view.canvas_to_screen_point(point.position()),
                                    radius: 5.0,
                                },
                                reference,
                            );
                        }
                    }

                    if let Some((capture, best)) = capture.capture(input) {
                        // Something was close enough to the cursor, we should select it
                        Some(SelectState::Down(
                            capture,
                            Selection {
                                items: vec![best.into()],
                            },
                        ))
                    } else if let Some(capture) = input.capture_click(winit::event::MouseButton::Left) {
                        // Start with empty selection
                        Some(SelectState::Down(capture, Selection { items: vec![] }))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            Some(state) => match state {
                // Note: !pressed matches should come first, otherwise mouse_up might be missed if multiple state transitions were to happen in a single frame
                SelectState::DragSelect(capture) if !capture.is_pressed(input) => {
                    // Only select if mouse was released inside the window
                    if capture.on_up(input) {
                        // Select
                        // TODO: Start point should be stored in canvas space, in case the view is zoomed or moved
                        let selection_rect = CanvasRect::from_points(vec![
                            view.screen_to_canvas_point(capture.mouse_start),
                            view.screen_to_canvas_point(input.mouse_position),
                        ]);
                        self.selected = Some(document.select_rect(selection_rect));
                        None
                    } else {
                        self.selected = None;
                        None
                    }
                }
                SelectState::Down(capture, selection) if !capture.is_pressed(input) => {
                    if capture.on_up(input) {
                        // Select
                        self.selected = Some(selection);
                    }
                    None
                }
                SelectState::Down(capture, _)
                    if (capture.mouse_start - input.mouse_position).square_length() > 5.0 * 5.0 =>
                {
                    if let Some(selection) = &self.selected {
                        let drag = drag_at_point(
                            selection,
                            &document.paths,
                            view.screen_to_canvas_point(capture.mouse_start),
                            ScreenLength::new(5.0) * view.screen_to_canvas_scale(),
                        );

                        if let Some(drag) = drag {
                            let original_position = drag.items.iter().map(|x| x.position(&document.paths)).collect();
                            Some(SelectState::Dragging(capture, drag, original_position))
                        } else {
                            Some(SelectState::DragSelect(capture))
                        }
                    } else {
                        // Nothing selected. We should do a drag-select
                        Some(SelectState::DragSelect(capture))
                    }
                }
                SelectState::Dragging(capture, selection, original_positions) => {
                    let offset = view.screen_to_canvas_vector(input.mouse_position - capture.mouse_start);

                    // Move all points
                    for (vertex, &original_position) in selection.items.iter().zip(original_positions.iter()) {
                        if let SelectionReference::VertexReference(vertex) = vertex {
                            document
                                .paths
                                .resolve_mut(vertex)
                                .set_position(original_position + offset);
                        }
                    }

                    // Move all control points
                    // Doing this before moving points will lead to weird results
                    for (vertex, &original_position) in selection.items.iter().zip(original_positions.iter()) {
                        if let SelectionReference::ControlPointReference(vertex) = vertex {
                            // Move control point
                            document
                                .paths
                                .resolve_mut(vertex)
                                .set_position(original_position + offset);

                            // Move opposite control point
                            let center = document.paths.resolve(vertex).vertex().position();
                            let dir = original_position + offset - center;
                            document
                                .paths
                                .resolve_mut(&vertex.opposite_control(&document.paths))
                                .set_position(center - dir);
                        }
                    }

                    if !capture.is_pressed(input) {
                        // Stop drag
                        None
                    } else {
                        Some(SelectState::Dragging(capture, selection, original_positions))
                    }
                }
                SelectState::DragSelect(..) => {
                    // Change visualization?
                    Some(state)
                }
                SelectState::Down(..) => Some(state),
            },
        };

        // self.select_state = state;
        // self.select_state.take

        // if let SelectState::Down(capture, selection) = &self.select_state {
        //     if (capture.mouse_start - input.mouse_position).square_length() > 5.0*5.0 {
        //         self.select_state = SelectState::Dragging(*capture, *selection)
        //     }
        // }

        //     }
        //     SelectState::Down(capture, selection) => {
        //         if (capture.mouse_start - input.mouse_position).square_length() > 5.0*5.0 {
        //             SelectState::Dragging(capture, selection)
        //         } else {
        //             SelectState::Down(capture, selection)
        //         }
        //     }
        //     state => state
        // }
    }

    pub fn update(
        &mut self,
        ui_document: &mut Document,
        document: &mut Document,
        view: &CanvasView,
        input: &mut InputManager,
        tool: &ToolType,
    ) {
        let canvas_mouse_pos = view.screen_to_canvas_point(input.mouse_position);

        match tool {
            ToolType::Select => {}
            _ => self.select_state = None,
        }

        match tool {
            ToolType::Select => {
                self.update_selection(document, view, input);
            }
            ToolType::Pencil => {
                if let Some(_) = input.capture_click(winit::event::MouseButton::Left) {
                    if document.paths.len() == 0 {
                        document.paths.push(PathData::new());
                    }
                    let path = document.paths.last_mut().unwrap();
                    path.line_to(canvas_mouse_pos);
                }
            }
            _ => {}
        }

        if input.on_combination(
            &KeyCombination::new()
                .and(VirtualKeyCode::LControl)
                .and(VirtualKeyCode::S),
        ) {
            // Make smooth
            if let Some(selected) = &self.selected {
                smooth_vertices(selected, &mut document.paths);
            }
        }

        if input.on_combination(
            &KeyCombination::new()
                .and(VirtualKeyCode::LControl)
                .and(VirtualKeyCode::A),
        ) {
            let everything = Some(document.select_everything());
            if self.selected == everything {
                self.selected = None;
            } else {
                self.selected = everything;
            }
        }

        if input.on_combination(
            &KeyCombination::new()
                .and(VirtualKeyCode::LControl)
                .and(VirtualKeyCode::P),
        ) {
            // Debug print selected vertex
            if let Some(selected) = &self.selected {
                if let Some(SelectionReference::VertexReference(vertex)) = selected.items.first() {
                    let vertex = document.paths.resolve(vertex);
                    let prev = vertex.prev().unwrap();
                    let next = vertex.next().unwrap();
                    println!("builder.move_to(point{});", prev.position());
                    println!(
                        "builder.cubic_bezier_to(point{}, point{}, point{});",
                        prev.control_after(),
                        vertex.control_before(),
                        vertex.position()
                    );
                    println!(
                        "builder.cubic_bezier_to(point{}, point{}, point{});",
                        vertex.control_after(),
                        next.control_before(),
                        next.position()
                    );
                    println!(
                        "Tolerance: {}",
                        (ScreenLength::new(0.1) * view.screen_to_canvas_scale())
                            .get()
                            .max(0.001)
                    );
                }
            }
        }

        self.update_ui(ui_document, document, view, input, tool);
    }
}
