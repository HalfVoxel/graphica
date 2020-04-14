use crate::main::{Document};
use crate::path::*;
use lyon::math::*;
use std::sync::Arc;
use crate::input::*;
use crate::canvas::CanvasView;
use crate::geometry_utilities::types::*;
use crate::geometry_utilities::{poisson_disc_sampling, sqr_distance_bezier_point, sqr_distance_bezier_point_binary, sqr_distance_bezier_point_lower_bound, VectorField, VectorFieldPrimitive};
use crate::path_collection::{
    ControlPointReference, MutableReferenceResolver, PathCollection, ReferenceResolver, SelectionReference,
    VertexReference, PathReference
};
use std::collections::{HashMap, HashSet};
use rand::{rngs::StdRng, SeedableRng};
use crate::toolbar::ToolType;
use palette::Srgba;

pub struct BrushEditor {
}

impl BrushEditor {
    pub fn new() -> BrushEditor {
        BrushEditor {}
    }
    
    pub fn update(&mut self, ui_document: &mut Document, document: &mut Document, view: &CanvasView, input: &mut InputManager, tool: &ToolType) {
        let data = &mut document.brushes;
        let mouse_pos_canvas = view.screen_to_canvas_point(input.mouse_position);

        match tool {
            ToolType::Brush => {
                if let Some(captured) = input.capture_click(MouseButton::Left) {
                    data.line_to(mouse_pos_canvas, Srgba::new(1.0, 0.0, 0.0, 1.0));
                }
            }
            _ => {}
        }
    }
}

struct BrushVertexAttrs {
    time: f32,
    pressure: f32,
}

struct BrushSubpathAttrs {
    color: Srgba,
}

pub struct TextureReference {
    pub id: i64,
}

pub struct Brush {
    pub tip_texture: TextureReference,
    pub spacing: f32,
}

pub struct BrushData {
    pub path: PathData,
    pub colors: Vec<Srgba>,
    pub brush: Brush,
}

impl BrushData {
    pub fn new() -> BrushData {
        BrushData {
            path: PathData::new(),
            colors: vec![],
            brush: Brush {
                tip_texture: TextureReference { id: 0 },
                spacing: 5.0,
            }
        }
    }

    fn line_to(&mut self, p: CanvasPoint, color: Srgba) {
        self.path.line_to(p);
        self.colors.push(color);
    }
}

// {
//     // Mode: fn(Option<prev>, current)
//     // From current, from previous

//     position,
//     control_before,
//     control_after,
//     attr1,
//     attr2,
// }

// trait Storage {
//     fn sub_storage(&self) -> impl Iterator<Item=SubStorage>;
// }

// // enum UndoItem {
// //     Connect,
// //     ModifyVector2(bufferindex, index, value),
// //     ModifyF32(bufferindex, index, value),
// //     ModifyI32(bufferindex, value),
// //     ExtendBufferBundle(bundleindex, )
// //     Select
// // }

// // Path add vertex tool semantics
// // Click
// //    if over vertex: select vertex
// //    if over vertex or control: drag
// //    if over curve: split at point
// //    otherwise: add new subpath with point

// enum SelectionEvent {
//     Select(rect),
//     MoveSelection { selection_id: i32, offset: Vector2D },
// }

// PathEditorEvent {
//     AddVertexAfter(vertex_reference, id: vertex_id),
//     AddVertex(path),
//     RemoveVertex(vertex_reference),
// }

// enum Event {
//     Select(rect),
//     Selection(SelectionEvent)
//     PathEditor(PathEditor::Event),
//     BrushEditor(BrushEditor::Event),
// }

// // Move vertices
// tracker = ...
// {
//     buffer_mut = tracker.modify_buffer(buffer);
//     buffer_mut.set(vertex, buffer[vertex]);
// }
// document.apply(tracker);

// impl Storage for Data {
//     fn sub_storage(&self) -> impl Iterator<Item=SubStorage> {
        
//     }
// }

// trait SubStorage {
//     fn add();
// }

// impl ColorStorage for Data {
//     fn color(Vertex) {
//         self.color[vertex]
//     }
// }

// impl PathStorage {
//     fn meta() -> &PathMeta {}
// }

// impl GeometryStorage for A where A: PathStorage+PositionStorage+ControlBeforeStorage+ControlAfterStorage {

// }
// trait GeometryStorage: PathStorage, PositionStorage, ControlBeforeStorage, ControlAfterStorage {
    
// }

// for vertex in path {
//     x = path.position(vertex);
//     *path.position_mut(vertex) = point(0.0);
//     path.prev_control(path.prev(vertex));
//     vertex.prev().prev_control();
// }

// enum Attribute {
//     Position,
//     ControlBefore,
//     ControlAfter,
//     Pressure,
//     Time,
// }

// position, color, control_before, control_after = resolve!(vertex, [Attribute::Position, Attribute::Color, Attribute::ControlBefore, Attribute::ControlAfter]);