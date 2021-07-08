use crate::canvas::CanvasView;
use crate::geometry_utilities::types::*;

use crate::input::*;
use crate::main::Document;
use crate::path::*;
use crate::path_collection::PathReference;
use crate::toolbar::ToolType;

use palette::Srgba;

enum BrushEditorState {
    Dragging(CapturedClick, CanvasPoint),
}

pub struct BrushEditor {
    state: Option<BrushEditorState>,
    debug_path: Option<PathReference>,
    color: Srgba,
}

impl BrushEditor {
    pub fn new() -> BrushEditor {
        BrushEditor {
            state: None,
            debug_path: None,
            color: Srgba::new(1.0, 0.0, 0.0, 1.0),
        }
    }

    pub fn update(
        &mut self,
        _ui_document: &mut Document,
        document: &mut Document,
        view: &CanvasView,
        input: &mut InputManager,
        tool: &ToolType,
    ) {
        puffin::profile_function!();
        let data = &mut document.brushes;
        let mouse_pos_canvas = view.screen_to_canvas_point(input.mouse_position);
        if self.debug_path.is_none() {
            self.debug_path = Some(document.paths.push(PathData::default()));
        }

        let p = document.paths.resolve_path_mut(&self.debug_path.unwrap());
        p.clear();

        #[allow(clippy::single_match)]
        match tool {
            ToolType::Brush => {
                // p.copy_from(&data.path);
                self.state = match self.state.take() {
                    None => {
                        if let Some(captured) = input.capture_click(MouseButton::Left) {
                            data.move_to(mouse_pos_canvas, self.color);
                            Some(BrushEditorState::Dragging(captured, mouse_pos_canvas))
                        } else {
                            None
                        }
                    }
                    Some(brush_state) => match brush_state {
                        BrushEditorState::Dragging(capture, last_point) => {
                            if !capture.is_pressed(input) {
                                None
                            } else if (last_point - mouse_pos_canvas).length() > 2.0 {
                                data.line_to(mouse_pos_canvas, self.color);
                                Some(BrushEditorState::Dragging(capture, mouse_pos_canvas))
                            } else {
                                Some(BrushEditorState::Dragging(capture, last_point))
                            }
                        }
                    },
                }
            }
            _ => {}
        }
    }
}

#[allow(dead_code)]
struct BrushVertexAttrs {
    time: f32,
    pressure: f32,
}

#[allow(dead_code)]
struct BrushSubpathAttrs {
    color: Srgba,
}

pub struct TextureReference {
    pub id: i64,
}

pub struct Brush {
    pub tip_texture: TextureReference,
    pub spacing: f32,
    pub size: f32,
}

pub struct BrushData {
    pub path: PathData,
    pub colors: Vec<Srgba>,
    pub brush: Brush,
}

impl BrushData {
    #[allow(clippy::new_without_default)]
    pub fn new() -> BrushData {
        BrushData {
            path: PathData::default(),
            colors: vec![],
            brush: Brush {
                tip_texture: TextureReference { id: 0 },
                spacing: 0.25,
                size: 20.0,
            },
        }
    }

    fn move_to(&mut self, p: CanvasPoint, color: Srgba) {
        self.path.move_to(p);
        self.colors.push(color);
        self.path.dirty();
    }

    fn line_to(&mut self, p: CanvasPoint, color: Srgba) {
        self.path.line_to(p);
        self.colors.push(color);
        self.path.dirty();
    }

    pub fn clear(&mut self) {
        self.path.clear();
        self.colors.clear();
        self.path.dirty();
    }

    pub fn hash(&self) -> u64 {
        self.path.hash()
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
