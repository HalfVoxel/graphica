use crate::canvas::CanvasView;
use crate::gui::*;
use crate::input::{CapturedClick, InputManager, MouseButton};
use crate::main::Document;
use crate::path::{BorderRadii, PathData};
use crate::path_collection::PathReference;
use euclid::default::SideOffsets2D;
use lyon::math::*;

pub trait Tool {
    fn activate(&mut self);
    fn deactivate(&mut self);
    fn update_ui(&mut self);
}

#[derive(Clone, Copy, Debug)]
pub enum ToolType {
    Pencil,
    Brush,
    Select,
}

pub struct ToolSelected(ToolType);

pub struct Toolbar {
    ui: Option<PathReference>,
    tools: Vec<ToolType>,
}

impl Toolbar {
    pub fn new() -> Toolbar {
        Toolbar {
            ui: None,
            tools: vec![ToolType::Select, ToolType::Pencil, ToolType::Brush],
        }
    }
}

impl WidgetTrait for Toolbar {
    type EventType = ToolSelected;

    fn mount(&mut self, context: &mut WidgetContext<Self>) {
        for (i, &tool) in self.tools.iter().enumerate() {
            // context.add(Button { rect: rect(50.0 + 50.0*(i as f32), 50.0, 40.0, 40.0), path: None }).listen(context, |s, context, event| { s.on_click(context, i) })
            let reference = context.reference();
            context
                .add(Button::new(rect(50.0 + 50.0 * (i as f32), 10.0, 40.0, 40.0)))
                .listen_closure(&reference, move |_, context, _| {
                    println!("Got callback!");
                    context.send(&ToolSelected(tool))
                })
        }
    }

    fn render(&mut self, ui_document: &mut Document, view: &CanvasView) {
        if self.ui.is_none() {
            self.ui = Some(ui_document.paths.push(PathData::new()));
        }

        let path = ui_document.paths.resolve_path_mut(&self.ui.unwrap());
        path.clear();
        path.add_rounded_rect(
            &rect(40.0, -10.0, view.resolution.width as f32 - 40.0 * 2.0, 70.0)
                .round()
                .outer_rect(SideOffsets2D::new_all_same(0.5)),
            BorderRadii::new_uniform(3.0),
        );
    }
}

pub struct GUIRoot {
    pub tool: ToolType,
}

impl GUIRoot {
    pub fn new() -> GUIRoot {
        GUIRoot { tool: ToolType::Select }
    }

    fn on_change_tool(&mut self, _context: &mut WidgetContext<Self>, ev: &ToolSelected) {
        let tool = ev.0;
        self.tool = tool;
    }
}

impl WidgetTrait for GUIRoot {
    type EventType = ();

    fn mount(&mut self, context: &mut WidgetContext<Self>) {
        let reference = context.reference();
        context
            .add(Toolbar::new())
            .listen_closure(&reference, Self::on_change_tool);
    }
}

struct Button {
    rect: Rect,
    path: Option<PathReference>,
    capture: Option<CapturedClick>,
}

impl Button {
    fn new(rect: Rect) -> Button {
        Button {
            rect,
            path: None,
            capture: None,
        }
    }
}

impl WidgetTrait for Button {
    type EventType = ButtonEvent;

    fn mount(&mut self, _context: &mut WidgetContext<Self>) {}

    fn render(&mut self, ui_document: &mut Document, _view: &CanvasView) {
        if self.path.is_none() {
            self.path = Some(ui_document.paths.push(PathData::new()));
        }
        let path = ui_document.paths.resolve_path_mut(&self.path.unwrap());
        path.clear();
        let active = self.capture.is_some();
        path.add_rounded_rect(
            &self
                .rect
                .round()
                .outer_rect(SideOffsets2D::new_all_same(0.5))
                .inner_rect(SideOffsets2D::new_all_same(if active { 1.0 } else { 0.0 })),
            BorderRadii::new_uniform(3.0),
        );
    }

    fn update(&mut self, _context: &mut WidgetContext<Self>) {}

    fn input(&mut self, context: &mut WidgetContext<Self>, input: &mut InputManager) {
        let inside = self.rect.contains(input.mouse_position.to_untyped());
        if inside && self.capture.is_none() {
            self.capture = input.capture_click(MouseButton::Left);
        }

        if let Some(capture) = &self.capture {
            if capture.on_up(input) {
                self.capture = None;
                if inside {
                    context.send(&ButtonEvent::Click);
                }
            }
        }
    }
}

// impl WidgetTrait for Widget {
//     type EventType = ButtonEvent;
// }

enum ButtonEvent {
    Click,
}

impl Widget {
    fn on_click(&mut self, context: &mut WidgetContext<Self>, ev: &ButtonEvent) {
        context.add(Button::new(rect(0.0, 0.0, 0.0, 0.0)));
        context.send(ev);
    }
}

impl WidgetTrait for Widget {
    type EventType = ButtonEvent;

    fn mount(&mut self, context: &mut WidgetContext<Self>) {
        let reference = context.reference();
        context
            .add(Button::new(rect(0.0, 0.0, 0.0, 0.0)))
            .listen_closure(&reference, Self::on_click);

        // let v = reference.get(context);
        // let v2 = reference.get(context);
        // context.remove(reference.reference());
        // context.listen(reference, Self::on_click);

        // self.listeners.send(root, &ButtonEvent::Click);
    }

    fn update(&mut self, _context: &mut WidgetContext<Self>) {
        // context.add(Button::new());
    }
}

struct Widget {}

#[test]
fn test_gui() {
    let mut root = Root::new();
    let r1 = root.add(Widget {});
    root.remove(r1);
    root.update();
    // assert_eq!(root.widgets.len(), 3);
    // panic!();
}
