use crate::main::{PathEditor, Document};
use crate::canvas::CanvasView;
use crate::input::InputManager;
use crate::path_collection::PathReference;
use crate::path::{PathData, BorderRadii};
use lyon::math::*;
use euclid::default::SideOffsets2D;
use crate::gui::*;

pub trait Tool {
    fn activate(&mut self);
    fn deactivate(&mut self);
    fn update_ui(&mut self);
}

#[derive(Clone, Copy)]
pub enum ToolType {
    Pencil,
    Brush,
    Select,
}

pub enum ToolbarEvent {
    ToolSelected(ToolType)
}

pub struct Toolbar {
    ui: PathReference,
    tools: Vec<Box<dyn Tool>>,
}

impl Toolbar {
    pub fn new(document: &mut Document) -> Toolbar {
        Toolbar {
            ui: document.paths.push(PathData::new()),
            tools: vec![],
        }
    }

    fn on_click(&mut self, context: &mut WidgetContext<Self>, button_index: i32) {

    }
}

impl WidgetTrait for Toolbar {
    type EventType = ToolbarEvent;

    fn mount(&mut self, context: &mut WidgetContext<Self>) {
        let tools = vec![
            ToolType::Select,
            ToolType::Pencil,
            ToolType::Brush,
        ];
        for (i, &tool) in tools.iter().enumerate() {
            // context.add(Button { rect: rect(50.0 + 50.0*(i as f32), 50.0, 40.0, 40.0), path: None }).listen(context, |s, context, event| { s.on_click(context, i) })
            context.add(Button { rect: rect(50.0 + 50.0*(i as f32), 50.0, 40.0, 40.0), path: None })
                .listen_closure(context, move|_, context, _| {
                    context.send(&ToolbarEvent::ToolSelected(tool))
                })
        }
    }

    fn render(&mut self, ui_document: &mut Document, view: &CanvasView) {
        // let path = ui_document.paths.resolve_path_mut(&self.ui);
        // path.clear();
        // path.add_rounded_rect(
        //     &rect(50.0, -10.0, view.resolution.width as f32 - 50.0 * 2.0, 60.0).round().outer_rect(SideOffsets2D::new_all_same(0.5)),
        //     BorderRadii::new_uniform(3.0)
        // );
    }
}

struct Button {
    rect: Rect,
    path: Option<PathReference>,
}

impl Button {
    fn new() -> Button {
        Button {
            rect: rect(10.0, 10.0, 10.0, 10.0),
            path: None,
        }
    }
}

impl WidgetTrait for Button {
    type EventType = ButtonEvent;

    fn mount(&mut self, context: &mut WidgetContext<Self>) {
        // context.send(&ButtonEvent::Click);
    }

    fn render(&mut self, ui_document: &mut Document, view: &CanvasView) {
        // if self.path.is_none() {
        //     self.path = Some(ui_document.paths.push(PathData::new()));
        // }
        // let path = ui_document.paths.resolve_path_mut(&self.path.unwrap());
        // path.clear();
        // path.add_rounded_rect(
        //     &self.rect,
        //     BorderRadii::new_uniform(3.0)
        // );
    }

    fn update(&mut self, context: &mut WidgetContext<Self>) {

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
        context.add(Button::new());
        context.send(ev);
    }
}

impl WidgetTrait for Widget {
    type EventType = ButtonEvent;

    fn mount(&mut self, context: &mut WidgetContext<Self>) {
        let reference = context.add(Button::new());
        reference.listen(context, Self::on_click);

        // let v = reference.get(context);
        // let v2 = reference.get(context);
        context.remove(reference);
        // context.listen(reference, Self::on_click);

        // self.listeners.send(root, &ButtonEvent::Click);
    }

    fn update(&mut self, context: &mut WidgetContext<Self>) {
        // context.add(Button::new());
    }
}

struct Widget {
}

#[test]
fn test_gui() {
    let mut root = Root::new();
    let r1 = root.add(Widget{});
    root.remove(r1);
    root.update();
    // assert_eq!(root.widgets.len(), 3);
    // panic!();
}