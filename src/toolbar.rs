use crate::main::{PathEditor, Document};
use crate::canvas::CanvasView;
use crate::input::InputManager;
use crate::path_collection::PathReference;
use crate::path::{PathData, BorderRadii};
use lyon::math::*;
use euclid::default::SideOffsets2D;
use std::any::Any;
use std::cell::{RefCell, RefMut};
use std::ops::DerefMut;
use std::marker::PhantomData;

pub struct Toolbar {
    ui: PathReference,
    tools: Vec<Box<dyn Tool>>
}

pub trait Tool {
    fn activate(&mut self);
    fn deactivate(&mut self);
    fn update_ui(&mut self);
}

impl Toolbar {
    pub fn new(document: &mut Document) -> Toolbar {
        Toolbar {
            ui: document.paths.push(PathData::new()),
            tools: vec![],
        }
    }

    pub fn update_ui(&mut self, ui_document: &mut Document, document: &mut Document, view: &CanvasView, input: &mut InputManager) {
        let path = ui_document.paths.resolve_path_mut(&self.ui);
        path.clear();
        path.add_rounded_rect(
            &rect(50.0, -10.0, view.resolution.width as f32 - 50.0 * 2.0, 60.0).round().outer_rect(SideOffsets2D::new_all_same(0.5)),
            BorderRadii::new_uniform(3.0)
        );
    }
}