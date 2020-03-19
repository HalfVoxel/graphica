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

struct WrappedCallback<T:WidgetTrait+'static,U:'static> {
    callback: fn(&mut T, &mut WidgetContext<T>, &U),
    reference: TypedWidgetReference<T>,
}

#[derive(Copy,Clone)]
struct WidgetReference {
    index: u32,
    version: u32,
}

#[derive(Copy,Clone)]
struct TypedWidgetReference<T:WidgetTrait> {
    reference: WidgetReference,
    _p : PhantomData<T>
}

impl<T:WidgetTrait> TypedWidgetReference<T> {
    fn new() -> TypedWidgetReference<T> {
        TypedWidgetReference {
            reference: WidgetReference { index: 0, version: 0 },
            _p: PhantomData{}
        }
    }
}

impl<T:WidgetTrait> TypedWidgetReference<T> {
    fn listen<U:WidgetTrait>(&self, root: &Root, callback: WrappedCallback<U,T::EventType>) {
        let mut wrapper = root.resolve_mut_wrapper(&self.reference);
        let wrapper: &mut WidgetWrapper<T> = wrapper.as_any_mut().downcast_mut::<WidgetWrapper<T>>().unwrap();
        wrapper.listeners.listen(&callback.reference, callback.callback)
    }

    fn get<'a>(&self, root: &'a Root) -> RefMut<'a, T> {
        // Unwrap: We know the type if T since we have a typed reference to it
        RefMut::map(root.resolve_mut(&self.reference), |v| v.downcast_mut::<T>().unwrap())
    }
}

struct Listeners<U:Sized+'static> {
    listeners: Vec<(WidgetReference, Box<dyn Fn(&mut dyn Any, &mut Root, &U)>)>
}

impl<U> Listeners<U> {
    fn listen<T:WidgetTrait>(&mut self, reference: &TypedWidgetReference<T>, callback: fn(&mut T, &mut WidgetContext<T>, &U)) where T : 'static{
        let c = move|v: &mut dyn Any, root: &mut Root, value: &U| {
            let v = v.downcast_mut::<WidgetWrapper<T>>().unwrap();
            let context = &mut WidgetContext { root, listeners: &mut v.listeners, index: v.index };
            callback(&mut v.widget, context, value);
        };
        self.listeners.push((reference.reference, Box::new(c)));
    }

    fn send(&self, root: &mut Root, value: &U) {
        for l in &self.listeners {
            let mut wrapper = RefMut::map(root.resolve_mut_wrapper(&l.0), |wrapper| wrapper.as_any_mut());

            // SAFETY: Same as in Root::add
            let root_unsafe = unsafe { &mut *((root as *const Root) as *mut Root) };
            l.1(wrapper.deref_mut(), root_unsafe, value)
        }
    }
}

struct Root {
    widgets: Vec<Option<Box<RefCell<dyn WidgetTypeTrait>>>>,
}

impl<'a> Root {
    fn resolve_mut(&'a self, reference: &WidgetReference) -> RefMut<dyn Any> {
        RefMut::map(self.widgets[reference.index as usize].as_ref().unwrap().borrow_mut(), |v| v.borrow_inner_mut())
        // self.root.borrow_mut()
    }

    fn resolve_mut_wrapper(&'a self, reference: &WidgetReference) -> RefMut<dyn WidgetTypeTrait> {
        self.widgets[reference.index as usize].as_ref().unwrap().borrow_mut()
    }

    fn remove(&mut self, reference: &WidgetReference) {
        // SAFATEY: Borrow the item mutably before dropping it.
        // This is important to guarantee we cannot get
        // undefined behaviour as mentioned in #add.
        self.widgets[reference.index as usize].as_ref().unwrap().try_borrow_mut().expect("Cannot remove a widget that is borrowed somewhere.");
        self.widgets[reference.index as usize].take();
    }

    fn add<T:WidgetTrait>(&mut self, widget: T) -> TypedWidgetReference<T> {
        self.widgets.push(Some(into_wrapper(widget, self.widgets.len() as u32)));
        let reference = TypedWidgetReference {
            reference: WidgetReference {
                index: (self.widgets.len() - 1) as u32,
                version: 0,
            },
            _p: PhantomData,
        };
        let mut v = self.resolve_mut_wrapper(&reference.reference);

        // SAFETY: Here we give a mutable pointer to &Root while
        // still allowing #v to be mutated inside v.mount.
        // This is safe because v.mount cannot access any part of Root mutably or immutably
        // that overlaps with the reference in v.
        // Note that even if another widget is added inside #mount which causes
        // the #widgets vector to be resized the #v borrow points to inside the Box
        // so it will not be affected.
        // The only case where things could go wrong is if the Box that stores
        // the widget is dropped. This is not allowed by the API however since
        // all calls that do that must check if they have mutable access to the widget's RefCell first.
        let root_unsafe = unsafe { &mut *((self as *const Root) as *mut Root) };
        v.mount(root_unsafe);
        reference
    }

    fn update(&mut self) {
        // SAFETY: Same as in #add
        let root_unsafe = unsafe { &mut *((self as *const Root) as *mut Root) };

        for widget in &self.widgets {
            if let Some(widget) = widget {
                widget.borrow_mut().update(root_unsafe);
            }
        }
    }
}

fn into_wrapper<T:WidgetTrait>(value: T, index: u32) -> Box<RefCell<dyn WidgetTypeTrait>> {
    Box::new(RefCell::new(WidgetWrapper {
        widget: value,
        listeners: Listeners {
            listeners: vec![],
        },
        index,
    }))
}

struct OutEvents {}

struct Widget {
}

trait WidgetTypeTrait:Any {
    fn mount(&mut self, root: &mut Root);
    fn update(&mut self, root: &mut Root);
    fn borrow_inner_mut(&mut self) -> &mut dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

trait WidgetTrait:Any {
    type EventType;
    fn mount(&mut self, context: &mut WidgetContext<Self>) {}
    fn update(&mut self, context: &mut WidgetContext<Self>) {}
}

struct Button {
}

impl WidgetTrait for Button {
    type EventType = ButtonEvent;

    fn mount(&mut self, context: &mut WidgetContext<Self>) {
        context.send(&ButtonEvent::Click);
    }
}

// impl WidgetTrait for Widget {
//     type EventType = ButtonEvent;
// }

enum ButtonEvent {
    Click,
}

struct WidgetWrapper<T:WidgetTrait+'static> {
    widget: T,
    listeners: Listeners<T::EventType>,
    index: u32,
}

impl<T:WidgetTrait> WidgetTypeTrait for WidgetWrapper<T> {
    fn mount(&mut self, root: &mut Root) {
        self.widget.mount(&mut WidgetContext { root, listeners: &mut self.listeners, index: self.index });
    }
    fn update(&mut self, root: &mut Root) {
        self.widget.mount(&mut WidgetContext { root, listeners: &mut self.listeners, index: self.index });
    }
    fn borrow_inner_mut(&mut self) -> &mut dyn Any {
        &mut self.widget
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

struct WidgetContext<'a, T:WidgetTrait+'static+?Sized> {
    root: &'a mut Root,
    listeners: &'a mut Listeners<T::EventType>,
    index: u32,
}

impl<'a, T:WidgetTrait> WidgetContext<'a, T> {
    fn reference(&self) -> TypedWidgetReference<T> {
        TypedWidgetReference {
            reference: WidgetReference {
                index: self.index,
                version: 0,
            },
            _p: PhantomData,
        }
    }
    fn wrap<U>(&self, callback: fn(&mut T, &mut WidgetContext<T>, &U)) -> WrappedCallback<T,U> {
        WrappedCallback {
            callback,
            reference: self.reference(),
        }
    }

    fn send(&mut self, event: &T::EventType) {
        self.listeners.send(self.root, event);
    }
}

impl Widget {
    fn on_click(&mut self, context: &mut WidgetContext<Self>, ev: &ButtonEvent) {
        context.root.add(Button{});
        context.send(ev);
    }
}

impl WidgetTrait for Widget {
    type EventType = ButtonEvent;

    fn mount(&mut self, context: &mut WidgetContext<Self>) {
        let reference = context.root.add(Button{});
        reference.listen(context.root, context.wrap(Self::on_click));

        // let v = reference.get(context.root);
        // let v2 = reference.get(context.root);
        context.root.remove(&reference.reference);
        // context.listen(reference, Self::on_click);

        // self.listeners.send(root, &ButtonEvent::Click);
    }

    fn update(&mut self, context: &mut WidgetContext<Self>) {
        context.root.add(Button{});
    }
}

#[test]
fn test_gui() {
    let mut root = Root {
        widgets: vec![],
    };

    let r1 = root.add(Widget{});
    root.update();
    assert_eq!(root.widgets.len(), 3);
    panic!();
}