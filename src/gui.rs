use std::any::Any;
use std::cell::{RefCell, RefMut};
use std::ops::DerefMut;
use std::marker::PhantomData;
use crate::main::Document;
use crate::canvas::CanvasView;

pub struct WidgetWrapper<T:WidgetTrait+'static> {
    widget: T,
    listeners: Listeners<T::EventType>,
    index: u32,
}

impl<T:WidgetTrait> WidgetTypeTrait for WidgetWrapper<T> {
    fn mount(&mut self, root: &mut Root) {
        self.widget.mount(&mut WidgetContext { root, listeners: &mut self.listeners, index: self.index });
    }
    fn update(&mut self, root: &mut Root) {
        self.widget.update(&mut WidgetContext { root, listeners: &mut self.listeners, index: self.index });
    }
    fn render(&mut self, root: &mut Root, document: &mut Document, view: &CanvasView) {
        *root = Root::new();
        self.widget.render(document, view);
    }

    fn borrow_inner_mut(&mut self) -> &mut dyn Any {
        &mut self.widget
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

enum PointerEvent {
    PointerDown,
    PointerUp,
    PointerEnter,
    PointerExit,
    PointerActive(bool),
}

pub struct WidgetContext<'a, T:WidgetTrait+'static+?Sized> {
    root: &'a mut Root,
    listeners: &'a mut Listeners<T::EventType>,
    index: u32,
}

impl<'a, T:WidgetTrait> WidgetContext<'a, T> {
    pub fn reference(&self) -> TypedWidgetReference<T> {
        TypedWidgetReference {
            reference: WidgetReference {
                index: self.index,
                version: 0,
            },
            _p: PhantomData,
        }
    }
    pub fn wrap<U>(&self, callback: fn(&mut T, &mut WidgetContext<T>, &U)) -> WrappedCallback<T,U> {
        WrappedCallback {
            callback,
            reference: self.reference(),
        }
    }

    // pub fn add_pointer_area(shape, callback: WrappedCallback<T,PointerEvent>) {

    // }

    pub fn send(&mut self, event: &T::EventType) {
        self.listeners.send(self.root, event);
    }

    pub fn add<U:WidgetTrait>(&mut self, widget: U) -> TypedWidgetReference<U> {
        self.root.add(widget)
    }

    pub fn remove(&mut self, reference: impl Into<WidgetReference>) {
        self.root.remove(reference)
    }

    pub fn resolve_mut(&'a self, reference: &WidgetReference) -> RefMut<dyn Any> {
        self.root.resolve_mut(reference)
    }
}

pub struct WrappedCallback<T:WidgetTrait+'static,U:'static> {
    callback: fn(&mut T, &mut WidgetContext<T>, &U),
    reference: TypedWidgetReference<T>,
}

#[derive(Copy,Clone)]
pub struct WidgetReference {
    index: u32,
    version: u32,
}

#[derive(Copy,Clone)]
pub struct TypedWidgetReference<T:WidgetTrait> {
    reference: WidgetReference,
    _p : PhantomData<T>
}

impl<T:WidgetTrait> From<TypedWidgetReference<T>> for WidgetReference {
    fn from(v: TypedWidgetReference<T>) -> WidgetReference {
        v.reference
    }
}

impl<T:WidgetTrait> TypedWidgetReference<T> {
    pub fn listen<U:WidgetTrait>(&self, context: &WidgetContext<U>, callback: fn(&mut U, &mut WidgetContext<U>, &T::EventType)) {
        let callback = context.wrap(callback);
        let mut wrapper = context.root.resolve_mut_wrapper(&self.reference);
        let wrapper: &mut WidgetWrapper<T> = wrapper.as_any_mut().downcast_mut::<WidgetWrapper<T>>().unwrap();
        wrapper.listeners.listen(&callback.reference, callback.callback)
    }

    pub fn listen_closure<U:WidgetTrait, C:Fn(&mut U, &mut WidgetContext<U>, &T::EventType)+'static>(&self, context: &WidgetContext<U>, callback: C) {
        let mut wrapper = context.root.resolve_mut_wrapper(&self.reference);
        let wrapper: &mut WidgetWrapper<T> = wrapper.as_any_mut().downcast_mut::<WidgetWrapper<T>>().unwrap();
        wrapper.listeners.listen_closure(&context.reference(), callback)
    }

    fn get<'a>(&self, root: &'a Root) -> RefMut<'a, T> {
        // Unwrap: We know the type if T since we have a typed reference to it
        RefMut::map(root.resolve_mut(&self.reference), |v| v.downcast_mut::<T>().unwrap())
    }
}

pub struct Listeners<U:Sized+'static> {
    listeners: Vec<(WidgetReference, Box<dyn Fn(&mut dyn Any, &mut Root, &U)>)>
}

impl<U> Listeners<U> {
    pub fn listen<T:WidgetTrait>(&mut self, reference: &TypedWidgetReference<T>, callback: fn(&mut T, &mut WidgetContext<T>, &U)) where T : 'static{
        let c = move|v: &mut dyn Any, root: &mut Root, value: &U| {
            let v = v.downcast_mut::<WidgetWrapper<T>>().unwrap();
            let context = &mut WidgetContext { root, listeners: &mut v.listeners, index: v.index };
            callback(&mut v.widget, context, value);
        };
        self.listeners.push((reference.reference, Box::new(c)));
    }

    pub fn listen_closure<T:WidgetTrait, C:Fn(&mut T, &mut WidgetContext<T>, &U)+'static>(&mut self, reference: &TypedWidgetReference<T>, callback: C) where T : 'static{
        let c = move|v: &mut dyn Any, root: &mut Root, value: &U| {
            let v = v.downcast_mut::<WidgetWrapper<T>>().unwrap();
            let context = &mut WidgetContext { root, listeners: &mut v.listeners, index: v.index };
            callback(&mut v.widget, context, value);
        };
        self.listeners.push((reference.reference, Box::new(c)));
    }

    pub fn send(&self, root: &mut Root, value: &U) {
        for l in &self.listeners {
            let mut wrapper = RefMut::map(root.resolve_mut_wrapper(&l.0), |wrapper| wrapper.as_any_mut());

            // SAFETY: Same as in Root::add
            let root_unsafe = unsafe { &mut *((root as *const Root) as *mut Root) };
            l.1(wrapper.deref_mut(), root_unsafe, value)
        }
    }
}

pub struct Root {
    widgets: Vec<Option<Box<RefCell<dyn WidgetTypeTrait>>>>,
}

impl<'a> Root {
    pub fn new() -> Root {
        Root {
            widgets: vec![]
        }
    }

    pub fn resolve_mut(&'a self, reference: &WidgetReference) -> RefMut<dyn Any> {
        RefMut::map(self.widgets[reference.index as usize].as_ref().unwrap().borrow_mut(), |v| v.borrow_inner_mut())
        // self.root.borrow_mut()
    }

    fn resolve_mut_wrapper(&'a self, reference: &WidgetReference) -> RefMut<dyn WidgetTypeTrait> {
        self.widgets[reference.index as usize].as_ref().unwrap().borrow_mut()
    }

    pub fn remove(&mut self, reference: impl Into<WidgetReference>) {
        let reference = reference.into();
        // SAFATEY: Borrow the item mutably before dropping it.
        // This is important to guarantee we cannot get
        // undefined behaviour as mentioned in #add.
        self.widgets[reference.index as usize].as_ref().unwrap().try_borrow_mut().expect("Cannot remove a widget that is borrowed somewhere.");
        self.widgets[reference.index as usize].take();
    }

    pub fn add<T:WidgetTrait>(&mut self, widget: T) -> TypedWidgetReference<T> {
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
        // Note that we *do* alias &mut self here, however no writes will be done to the pointer
        // using root_unsafe.
        // Quoting the rustonomicon: we don't actually care if aliasing occurs if there aren't any actual writes to memory happening.
        // TODO: This is actually not safe as mount might modify the widgets list and therefore if we are iterating over that in a parent scope
        // then things will go badly.
        let root_unsafe = unsafe { &mut *((self as *const Root) as *mut Root) };
        v.mount(root_unsafe);
        reference
    }

    pub fn update(&mut self) {
        println!("Update");
        for widget in &self.widgets {
            if let Some(widget) = widget {
                // SAFETY: Same as in #add
                let root_unsafe = unsafe { &mut *((self as *const Root) as *mut Root) };
                dbg!(widget.as_ptr());
                widget.borrow_mut().update(root_unsafe);
            }
        }
        for widget in &self.widgets {
            if let Some(widget) = widget {
                widget.borrow_mut();
            }
        }
    }

    pub fn render(&mut self, ui_document: &mut Document, view: &CanvasView) {
        println!("Render");
        for widget in &self.widgets {
            if let Some(widget) = widget {
                // SAFETY: Same as in #add
                let root_unsafe = unsafe { &mut *((self as *const Root) as *mut Root) };
                dbg!(widget.as_ptr());
                widget.borrow_mut().render(root_unsafe, ui_document, view);
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

trait WidgetTypeTrait:Any {
    fn mount(&mut self, root: &mut Root);
    fn update(&mut self, root: &mut Root);
    fn render(&mut self, root: &mut Root, ui_document: &mut Document, view: &CanvasView);
    fn borrow_inner_mut(&mut self) -> &mut dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait WidgetTrait:Any {
    type EventType;
    fn mount(&mut self, _context: &mut WidgetContext<Self>) {}
    fn update(&mut self, _context: &mut WidgetContext<Self>) {}
    fn render(&mut self, ui_document: &mut Document, view: &CanvasView) {}
}
