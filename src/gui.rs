use crate::canvas::CanvasView;
use crate::input::InputManager;
use crate::main::Document;
use std::any::Any;
use std::cell::{RefCell, RefMut};
use std::marker::PhantomData;
use std::ops::DerefMut;

pub struct WidgetWrapper<T: WidgetTrait + 'static> {
    widget: T,
    listeners: Listeners<T::EventType>,
    index: u32,
}

impl<T: WidgetTrait + 'static> WidgetWrapper<T> {
    pub fn listen_closure<
        U: WidgetTrait + 'static,
        C: Fn(&mut U, &mut WidgetContext<U>, &T::EventType) + 'static,
    >(
        &mut self,
        reference: &TypedWidgetReference<U>,
        callback: C,
    ) {
        self.listeners.listen_closure(reference, callback);
    }
}

impl<T: WidgetTrait> WidgetTypeTrait for WidgetWrapper<T> {
    fn mount(&mut self, root: &mut RootWrapper) {
        self.widget.mount(&mut WidgetContext {
            root,
            listeners: &mut self.listeners,
            index: self.index,
        });
    }
    fn update(&mut self, root: &mut RootWrapper) {
        self.widget.update(&mut WidgetContext {
            root,
            listeners: &mut self.listeners,
            index: self.index,
        });
    }
    fn input(&mut self, root: &mut RootWrapper, input: &mut InputManager) {
        self.widget.input(
            &mut WidgetContext {
                root,
                listeners: &mut self.listeners,
                index: self.index,
            },
            input,
        );
    }
    fn render(&mut self, _root: &mut RootWrapper, document: &mut Document, view: &CanvasView) {
        self.widget.render(document, view);
    }

    fn borrow_inner_mut(&mut self) -> &mut dyn Any {
        &mut self.widget
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// enum PointerEvent {
//     PointerDown,
//     PointerUp,
//     PointerEnter,
//     PointerExit,
//     PointerActive(bool),
// }

pub struct WidgetContext<'a, 'b, T: WidgetTrait + 'static + ?Sized> {
    root: &'a mut RootWrapper<'b>,
    listeners: &'a mut Listeners<T::EventType>,
    index: u32,
}

impl<'a, 'b, T: WidgetTrait> WidgetContext<'a, 'b, T> {
    pub fn reference(&self) -> TypedWidgetReference<T> {
        TypedWidgetReference {
            reference: WidgetReference {
                index: self.index,
                _version: 0,
            },
            _p: PhantomData,
        }
    }
    pub fn wrap<U>(&self, callback: fn(&mut T, &mut WidgetContext<T>, &U)) -> WrappedCallback<T, U> {
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

    pub fn add<U: WidgetTrait>(&mut self, widget: U) -> RefMut<WidgetWrapper<U>> {
        self.root.add(widget)
    }

    pub fn remove(&mut self, reference: impl Into<WidgetReference>) {
        self.root.remove(reference)
    }

    pub fn resolve_mut(&'a self, reference: &WidgetReference) -> RefMut<dyn Any> {
        self.root.resolve_mut(reference)
    }
}

pub struct WrappedCallback<T: WidgetTrait + 'static, U: 'static> {
    callback: fn(&mut T, &mut WidgetContext<T>, &U),
    reference: TypedWidgetReference<T>,
}

#[derive(Copy, Clone)]
pub struct WidgetReference {
    index: u32,
    _version: u32,
}

#[derive(Copy, Clone)]
pub struct TypedWidgetReference<T: WidgetTrait> {
    reference: WidgetReference,
    _p: PhantomData<T>,
}

impl<T: WidgetTrait> From<TypedWidgetReference<T>> for WidgetReference {
    fn from(v: TypedWidgetReference<T>) -> WidgetReference {
        v.reference
    }
}

impl<T: WidgetTrait> TypedWidgetReference<T> {
    pub fn listen<U: WidgetTrait>(
        &self,
        context: &WidgetContext<U>,
        callback: fn(&mut U, &mut WidgetContext<U>, &T::EventType),
    ) {
        let callback = context.wrap(callback);
        let mut wrapper = context.root.resolve_mut_wrapper(&self.reference);
        let wrapper: &mut WidgetWrapper<T> = wrapper.as_any_mut().downcast_mut::<WidgetWrapper<T>>().unwrap();
        wrapper.listeners.listen(&callback.reference, callback.callback)
    }

    pub fn listen_closure<U: WidgetTrait, C: Fn(&mut U, &mut WidgetContext<U>, &T::EventType) + 'static>(
        &self,
        context: &WidgetContext<U>,
        callback: C,
    ) {
        let mut wrapper = context.root.resolve_mut_wrapper(&self.reference);
        let wrapper: &mut WidgetWrapper<T> = wrapper.as_any_mut().downcast_mut::<WidgetWrapper<T>>().unwrap();
        wrapper.listeners.listen_closure(&context.reference(), callback)
    }

    pub fn get<'a>(&self, root: &'a Root) -> RefMut<'a, T> {
        // Unwrap: We know the type if T since we have a typed reference to it
        RefMut::map(root.resolve_mut(&self.reference), |v| v.downcast_mut::<T>().unwrap())
    }
}

pub struct Listeners<U: Sized + 'static> {
    listeners: Vec<(
        WidgetReference,
        Box<dyn for<'a, 'b> Fn(&mut dyn Any, &'b mut RootWrapper<'a>, &U)>,
    )>,
}

impl<U> Listeners<U> {
    pub fn listen<T: WidgetTrait>(
        &mut self,
        reference: &TypedWidgetReference<T>,
        callback: fn(&mut T, &mut WidgetContext<T>, &U),
    ) where
        T: 'static,
    {
        let c = move |v: &mut dyn Any, root: &mut RootWrapper, value: &U| {
            let v = v.downcast_mut::<WidgetWrapper<T>>().unwrap();
            let context = &mut WidgetContext {
                root,
                listeners: &mut v.listeners,
                index: v.index,
            };
            callback(&mut v.widget, context, value);
        };
        self.listeners.push((reference.reference, Box::new(c)));
    }

    pub fn listen_closure<T: WidgetTrait, C: Fn(&mut T, &mut WidgetContext<T>, &U) + 'static>(
        &mut self,
        reference: &TypedWidgetReference<T>,
        callback: C,
    ) where
        T: 'static,
    {
        let c = move |v: &mut dyn Any, root: &mut RootWrapper, value: &U| {
            let v = v.downcast_mut::<WidgetWrapper<T>>().unwrap();
            let context = &mut WidgetContext {
                root,
                listeners: &mut v.listeners,
                index: v.index,
            };
            callback(&mut v.widget, context, value);
        };
        self.listeners.push((reference.reference, Box::new(c)));
    }

    pub fn send(&self, root: &mut RootWrapper, value: &U) {
        for l in &self.listeners {
            let mut wrapper = RefMut::map(root.resolve_mut_wrapper(&l.0), |wrapper| wrapper.as_any_mut());

            // SAFETY: Same as in Root::add
            // let root_unsafe = unsafe { &mut *((root as *const Root) as *mut Root) };
            l.1(wrapper.deref_mut(), root, value)
        }
    }
}

pub struct RootWrapper<'a> {
    root: &'a Root,
    to_add: Vec<Box<RefCell<dyn WidgetTypeTrait>>>,
    to_remove: Vec<WidgetReference>,
}

impl<'a> RootWrapper<'a> {
    fn new(root: &Root) -> RootWrapper {
        RootWrapper {
            root,
            to_add: vec![],
            to_remove: vec![],
        }
    }

    pub fn resolve_mut(&self, reference: &WidgetReference) -> RefMut<dyn Any> {
        self.root.resolve_mut(reference)
    }

    fn resolve_mut_wrapper(&self, reference: &WidgetReference) -> RefMut<'a, dyn WidgetTypeTrait> {
        dbg!("Resolving", reference.index);
        // if reference.index as usize >= self.root.widgets.len() {
        // self.to_add[reference.index as usize].borrow_mut()
        // } else {
        self.root.resolve_mut_wrapper(reference)
        // }
    }

    fn add<T: WidgetTrait>(&mut self, widget: T) -> RefMut<WidgetWrapper<T>> {
        let index = (self.root.widgets.len() + self.to_add.len()) as u32;
        self.to_add.push(into_wrapper(widget, index));

        // TypedWidgetReference {
        //     reference: WidgetReference {
        //         index,
        //         _version: 0,
        //     },
        //     _p: PhantomData,
        // }
        return RefMut::map(self.to_add.last().unwrap().borrow_mut(), |x| {
            x.as_any_mut().downcast_mut::<WidgetWrapper<T>>().unwrap()
        });
    }

    fn remove(&mut self, reference: impl Into<WidgetReference>) {
        self.to_remove.push(reference.into());
    }
}

#[derive(Default)]
pub struct Root {
    widgets: Vec<Option<Box<RefCell<dyn WidgetTypeTrait>>>>,
}

impl<'a> Root {
    pub fn resolve_mut(&'a self, reference: &WidgetReference) -> RefMut<dyn Any> {
        RefMut::map(
            self.widgets[reference.index as usize].as_ref().unwrap().borrow_mut(),
            |v| v.borrow_inner_mut(),
        )
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
        self.widgets[reference.index as usize]
            .as_ref()
            .unwrap()
            .try_borrow_mut()
            .expect("Cannot remove a widget that is borrowed somewhere.");
        self.widgets[reference.index as usize].take();
    }

    pub fn add<T: WidgetTrait>(&mut self, widget: T) -> TypedWidgetReference<T> {
        self.widgets.push(Some(into_wrapper(widget, self.widgets.len() as u32)));
        let reference = TypedWidgetReference {
            reference: WidgetReference {
                index: (self.widgets.len() - 1) as u32,
                _version: 0,
            },
            _p: PhantomData,
        };

        let mut mods = RootWrapper::new(self);
        self.resolve_mut_wrapper(&reference.reference).mount(&mut mods);

        let RootWrapper { to_add, to_remove, .. } = mods;
        self.apply_modifications(to_add, to_remove);

        reference
    }

    fn apply_modifications(&mut self, to_add: Vec<Box<RefCell<dyn WidgetTypeTrait>>>, to_remove: Vec<WidgetReference>) {
        for r in to_remove {
            self.remove(r);
        }
        if !to_add.is_empty() {
            let i0 = self.widgets.len();
            for w in to_add {
                self.widgets.push(Some(w));
            }
            let mut mods = RootWrapper::new(self);
            for w in &self.widgets[i0..] {
                w.as_ref().unwrap().borrow_mut().mount(&mut mods);
            }
            // Recurse to add children of children
            let RootWrapper { to_add, to_remove, .. } = mods;
            self.apply_modifications(to_add, to_remove);
        }
    }

    pub fn update(&mut self) {
        puffin::profile_function!();
        let mut mods = RootWrapper::new(self);
        for widget in self.widgets.iter().flatten() {
            widget.borrow_mut().update(&mut mods);
        }

        let RootWrapper { to_add, to_remove, .. } = mods;
        self.apply_modifications(to_add, to_remove);
    }

    pub fn render(&mut self, ui_document: &mut Document, view: &CanvasView) {
        puffin::profile_function!();
        let mut mods = RootWrapper::new(self);
        for widget in self.widgets.iter().flatten() {
            widget.borrow_mut().render(&mut mods, ui_document, view);
        }

        let RootWrapper { to_add, to_remove, .. } = mods;
        self.apply_modifications(to_add, to_remove);
    }

    pub fn input(&mut self, _ui_document: &mut Document, input: &mut InputManager) {
        puffin::profile_function!();
        let mut mods = RootWrapper::new(self);
        for widget in self.widgets.iter().flatten() {
            widget.borrow_mut().input(&mut mods, input);
        }

        let RootWrapper { to_add, to_remove, .. } = mods;
        self.apply_modifications(to_add, to_remove);
    }
}

fn into_wrapper<T: WidgetTrait>(value: T, index: u32) -> Box<RefCell<dyn WidgetTypeTrait>> {
    Box::new(RefCell::new(WidgetWrapper {
        widget: value,
        listeners: Listeners { listeners: vec![] },
        index,
    }))
}

trait WidgetTypeTrait: Any {
    fn mount(&mut self, root: &mut RootWrapper);
    fn update(&mut self, root: &mut RootWrapper);
    fn input(&mut self, root: &mut RootWrapper, input: &mut InputManager);
    fn render(&mut self, root: &mut RootWrapper, ui_document: &mut Document, view: &CanvasView);
    fn borrow_inner_mut(&mut self) -> &mut dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

pub trait WidgetTrait: Any {
    type EventType;
    fn mount(&mut self, _context: &mut WidgetContext<Self>) {}
    fn update(&mut self, _context: &mut WidgetContext<Self>) {}
    fn render(&mut self, _ui_document: &mut Document, _view: &CanvasView) {}
    fn input(&mut self, _context: &mut WidgetContext<Self>, _input: &mut InputManager) {}
}
