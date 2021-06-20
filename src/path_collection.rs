use crate::geometry_utilities::types::*;
use crate::path::*;

#[derive(Clone, Hash, Eq, PartialEq)]
pub enum SelectionReference {
    VertexReference(VertexReference),
    ControlPointReference(ControlPointReference),
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct PathReference {
    // path: ByAddress<Rc<RefCell<PathData>>>,
    path_index: u32,
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct VertexReference {
    // path: ByAddress<Rc<RefCell<PathData>>>,
    path_index: u32,
    vertex_index: u32,
}

#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct ControlPointReference {
    // path: ByAddress<Rc<RefCell<PathData>>>,
    path_index: u32,
    vertex_index: u32,
}

impl From<VertexReference> for SelectionReference {
    fn from(v: VertexReference) -> Self {
        SelectionReference::VertexReference(v)
    }
}

impl From<ControlPointReference> for SelectionReference {
    fn from(v: ControlPointReference) -> Self {
        SelectionReference::ControlPointReference(v)
    }
}

impl SelectionReference {
    pub fn position(&self, paths: &PathCollection) -> CanvasPoint {
        match self {
            SelectionReference::VertexReference(x) => paths.resolve(x).position(),
            SelectionReference::ControlPointReference(x) => paths.resolve(x).position(),
        }
    }
}

// impl PartialEq for VertexReference {
//     fn eq(&self, other: &VertexReference) -> bool {
//         self.vertex_index == other.vertex_index && Rc::ptr_eq(&self.path, &other.path)
//     }
// }
// impl Eq for VertexReference {}

pub struct PathCollection {
    pub(crate) paths: Vec<PathData>,
}

impl PathCollection {
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn push(&mut self, mut item: PathData) -> PathReference {
        item.path_index = self.len() as u32;
        self.paths.push(item);
        self.as_path_ref(self.paths.last().unwrap())
    }

    pub fn iter(&self) -> impl Iterator<Item = &PathData> {
        self.paths.iter()
    }

    pub fn last_mut(&mut self) -> Option<&mut PathData> {
        if let Some(p) = self.paths.last_mut() {
            p.dirty();
            Some(p)
        } else {
            None
        }
    }
}

pub trait ReferenceResolver<'a, A, B> {
    fn resolve(&'a self, reference: &A) -> B;
    fn as_ref(&self, point: &B) -> A;
}

pub trait MutableReferenceResolver<'a, A, B> {
    fn resolve_mut(&'a mut self, reference: &A) -> B;
}

impl<'a> ReferenceResolver<'a, VertexReference, ImmutablePathPoint<'a>> for PathCollection {
    fn resolve(&'a self, reference: &VertexReference) -> ImmutablePathPoint<'a> {
        self.paths[reference.path_index as usize].point(reference.vertex_index as i32)
    }
    fn as_ref(&self, point: &ImmutablePathPoint<'a>) -> VertexReference {
        VertexReference {
            path_index: point.data.path_index,
            vertex_index: point.index as u32,
        }
    }
}

impl<'a> MutableReferenceResolver<'a, VertexReference, MutablePathPoint<'a>> for PathCollection {
    fn resolve_mut(&'a mut self, reference: &VertexReference) -> MutablePathPoint<'a> {
        self.paths[reference.path_index as usize].dirty();
        self.paths[reference.path_index as usize].point_mut(reference.vertex_index as i32)
    }
}

impl<'a> ReferenceResolver<'a, ControlPointReference, ImmutableControlPoint<'a>> for PathCollection {
    fn resolve(&'a self, reference: &ControlPointReference) -> ImmutableControlPoint<'a> {
        self.paths[reference.path_index as usize].control_point(reference.vertex_index as i32)
    }
    fn as_ref(&self, point: &ImmutableControlPoint<'a>) -> ControlPointReference {
        ControlPointReference {
            path_index: point.data.path_index,
            vertex_index: point.index as u32,
        }
    }
}

impl<'a> MutableReferenceResolver<'a, ControlPointReference, MutableControlPoint<'a>> for PathCollection {
    fn resolve_mut(&'a mut self, reference: &ControlPointReference) -> MutableControlPoint<'a> {
        self.paths[reference.path_index as usize].dirty();
        self.paths[reference.path_index as usize].control_point_mut(reference.vertex_index as i32)
    }
}

impl<'a> PathCollection {
    pub fn resolve_path(&'a self, reference: &PathReference) -> &'a PathData {
        &self.paths[reference.path_index as usize]
    }

    pub fn resolve_path_mut(&'a mut self, reference: &PathReference) -> &'a mut PathData {
        self.paths[reference.path_index as usize].dirty();
        &mut self.paths[reference.path_index as usize]
    }

    pub fn as_path_ref(&self, path: &PathData) -> PathReference {
        PathReference {
            path_index: path.path_index,
        }
    }
}

impl VertexReference {
    pub fn new(path_index: u32, vertex_index: u32) -> VertexReference {
        VertexReference {
            path_index,
            vertex_index,
        }
    }

    pub fn control_before(&self, path_collection: &PathCollection) -> ControlPointReference {
        // Pretty slow
        let vertex = path_collection.resolve(self);
        let k = ControlPointReference {
            path_index: self.path_index,
            vertex_index: control_before_index(vertex.data, vertex.sub_path, vertex.index) as u32,
        };
        path_collection.resolve(&k).position();
        k
    }

    pub fn control_after(&self, path_collection: &PathCollection) -> ControlPointReference {
        // Pretty slow
        let vertex = path_collection.resolve(self);
        ControlPointReference {
            path_index: self.path_index,
            vertex_index: control_after_index(vertex.index) as u32,
        }
    }

    pub fn prev(&self, path_collection: &PathCollection) -> Option<VertexReference> {
        // Pretty slow
        path_collection.resolve(self).prev().map(|p| path_collection.as_ref(&p))

        // if let Some(prev) = path_collection.resolve(self).prev() {
        //     Some(VertexReference {
        //         path_index: self.path_index,
        //         vertex_index: prev.index() as u32,
        //     })
        // } else {
        //     None
        // }
    }

    pub fn next(&self, path_collection: &PathCollection) -> Option<VertexReference> {
        path_collection.resolve(self).next().map(|p| path_collection.as_ref(&p))
    }
}

impl ControlPointReference {
    pub fn vertex(&self, paths: &PathCollection) -> VertexReference {
        VertexReference {
            path_index: self.path_index,
            vertex_index: paths.resolve(self).vertex().index as u32,
        }
    }

    pub fn opposite_control(&self, paths: &PathCollection) -> ControlPointReference {
        // Pretty slow
        let v = self.vertex(paths);
        if *self == v.control_after(paths) {
            v.control_before(paths)
        } else {
            v.control_after(paths)
        }
    }
}
