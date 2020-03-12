use crate::something::PathEditor;
use crate::path_collection::PathReference;

struct Toolbar {
    ui: PathReference,
    tools: Vec<Box<dyn Tool>>
}

trait Tool {
    fn activate();
    fn deactivate();
    fn update_ui();
}

impl Toolbar {
    fn new(scene: &mut PathEditor) -> Toolbar {
        Toolbar {
            ui: scene.paths.push(PathData::new())
            tools: vec![]
        }
    }

    fn update_ui(&mut self, scene: &mut PathEditor) {
        let path = scene.paths.resolve_path_mut(self.ui);
        path.clear();
        
    }
}