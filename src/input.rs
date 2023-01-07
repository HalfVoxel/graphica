use crate::geometry_utilities::types::ScreenPoint;
use crate::geometry_utilities::types::ScreenVector;
use euclid::default::Vector2D;
use euclid::point2 as point;
use euclid::vec2 as vector;
use std::collections::HashMap;
use std::time::Instant;
use winit::event::{ElementState, Event, KeyboardInput, WindowEvent};

pub use winit::event::{MouseButton, VirtualKeyCode};

struct KeyState {
    last_down_time: Option<Instant>,
    down_frame: i32,
    up_frame: i32,
    captured: bool,
}

impl KeyState {
    fn is_pressed(&self) -> bool {
        self.down_frame > self.up_frame
    }
}

pub struct CapturedClick {
    pub key: ExtendedKey,
    pub down_frame: i32,
    pub mouse_start: ScreenPoint,
}

impl CapturedClick {
    pub fn is_pressed(&self, input: &InputManager) -> bool {
        if let Some(btn_state) = input.states.get(&self.key) {
            btn_state.down_frame == self.down_frame && btn_state.is_pressed()
        } else {
            false
        }
    }

    pub fn on_down(&self, input: &InputManager) -> bool {
        if let Some(btn_state) = input.states.get(&self.key) {
            btn_state.down_frame == self.down_frame && btn_state.down_frame == input.frame_count
        } else {
            false
        }
    }

    pub fn on_up(&self, input: &InputManager) -> bool {
        if let Some(btn_state) = input.states.get(&self.key) {
            btn_state.down_frame == self.down_frame && btn_state.up_frame == input.frame_count
        } else {
            false
        }
    }
}

pub struct CapturedDrag {
    click: CapturedClick,
}

impl CapturedDrag {
    pub fn uncaptured() -> Self {
        Self {
            click: CapturedClick {
                key: ExtendedKey::MouseButton(MouseButton::Left), // Irrelevant
                down_frame: -1,
                mouse_start: point(0.0, 0.0),
            },
        }
    }

    pub fn is_captured(&self, input: &InputManager) -> bool {
        self.click.is_pressed(input)
    }

    pub fn try_recapture(&mut self, input: &mut InputManager, btn: MouseButton) {
        if !self.click.is_pressed(input) {
            if let Some(capture) = input.capture_drag(btn) {
                *self = capture;
            }
        }
    }

    pub fn new(click: CapturedClick) -> Self {
        Self { click }
    }

    pub fn drag_delta(&self, input: &InputManager) -> ScreenVector {
        if self.click.is_pressed(input) {
            input.mouse_position_delta()
        } else {
            vector(0.0, 0.0)
        }
    }
}

#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum ExtendedKey {
    VirtualKeyCode(VirtualKeyCode),
    MouseButton(MouseButton),
}

impl From<VirtualKeyCode> for ExtendedKey {
    fn from(key: VirtualKeyCode) -> Self {
        ExtendedKey::VirtualKeyCode(key)
    }
}

impl From<MouseButton> for ExtendedKey {
    fn from(key: MouseButton) -> Self {
        ExtendedKey::MouseButton(key)
    }
}

pub struct BatchedMouseCapture<T> {
    point: ScreenPoint,
    best: Option<T>,
    best_score: f32,
    mouse_btn: MouseButton,
}

impl<T> BatchedMouseCapture<T> {
    pub fn add(&mut self, shape: impl Shape, value: T) {
        let score = shape.score(self.point);
        if score > self.best_score {
            self.best = Some(value);
            self.best_score = score;
        }
    }

    pub fn capture(self, input: &mut InputManager) -> Option<(CapturedClick, T)> {
        if let Some(best) = self.best {
            input.capture_click(self.mouse_btn).map(|capture| (capture, best))
        } else {
            None
        }
    }
}

pub trait Shape {
    fn score(&self, point: ScreenPoint) -> f32;
}

pub struct CircleShape {
    pub center: ScreenPoint,
    pub radius: f32,
}

impl Shape for CircleShape {
    fn score(&self, point: ScreenPoint) -> f32 {
        let dist = (point - self.center).square_length();
        if dist < self.radius * self.radius {
            1.0 / self.radius
        } else {
            0.0
        }
    }
}

#[derive(Default)]
pub struct InputManager {
    states: HashMap<ExtendedKey, KeyState>,
    pub mouse_position: ScreenPoint,
    last_mouse_position: ScreenPoint,
    pub frame_count: i32,
    pub scroll_delta: Vector2D<f32>,
}

#[derive(Default)]
pub struct KeyCombination {
    keys: Vec<ExtendedKey>,
}

impl KeyCombination {
    pub fn new() -> KeyCombination {
        KeyCombination { keys: vec![] }
    }

    pub fn and(mut self, key: impl Into<ExtendedKey>) -> Self {
        self.keys.push(key.into());
        self
    }
}

impl InputManager {
    pub fn mouse_position_delta(&self) -> ScreenVector {
        if self.frame_count > 0 {
            self.mouse_position - self.last_mouse_position
        } else {
            vector(0.0, 0.0)
        }
    }

    pub fn tick_frame(&mut self) {
        self.frame_count += 1;
        self.scroll_delta = vector(0.0, 0.0);
        self.last_mouse_position = self.mouse_position;
        for state in self.states.values_mut() {
            if !state.is_pressed() {
                state.captured = false;
            }
        }
    }

    pub fn block_egui_captured_input(&mut self, ctx: &egui::Context) {
        if ctx.wants_pointer_input() {
            self.scroll_delta = vector(0.0, 0.0);
            self.capture_click(MouseButton::Left);
            self.capture_click(MouseButton::Right);
            self.capture_click(MouseButton::Middle);
        }
        if ctx.wants_keyboard_input() {
            for state in self.states.values_mut() {
                if state.is_pressed() {
                    state.captured = true;
                }
            }
        }
    }

    pub fn event(&mut self, event: &Event<()>) {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CursorMoved { position, .. } => {
                    self.mouse_position = point(position.x as f32, position.y as f32);
                }
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state,
                            virtual_keycode: Some(key),
                            ..
                        },
                    ..
                } => {
                    self.event_key(*state, ExtendedKey::VirtualKeyCode(*key));
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    self.event_key(*state, ExtendedKey::MouseButton(*button));
                }
                _ => {}
            },
            Event::DeviceEvent {
                event: winit::event::DeviceEvent::Motion { axis, value },
                ..
            } => {
                match axis {
                    // TODO Correct axis for hoizontal scroll?
                    2 => {
                        self.scroll_delta.x += *value as f32;
                    }
                    3 => {
                        self.scroll_delta.y += *value as f32;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn event_key(&mut self, state: ElementState, key: ExtendedKey) {
        self.states.entry(key).or_insert(KeyState {
            last_down_time: None,
            down_frame: -1,
            up_frame: -1,
            captured: false,
        });

        let mut btn_state = self.states.get_mut(&key).unwrap();
        if state == ElementState::Released && btn_state.is_pressed() {
            btn_state.up_frame = self.frame_count;
        }
        if state == ElementState::Pressed && !btn_state.is_pressed() {
            btn_state.down_frame = self.frame_count;
            btn_state.last_down_time = Some(Instant::now());
        }
    }

    pub fn is_pressed(&self, key: impl Into<ExtendedKey>) -> bool {
        let key = key.into();
        self.states.get(&key).map(KeyState::is_pressed).unwrap_or(false)
    }

    pub fn on_down(&self, key: impl Into<ExtendedKey>) -> bool {
        let key = key.into();
        self.states
            .get(&key)
            .map(|x| x.down_frame == self.frame_count)
            .unwrap_or(false)
    }

    pub fn on_up(&self, key: impl Into<ExtendedKey>) -> bool {
        let key = key.into();
        self.states
            .get(&key)
            .map(|x| x.up_frame == self.frame_count)
            .unwrap_or(false)
    }

    pub fn on_click(&self, key: impl Into<ExtendedKey>) -> bool {
        self.on_up(key)
    }

    fn is_modifier(key: &VirtualKeyCode) -> bool {
        use VirtualKeyCode::*;
        matches!(key, LControl | RControl | LAlt | RAlt | LShift | RShift | LWin | RWin)
    }

    fn priority(key: &ExtendedKey) -> i32 {
        match key {
            ExtendedKey::MouseButton { .. } => 2,
            ExtendedKey::VirtualKeyCode(key) if Self::is_modifier(key) => 0,
            ExtendedKey::VirtualKeyCode(_) => 1,
        }
    }

    pub fn on_combination(&self, combination: &KeyCombination) -> bool {
        // Item triggers iff all other items are pressed and the item is among the ones with the highest priority
        // Highest priority of all keys in the combination
        let highest_priority = combination.keys.iter().map(Self::priority).max().unwrap();
        // All keys in the combination must be pressed
        if combination.keys.iter().all(|v| self.is_pressed(*v) || self.on_up(*v)) {
            // At least one of the highest priority keys must have been triggered this frame
            combination
                .keys
                .iter()
                .filter(|v| Self::priority(v) == highest_priority)
                .any(|v| {
                    // Mouse keys trigger on click (key up) while other keys trigger on key down
                    if let ExtendedKey::MouseButton(_) = v {
                        self.on_click(*v)
                    } else {
                        self.on_down(*v)
                    }
                })
        } else {
            false
        }
    }

    pub fn capture_click_batch<T>(&mut self, mouse_btn: MouseButton) -> Option<BatchedMouseCapture<T>> {
        //let btn_state = self.states.get_mut(&mouse_btn).unwrap();
        if let Some(btn_state) = self.states.get_mut(&mouse_btn.into()) {
            if btn_state.down_frame == self.frame_count && !btn_state.captured {
                return Some(BatchedMouseCapture {
                    point: self.mouse_position,
                    best: None,
                    best_score: 0.0,
                    mouse_btn,
                });
            }
        }
        None
    }

    pub fn capture_click(&mut self, mouse_btn: MouseButton) -> Option<CapturedClick> {
        let key = mouse_btn.into();
        if let Some(btn_state) = self.states.get_mut(&key) {
            if btn_state.down_frame == self.frame_count && !btn_state.captured {
                btn_state.captured = true;
                return Some(CapturedClick {
                    key,
                    down_frame: self.frame_count,
                    mouse_start: self.mouse_position,
                });
            }
        }
        None
    }

    pub fn capture_drag(&mut self, mouse_btn: MouseButton) -> Option<CapturedDrag> {
        self.capture_click(mouse_btn).map(CapturedDrag::new)
    }

    pub fn capture_click_shape(&mut self, mouse_btn: MouseButton, shape: impl Shape) -> Option<CapturedClick> {
        if shape.score(self.mouse_position) > 0.0 {
            self.capture_click(mouse_btn)
        } else {
            None
        }
    }
}
