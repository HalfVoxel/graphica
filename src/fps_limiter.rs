use std::time::{Duration, Instant};

pub struct FPSLimiter {
    last_tick: Instant,
}

impl FPSLimiter {
    pub fn wait(&mut self, desired_delta_time: Duration) {
        // puffin::profile_function!();
        let wait_until = self.last_tick + desired_delta_time;
        if let Some(sleep_duration) = wait_until.checked_duration_since(Instant::now()) {
            std::thread::sleep(sleep_duration);
        }
        self.last_tick = Instant::now();
    }
}

impl Default for FPSLimiter {
    fn default() -> Self {
        FPSLimiter {
            last_tick: Instant::now() - Duration::from_secs(100000),
        }
    }
}
