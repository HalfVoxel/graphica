use std::time::{Duration, Instant};

pub struct FPSLimiter {
    last_tick: Instant,
}

impl FPSLimiter {
    pub fn new() -> FPSLimiter {
        FPSLimiter {
            last_tick: Instant::now() - Duration::from_secs(100000),
        }
    }

    pub fn wait(&mut self, desired_delta_time: Duration) {
        let wait_until = self.last_tick + desired_delta_time;
        if let Some(sleep_duration) = wait_until.checked_duration_since(Instant::now()) {
            std::thread::sleep(sleep_duration);
        }
        self.last_tick = Instant::now();
    }
}
