use wgpu_profiler::{GpuProfiler, GpuTimerScopeResult};

pub fn gpu_profiler_to_puffin(results: &[GpuTimerScopeResult], start_time_nanos: i64) -> puffin::Stream {
    fn inner(results: &[GpuTimerScopeResult], time_offset: i64, stream: &mut puffin::Stream) {
        for scope in results {
            let start_offset = stream.begin_scope(
                (scope.time.start * 1_000_000_000.0) as i64 + time_offset,
                &scope.label,
                "",
                "",
            );
            inner(&scope.nested_scopes, time_offset, stream);
            stream.end_scope(start_offset, (scope.time.end * 1_000_000_000.0) as i64 + time_offset);
        }
    }

    let mut stream = puffin::Stream::default();
    if !results.is_empty() {
        let time_offset = start_time_nanos - (results[0].time.start * 1_000_000_000.0) as i64;
        inner(results, time_offset, &mut stream);
    }
    stream
}

pub fn process_finished_frame(gpu_profiler: &mut GpuProfiler, queue_submit_ns: i64) {
    // Query for oldest finished frame (this is almost certainly not the one we just submitted!) and display results in the command line.
    if let Some(profiling_data) = gpu_profiler.process_finished_frame() {
        let stream = gpu_profiler_to_puffin(&profiling_data, queue_submit_ns);
        puffin::GlobalProfiler::lock().report(
            puffin::ThreadInfo {
                start_time_ns: None,
                name: "gpu".to_string(),
            },
            puffin::StreamInfo::parse(stream).unwrap(),
        )
    }
}
