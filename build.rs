use anyhow::*;
use glob::glob;
use shaderc::{OptimizationLevel, ResolvedInclude};
use std::path::PathBuf;
use std::{
    fs::{read_to_string, write},
    time::Instant,
};

struct ShaderData {
    src: String,
    src_path: PathBuf,
    spv_path: PathBuf,
    kind: shaderc::ShaderKind,
}

impl ShaderData {
    pub fn load(src_path: PathBuf) -> Result<Self> {
        let extension = src_path
            .extension()
            .context("File has no extension")?
            .to_str()
            .context("Extension cannot be converted to &str")?;
        let kind = match extension {
            "vert" => shaderc::ShaderKind::Vertex,
            "frag" => shaderc::ShaderKind::Fragment,
            "comp" => shaderc::ShaderKind::Compute,
            _ => bail!("Unsupported shader: {}", src_path.display()),
        };

        let src = read_to_string(src_path.clone())?;
        let spv_path = src_path.with_extension(format!("{}.spv", extension));

        Ok(Self {
            src,
            src_path,
            spv_path,
            kind,
        })
    }
}

fn main() -> Result<()> {
    let t0 = Instant::now();
    // Collect all shaders recursively within /shaders/
    let mut shader_paths = [
        glob("./shaders/**/*.vert")?,
        glob("./shaders/**/*.frag")?,
        glob("./shaders/**/*.comp")?,
    ];

    // This could be parallelized
    let shaders = shader_paths
        .iter_mut()
        .flatten()
        .map(|glob_result| ShaderData::load(glob_result?))
        .collect::<Vec<Result<_>>>()
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    let mut compiler = shaderc::Compiler::new().context("Unable to create shader compiler")?;
    let shader_dir = std::fs::canonicalize(PathBuf::from("./shaders"))?;

    let mut options = shaderc::CompileOptions::new().expect("Unsable to create shaderc options");
    options.set_include_callback(|include_path, include_type, source_path, _number| {
        let mut abs_include_path = match include_type {
            shaderc::IncludeType::Relative => {
                let mut x = PathBuf::from(source_path);
                x.pop();
                x.push(PathBuf::from(include_path));
                x
            }
            shaderc::IncludeType::Standard => PathBuf::from(include_path),
        };

        abs_include_path = std::fs::canonicalize(abs_include_path).map_err(|e| format!("Invalid path: {}", e))?;

        if !abs_include_path.extension().map(|s| s == "glsl").unwrap_or(false) {
            return Err(format!("not a valid extension for include {}", include_path));
        }

        if !abs_include_path.starts_with(&shader_dir) {
            Err(format!(
                "Include path '{:?}' is not in the shader directory",
                abs_include_path
            ))
        } else {
            Ok(ResolvedInclude {
                resolved_name: abs_include_path.to_str().unwrap().to_string(),
                content: std::fs::read_to_string(abs_include_path).map_err(|e| e.to_string())?,
            })
        }
    });
    options.set_generate_debug_info();
    options.set_optimization_level(OptimizationLevel::Performance);
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_1 as u32);
    options.set_target_spirv(shaderc::SpirvVersion::V1_3);

    // This can't be parallelized. The [shaderc::Compiler] is not
    // thread safe. Also, it creates a lot of resources. You could
    // spawn multiple processes to handle this, but it would probably
    // be better just to only compile shaders that have been changed
    // recently.
    for shader in shaders {
        // This tells cargo to rerun this script if something in /src/ changes.
        println!(
            "cargo:rerun-if-changed={}",
            shader.src_path.as_os_str().to_str().unwrap()
        );

        let compiled = compiler.compile_into_spirv(
            &shader.src,
            shader.kind,
            shader.src_path.to_str().unwrap(),
            "main",
            Some(&options),
        )?;
        write(shader.spv_path, compiled.as_binary_u8())?;
    }

    println!("Time to recompile shaders: {:.1}s", t0.elapsed().as_secs_f32());

    Ok(())
}
