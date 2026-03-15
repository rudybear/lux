//! CLI entry point for the Lux shader playground.
//!
//! Supports scene/pipeline architecture with auto-detection of render paths.
//! Can run headless (offscreen render to PNG) or interactively (winit window).

mod camera;
mod deferred_renderer;
pub mod gltf_loader;
mod mesh_renderer;
mod meshlet;
mod raster_renderer;
pub mod reflected_pipeline;
mod rt_renderer;
mod scene;
pub mod scene_manager;
mod splat_renderer;
mod screenshot;
mod spv_loader;
mod vulkan_context;

use clap::Parser;
use log::{error, info};
use std::path::Path;

/// Lux shader playground — render SPIR-V shaders using Vulkan.
#[derive(Parser)]
#[command(name = "lux-playground", about = "Lux shader playground")]
struct Args {
    /// Scene source: sphere, fullscreen, triangle, or path to .glb/.gltf
    #[arg(long, required_unless_present = "shader")]
    scene: Option<String>,

    /// Pipeline shader base path (auto-resolved from scene if not given)
    #[arg(long)]
    pipeline: Option<String>,

    /// [DEPRECATED: use --scene/--pipeline] Base path to .spv shader files
    #[arg(index = 1)]
    shader: Option<String>,

    /// [DEPRECATED: use --scene instead]
    #[arg(long, value_parser = ["triangle", "fullscreen", "pbr", "rt", "mesh", "deferred"])]
    mode: Option<String>,

    /// Output image width in pixels.
    #[arg(long, default_value = "512")]
    width: u32,

    /// Output image height in pixels.
    #[arg(long, default_value = "512")]
    height: u32,

    /// Output PNG file path.
    #[arg(long, default_value = "output.png")]
    output: String,

    /// Open an interactive preview window instead of headless rendering.
    #[arg(long)]
    interactive: bool,

    /// Force headless (offscreen) rendering. This is the default.
    #[arg(long)]
    headless: bool,

    /// IBL environment name (default: auto-detect pisa/neutral from assets/ibl/).
    #[arg(long)]
    ibl: Option<String>,

    /// Enable Vulkan validation layers even in release builds.
    #[arg(long)]
    validation: bool,

    /// Add 3 demo lights (directional + point + spot) with shadows.
    #[arg(long)]
    demo_lights: bool,

    /// Sponza courtyard lights (sun + orbiting torch + accent).
    #[arg(long)]
    sponza_lights: bool,

    /// Splat shader base for hybrid RT/mesh+splat rendering.
    #[arg(long)]
    splat_pipeline: Option<String>,
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    if let Err(e) = run(args) {
        error!("Error: {}", e);
        std::process::exit(1);
    }
}

/// Resolve a default pipeline shader base path from the scene source.
fn resolve_default_pipeline(scene: &str) -> Result<String, String> {
    if scene.ends_with(".glb") || scene.ends_with(".gltf") {
        // Try shadercache/gltf_pbr first (where compiled .spv files live),
        // fall back to examples/gltf_pbr
        if Path::new("shadercache/gltf_pbr.vert.spv").exists() {
            Ok("shadercache/gltf_pbr".to_string())
        } else {
            Ok("examples/gltf_pbr".to_string())
        }
    } else if scene == "fullscreen" {
        Err("--pipeline required for fullscreen scenes".to_string())
    } else if scene == "triangle" {
        Ok("shadercache/hello_triangle".to_string())
    } else {
        Ok("shadercache/pbr_basic".to_string())
    }
}

/// Auto-detect the render path from which .spv files exist for a pipeline base.
/// Prefers raster over RT when both exist (raster supports per-material draw calls).
/// Pass force_mode="rt" to override and use RT path.
/// Returns "splat" when KHR_gaussian_splatting data is detected in the scene
/// (caller must check scene data and override the result).
fn detect_render_path(pipeline_base: &str, force_mode: &str) -> &'static str {
    if force_mode == "splat" {
        return "splat";
    }
    if force_mode == "deferred" {
        return "deferred";
    }
    if force_mode == "rt" && Path::new(&format!("{}.rgen.spv", pipeline_base)).exists() {
        return "rt";
    }
    if force_mode == "mesh"
        && Path::new(&format!("{}.mesh.spv", pipeline_base)).exists()
        && Path::new(&format!("{}.frag.spv", pipeline_base)).exists()
    {
        return "mesh";
    }
    // Check for deferred rendering (gbuf.vert + gbuf.frag + light.vert + light.frag)
    if Path::new(&format!("{}.gbuf.vert.spv", pipeline_base)).exists()
        && Path::new(&format!("{}.light.frag.spv", pipeline_base)).exists()
    {
        return "deferred";
    }
    // Check for gaussian splat compute shader (comp + vert + frag)
    if Path::new(&format!("{}.comp.spv", pipeline_base)).exists()
        && Path::new(&format!("{}.vert.spv", pipeline_base)).exists()
        && Path::new(&format!("{}.frag.spv", pipeline_base)).exists()
    {
        // comp + vert + frag = gaussian splat pipeline
        return "splat";
    }
    // Prefer raster over RT when both exist (raster supports per-material draw calls)
    if Path::new(&format!("{}.vert.spv", pipeline_base)).exists()
        && Path::new(&format!("{}.frag.spv", pipeline_base)).exists()
    {
        "raster"
    } else if Path::new(&format!("{}.mesh.spv", pipeline_base)).exists()
        && Path::new(&format!("{}.frag.spv", pipeline_base)).exists()
    {
        "mesh"
    } else if Path::new(&format!("{}.rgen.spv", pipeline_base)).exists() {
        "rt"
    } else if Path::new(&format!("{}.frag.spv", pipeline_base)).exists() {
        "fullscreen"
    } else {
        panic!(
            "No shader files found for pipeline base: {}",
            pipeline_base
        );
    }
}

fn run(args: Args) -> Result<(), String> {
    // Handle backward compatibility: old --mode maps to new --scene
    let scene_source = if let Some(scene) = &args.scene {
        scene.clone()
    } else if let Some(mode) = &args.mode {
        match mode.as_str() {
            "triangle" => "triangle".to_string(),
            "fullscreen" => "fullscreen".to_string(),
            "pbr" => "sphere".to_string(),
            "rt" => "sphere".to_string(),
            _ => "sphere".to_string(),
        }
    } else {
        "sphere".to_string() // default
    };

    let mut pipeline_base = if let Some(p) = &args.pipeline {
        p.clone()
    } else if let Some(shader) = &args.shader {
        shader.clone()
    } else {
        resolve_default_pipeline(&scene_source).unwrap_or_else(|e| panic!("{}", e))
    };

    let force_mode = match args.mode.as_deref() {
        Some("rt") => "rt",
        Some("mesh") => "mesh",
        Some("deferred") => "deferred",
        _ => "",
    };
    let mut render_path = detect_render_path(&pipeline_base, force_mode);

    // Pre-scan glTF files for KHR_gaussian_splatting data to override render path
    // and auto-select the correct splat shader based on SH degree.
    if (scene_source.ends_with(".glb") || scene_source.ends_with(".gltf"))
        && (force_mode.is_empty() || force_mode == "splat")
    {
        if let Ok(scene) = gltf_loader::load_gltf(Path::new(&scene_source)) {
            if scene.splat_data.has_splats {
                info!("Detected KHR_gaussian_splatting in scene, switching to splat render path");
                render_path = "splat";

                // Auto-select splat pipeline based on SH degree (if user didn't specify)
                if args.pipeline.is_none() && args.shader.is_none() {
                    let sh_deg = scene.splat_data.sh_degree;
                    // Try shadercache first, then examples
                    let candidates = [
                        format!("shadercache/gaussian_splat_sh{}", sh_deg),
                        format!("examples/gaussian_splat_sh{}", sh_deg),
                    ];
                    for candidate in &candidates {
                        if Path::new(&format!("{}.comp.spv", candidate)).exists() {
                            info!("Auto-selected splat pipeline: {} (SH degree {})", candidate, sh_deg);
                            pipeline_base = candidate.clone();
                            break;
                        }
                    }
                }
            }
        }
    }

    info!("Scene: {}", scene_source);
    info!("Pipeline: {}", pipeline_base);
    info!("Render path: {}", render_path);
    info!("Resolution: {}x{}", args.width, args.height);
    info!("Output: {}", args.output);

    let ibl_name = args.ibl.as_deref().unwrap_or("");

    // Auto-detect Sponza scene by filename
    let mut sponza_lights = args.sponza_lights;
    if !sponza_lights && !args.demo_lights {
        if scene_source.to_lowercase().contains("ponza") {
            sponza_lights = true;
            info!("Auto-detected Sponza scene, enabling sponza lights");
        }
    }

    if args.interactive && !args.headless {
        // Use larger window for interactive mode if user didn't specify
        let (iw, ih) = if args.width == 512 && args.height == 512 {
            (1024, 768)
        } else {
            (args.width, args.height)
        };
        run_interactive(&pipeline_base, &scene_source, render_path, iw, ih, ibl_name, args.validation, args.demo_lights, sponza_lights)?;
    } else {
        run_headless(&pipeline_base, &scene_source, render_path, args.width, args.height, &args.output, ibl_name, args.validation, args.demo_lights, sponza_lights)?;
    }

    Ok(())
}

fn setup_demo_lights() -> Vec<scene_manager::SceneLight> {
    vec![
        // Light 1: warm directional (sun-like), casts shadow
        scene_manager::SceneLight {
            light_type: 0, // Directional
            position: glam::Vec3::ZERO,
            direction: glam::Vec3::new(0.6, -0.8, 0.4).normalize(),
            color: glam::Vec3::new(1.0, 0.95, 0.85),
            intensity: 1.2,
            range: 0.0,
            inner_cone_angle: 0.0,
            outer_cone_angle: 0.7854,
            casts_shadow: true,
            shadow_index: -1,
        },
        // Light 2: blue point light (left side)
        scene_manager::SceneLight {
            light_type: 1, // Point
            position: glam::Vec3::new(-2.0, 1.0, 1.0),
            direction: glam::Vec3::new(0.0, -1.0, 0.0),
            color: glam::Vec3::new(0.3, 0.5, 1.0),
            intensity: 3.0,
            range: 10.0,
            inner_cone_angle: 0.0,
            outer_cone_angle: 0.7854,
            casts_shadow: false,
            shadow_index: -1,
        },
        // Light 3: red spot light (right side), casts shadow
        scene_manager::SceneLight {
            light_type: 2, // Spot
            position: glam::Vec3::new(2.5, 2.0, 1.5),
            direction: glam::Vec3::new(-1.0, -1.0, -0.5).normalize(),
            color: glam::Vec3::new(1.0, 0.3, 0.2),
            intensity: 5.0,
            range: 15.0,
            inner_cone_angle: 0.2,
            outer_cone_angle: 0.5,
            casts_shadow: true,
            shadow_index: -1,
        },
    ]
}

fn setup_sponza_lights() -> Vec<scene_manager::SceneLight> {
    vec![
        // Sun: warm directional light casting shadows through arches
        scene_manager::SceneLight {
            light_type: 0, // Directional
            position: glam::Vec3::ZERO,
            direction: glam::Vec3::new(0.5, -0.7, 0.3).normalize(),
            color: glam::Vec3::new(1.0, 0.95, 0.85),
            intensity: 5.0,
            range: 0.0,
            inner_cone_angle: 0.0,
            outer_cone_angle: 0.7854,
            casts_shadow: true,
            shadow_index: -1,
        },
        // Torch: orange spot light that orbits inside the courtyard (animated per frame)
        scene_manager::SceneLight {
            light_type: 2, // Spot
            position: glam::Vec3::new(0.0, 600.0, 0.0),
            direction: glam::Vec3::new(0.0, -1.0, 0.0),
            color: glam::Vec3::new(1.0, 0.7, 0.3),
            intensity: 500000.0,
            range: 3000.0,
            inner_cone_angle: 0.3,
            outer_cone_angle: 0.7,
            casts_shadow: true,
            shadow_index: -1,
        },
        // Accent: blue point light (no shadow)
        scene_manager::SceneLight {
            light_type: 1, // Point
            position: glam::Vec3::new(-500.0, 400.0, -300.0),
            direction: glam::Vec3::new(0.0, -1.0, 0.0),
            color: glam::Vec3::new(0.3, 0.5, 1.0),
            intensity: 100000.0,
            range: 2000.0,
            inner_cone_angle: 0.0,
            outer_cone_angle: 0.7854,
            casts_shadow: false,
            shadow_index: -1,
        },
    ]
}

/// Run in headless mode: create context, render offscreen, save PNG.
fn run_headless(
    pipeline_base: &str,
    scene_source: &str,
    render_path: &str,
    width: u32,
    height: u32,
    output: &str,
    ibl_name: &str,
    force_validation: bool,
    demo_lights: bool,
    sponza_lights: bool,
) -> Result<(), String> {
    let enable_rt = render_path == "rt";

    info!("Creating Vulkan context (RT: {})...", enable_rt);
    let mut ctx = vulkan_context::VulkanContext::new(enable_rt, force_validation)?;

    let output_path = Path::new(output);

    let result = match render_path {
        "raster" => {
            info!("Rendering raster scene '{}' with pipeline '{}'...", scene_source, pipeline_base);
            raster_renderer::render_raster(
                &mut ctx,
                pipeline_base,
                scene_source,
                width,
                height,
                output_path,
                ibl_name,
                demo_lights,
                sponza_lights,
            )
        }
        "fullscreen" => {
            info!("Rendering fullscreen pipeline '{}'...", pipeline_base);
            raster_renderer::render_fullscreen(
                &mut ctx,
                pipeline_base,
                width,
                height,
                output_path,
                ibl_name,
            )
        }
        "rt" => {
            if !ctx.supports_rt() {
                return Err(
                    "Ray tracing mode requested but GPU does not support \
                     VK_KHR_ray_tracing_pipeline. Please use a different mode or a GPU \
                     with ray tracing support."
                        .to_string(),
                );
            }
            info!("Rendering ray traced image...");
            rt_renderer::render_rt(&mut ctx, pipeline_base, scene_source, width, height, output_path, ibl_name)
        }
        "mesh" => {
            if !ctx.mesh_shader_supported {
                return Err(
                    "Mesh shader pipeline requires VK_EXT_mesh_shader support. \
                     Please use a different mode or a GPU with mesh shader support."
                        .to_string(),
                );
            }
            info!("Rendering mesh shader scene '{}' with pipeline '{}'...", scene_source, pipeline_base);
            render_mesh_headless(&mut ctx, pipeline_base, scene_source, width, height, output_path, ibl_name, demo_lights)
        }
        "deferred" => {
            info!("Rendering deferred scene '{}' with pipeline '{}'...", scene_source, pipeline_base);
            render_deferred_headless(&mut ctx, pipeline_base, scene_source, width, height, output_path, ibl_name, demo_lights, sponza_lights)
        }
        "splat" => {
            use scene_manager::Renderer;

            info!(
                "Rendering gaussian splats: scene='{}' pipeline='{}'",
                scene_source, pipeline_base
            );

            // Detect hybrid scene: both triangle meshes AND gaussian splats
            let has_raster_meshes = (scene_source.ends_with(".glb") || scene_source.ends_with(".gltf"))
                && {
                    // Quick check: load glTF to see if there are non-POINTS primitives
                    if let Ok((doc, _, _)) = gltf::import(Path::new(scene_source)) {
                        doc.meshes().any(|mesh| {
                            mesh.primitives().any(|prim| prim.mode() != gltf::mesh::Mode::Points)
                        })
                    } else {
                        false
                    }
                };

            let mut splat_renderer = splat_renderer::GaussianSplatRenderer::new(
                &mut ctx, scene_source, pipeline_base, width, height,
            )?;

            let is_hybrid = has_raster_meshes;

            if is_hybrid {
                info!("Hybrid scene detected: raster meshes + gaussian splats");

                // 1. Render raster scene offscreen
                let raster_pipeline = if std::path::Path::new("shadercache/gltf_pbr.vert.spv").exists() {
                    "shadercache/gltf_pbr".to_string()
                } else if std::path::Path::new("examples/gltf_pbr.vert.spv").exists() {
                    "examples/gltf_pbr".to_string()
                } else if std::path::Path::new("../examples/gltf_pbr.vert.spv").exists() {
                    "../examples/gltf_pbr".to_string()
                } else {
                    return Err("Cannot find gltf_pbr shaders for hybrid rendering".to_string());
                };

                let mut raster = raster_renderer::PersistentRenderer::init(
                    &mut ctx,
                    &raster_pipeline,
                    scene_source,
                    width,
                    height,
                    ibl_name,
                    demo_lights,
                    sponza_lights,
                )?;

                // Update raster camera to match splat auto-camera
                if splat_renderer.has_scene_bounds() {
                    let aspect = width as f32 / height as f32;
                    raster.update_camera(
                        splat_renderer.auto_eye(),
                        splat_renderer.auto_target(),
                        splat_renderer.auto_up(),
                        45.0f32.to_radians(),
                        aspect,
                        0.1,
                        splat_renderer.auto_far(),
                    );
                }

                let raster_cmd = raster.render(&ctx)?;
                unsafe {
                    ctx.device.end_command_buffer(raster_cmd)
                        .map_err(|e| format!("Failed to end raster cmd: {:?}", e))?;
                }
                let raster_bufs = [raster_cmd];
                let raster_submit = ash::vk::SubmitInfo::default().command_buffers(&raster_bufs);
                let raster_fence = unsafe {
                    ctx.device.create_fence(&ash::vk::FenceCreateInfo::default(), None)
                        .map_err(|e| format!("Failed to create fence: {:?}", e))?
                };
                unsafe {
                    ctx.device.queue_submit(ctx.graphics_queue, &[raster_submit], raster_fence)
                        .map_err(|e| format!("Failed to submit raster: {:?}", e))?;
                    ctx.device.wait_for_fences(&[raster_fence], true, u64::MAX)
                        .map_err(|e| format!("Failed to wait: {:?}", e))?;
                    ctx.device.destroy_fence(raster_fence, None);
                    ctx.device.free_command_buffers(ctx.command_pool, &raster_bufs);
                }

                // 2. Blit raster color + depth into splat buffers for compositing
                splat_renderer.preload_background(
                    &ctx,
                    raster.output_image(),
                    raster.width(),
                    raster.height(),
                )?;
                splat_renderer.preload_depth(
                    &ctx,
                    raster.depth_image(),
                    raster.width(),
                    raster.height(),
                )?;

                raster.destroy(&mut ctx);
                info!("Hybrid: raster pass complete, compositing splats on top");
            }

            // Render splats (uses LOAD render pass if background was preloaded)
            let cmd = splat_renderer.render(&ctx)?;

            // End command buffer and submit
            unsafe {
                ctx.device
                    .end_command_buffer(cmd)
                    .map_err(|e| format!("Failed to end splat command buffer: {:?}", e))?;
            }

            let cmd_bufs = [cmd];
            let submit_info = ash::vk::SubmitInfo::default().command_buffers(&cmd_bufs);

            let fence_info = ash::vk::FenceCreateInfo::default();
            let fence = unsafe {
                ctx.device
                    .create_fence(&fence_info, None)
                    .map_err(|e| format!("Failed to create fence: {:?}", e))?
            };

            unsafe {
                ctx.device
                    .queue_submit(ctx.graphics_queue, &[submit_info], fence)
                    .map_err(|e| format!("Failed to submit splat command buffer: {:?}", e))?;
                ctx.device
                    .wait_for_fences(&[fence], true, u64::MAX)
                    .map_err(|e| format!("Failed to wait for fence: {:?}", e))?;
                ctx.device.destroy_fence(fence, None);
                ctx.device.free_command_buffers(ctx.command_pool, &cmd_bufs);
            }

            // Read back pixels from offscreen image
            let device_clone = ctx.device.clone();
            let cmd2 = ctx.begin_single_commands()?;

            let mut staging = screenshot::StagingBuffer::new(
                &device_clone,
                ctx.allocator_mut(),
                width,
                height,
            )?;

            screenshot::cmd_copy_image_to_buffer(
                &device_clone,
                cmd2,
                splat_renderer.output_image(),
                staging.buffer,
                width,
                height,
            );

            ctx.end_single_commands(cmd2)?;

            let pixels = staging.read_pixels(width, height)?;
            screenshot::save_png(&pixels, width, height, output_path)?;

            info!(
                "Saved {} render to {:?}",
                if is_hybrid { "hybrid (raster + splat)" } else { "splat" },
                output_path
            );

            staging.destroy(&device_clone, ctx.allocator_mut());
            splat_renderer.destroy(&mut ctx);

            Ok(())
        }
        _ => Err(format!("Unknown render path: {}", render_path)),
    };

    // Wait for GPU idle before cleanup
    unsafe {
        let _ = ctx.device.device_wait_idle();
    }

    ctx.destroy();

    match &result {
        Ok(()) => info!("Render complete: {}", output),
        Err(e) => error!("Render failed: {}", e),
    }

    result
}

/// Render a mesh shader scene headless and save to PNG.
fn render_mesh_headless(
    ctx: &mut vulkan_context::VulkanContext,
    pipeline_base: &str,
    scene_source: &str,
    width: u32,
    height: u32,
    output_path: &Path,
    ibl_name: &str,
    demo_lights: bool,
) -> Result<(), String> {
    use scene_manager::Renderer;

    let mut renderer = mesh_renderer::MeshShaderRenderer::new(
        ctx,
        scene_source,
        pipeline_base,
        width,
        height,
        ibl_name,
        demo_lights,
    )?;

    // Render frame
    let cmd = renderer.render(ctx)?;

    // End command buffer and submit
    unsafe {
        ctx.device
            .end_command_buffer(cmd)
            .map_err(|e| format!("Failed to end mesh command buffer: {:?}", e))?;
    }

    let cmd_bufs = [cmd];
    let submit_info = ash::vk::SubmitInfo::default().command_buffers(&cmd_bufs);

    let fence_info = ash::vk::FenceCreateInfo::default();
    let fence = unsafe {
        ctx.device
            .create_fence(&fence_info, None)
            .map_err(|e| format!("Failed to create fence: {:?}", e))?
    };

    unsafe {
        ctx.device
            .queue_submit(ctx.graphics_queue, &[submit_info], fence)
            .map_err(|e| format!("Failed to submit mesh command buffer: {:?}", e))?;
        ctx.device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| format!("Failed to wait for fence: {:?}", e))?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.free_command_buffers(ctx.command_pool, &cmd_bufs);
    }

    // Read back pixels from offscreen image
    let device_clone = ctx.device.clone();
    let cmd2 = ctx.begin_single_commands()?;

    let mut staging = screenshot::StagingBuffer::new(
        &device_clone,
        ctx.allocator_mut(),
        width,
        height,
    )?;

    screenshot::cmd_copy_image_to_buffer(
        &device_clone,
        cmd2,
        renderer.output_image(),
        staging.buffer,
        width,
        height,
    );

    ctx.end_single_commands(cmd2)?;

    let pixels = staging.read_pixels(width, height)?;
    screenshot::save_png(&pixels, width, height, output_path)?;

    info!("Saved mesh shader render to {:?}", output_path);

    staging.destroy(&device_clone, ctx.allocator_mut());
    renderer.destroy(ctx);

    Ok(())
}

/// Render a deferred scene headless and save to PNG.
fn render_deferred_headless(
    ctx: &mut vulkan_context::VulkanContext,
    pipeline_base: &str,
    scene_source: &str,
    width: u32,
    height: u32,
    output_path: &Path,
    ibl_name: &str,
    demo_lights: bool,
    sponza_lights: bool,
) -> Result<(), String> {
    use scene_manager::Renderer;

    let mut renderer = deferred_renderer::DeferredRenderer::new(
        ctx,
        scene_source,
        pipeline_base,
        width,
        height,
        ibl_name,
        demo_lights,
        sponza_lights,
    )?;

    // Render frame
    let cmd = renderer.render(ctx)?;

    // End command buffer and submit
    unsafe {
        ctx.device
            .end_command_buffer(cmd)
            .map_err(|e| format!("Failed to end deferred command buffer: {:?}", e))?;
    }

    let cmd_bufs = [cmd];
    let submit_info = ash::vk::SubmitInfo::default().command_buffers(&cmd_bufs);

    let fence_info = ash::vk::FenceCreateInfo::default();
    let fence = unsafe {
        ctx.device
            .create_fence(&fence_info, None)
            .map_err(|e| format!("Failed to create fence: {:?}", e))?
    };

    unsafe {
        ctx.device
            .queue_submit(ctx.graphics_queue, &[submit_info], fence)
            .map_err(|e| format!("Failed to submit deferred command buffer: {:?}", e))?;
        ctx.device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| format!("Failed to wait for fence: {:?}", e))?;
        ctx.device.destroy_fence(fence, None);
        ctx.device.free_command_buffers(ctx.command_pool, &cmd_bufs);
    }

    // Read back pixels from offscreen image
    let device_clone = ctx.device.clone();
    let cmd2 = ctx.begin_single_commands()?;

    let mut staging = screenshot::StagingBuffer::new(
        &device_clone,
        ctx.allocator_mut(),
        width,
        height,
    )?;

    screenshot::cmd_copy_image_to_buffer(
        &device_clone,
        cmd2,
        renderer.output_image(),
        staging.buffer,
        width,
        height,
    );

    ctx.end_single_commands(cmd2)?;

    let pixels = staging.read_pixels(width, height)?;
    screenshot::save_png(&pixels, width, height, output_path)?;

    info!("Saved deferred render to {:?}", output_path);

    staging.destroy(&device_clone, ctx.allocator_mut());
    renderer.destroy(ctx);

    Ok(())
}

/// Orbit camera state for interactive viewing.
struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: glam::Vec3,
    up: glam::Vec3,
    fov_y: f32,
    near_plane: f32,
    far_plane: f32,
    dragging: bool,
    last_x: f64,
    last_y: f64,
}

impl OrbitCamera {
    fn new() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.15,
            distance: 3.0,
            target: glam::Vec3::ZERO,
            up: glam::Vec3::new(0.0, 1.0, 0.0),
            fov_y: 45.0f32.to_radians(),
            near_plane: 0.1,
            far_plane: 100.0,
            dragging: false,
            last_x: 0.0,
            last_y: 0.0,
        }
    }

    fn init_from_auto_camera(
        &mut self,
        eye: glam::Vec3,
        target: glam::Vec3,
        up: glam::Vec3,
        far: f32,
    ) {
        self.target = target;
        self.up = up;
        self.far_plane = far;

        let dir = eye - target;
        self.distance = dir.length();
        if self.distance > 1e-6 {
            let d = dir / self.distance;
            self.yaw = d.x.atan2(d.z);
            self.pitch = d.y.asin();
        }
    }

    fn eye(&self) -> glam::Vec3 {
        let x = self.distance * self.pitch.cos() * self.yaw.sin();
        let y = self.distance * self.pitch.sin();
        let z = self.distance * self.pitch.cos() * self.yaw.cos();
        self.target + glam::Vec3::new(x, y, z)
    }
}

/// Run in interactive mode with a winit window and orbit camera.
fn run_interactive(
    pipeline_base: &str,
    scene_source: &str,
    render_path: &str,
    width: u32,
    height: u32,
    ibl_name: &str,
    force_validation: bool,
    demo_lights: bool,
    sponza_lights: bool,
) -> Result<(), String> {
    use ash::vk;
    use scene_manager::Renderer;
    use winit::application::ApplicationHandler;
    use winit::event::{ElementState, MouseButton, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::keyboard::{KeyCode, PhysicalKey};
    use winit::window::{Window, WindowId};

    let use_rt = render_path == "rt";
    let use_mesh = render_path == "mesh";
    let use_splat = render_path == "splat";
    let use_deferred = render_path == "deferred";

    struct App {
        window: Option<Window>,
        pipeline_base: String,
        scene_source: String,
        width: u32,
        height: u32,
        use_rt: bool,
        use_mesh: bool,
        use_splat: bool,
        use_deferred: bool,
        ibl_name: String,
        force_validation: bool,
        demo_lights: bool,
        sponza_lights: bool,
        start_time: std::time::Instant,
        // Vulkan state (initialized after window creation)
        ctx: Option<vulkan_context::VulkanContext>,
        renderer: Option<Box<dyn Renderer>>,
        // Hybrid rendering: separate splat renderer + raster renderer for mesh geometry
        hybrid_splat: Option<splat_renderer::GaussianSplatRenderer>,
        hybrid_raster: Option<raster_renderer::PersistentRenderer>,
        orbit: OrbitCamera,
        // Frame sync
        image_available_sem: vk::Semaphore,
        render_finished_sem: vk::Semaphore,
        in_flight_fence: vk::Fence,
        initialized: bool,
        is_hybrid: bool,
    }

    impl App {
        fn init_vulkan(&mut self) {
            let window = match self.window.as_ref() {
                Some(w) => w,
                None => return,
            };

            info!("Initializing Vulkan with window surface...");
            let mut ctx = match vulkan_context::VulkanContext::new_with_window(window, self.use_rt, self.force_validation) {
                Ok(c) => c,
                Err(e) => {
                    error!("Failed to create Vulkan context: {}", e);
                    return;
                }
            };

            // Create sync objects
            let sem_info = vk::SemaphoreCreateInfo::default();
            let fence_info = vk::FenceCreateInfo::default()
                .flags(vk::FenceCreateFlags::SIGNALED);

            let image_available = unsafe {
                ctx.device.create_semaphore(&sem_info, None).unwrap()
            };
            let render_finished = unsafe {
                ctx.device.create_semaphore(&sem_info, None).unwrap()
            };
            let fence = unsafe {
                ctx.device.create_fence(&fence_info, None).unwrap()
            };

            // Detect hybrid scene for splat path: check if glTF has non-POINTS primitives
            if self.use_splat
                && (self.scene_source.ends_with(".glb") || self.scene_source.ends_with(".gltf"))
            {
                if let Ok((doc, _, _)) = gltf::import(std::path::Path::new(&self.scene_source)) {
                    let has_meshes = doc.meshes().any(|mesh| {
                        mesh.primitives().any(|prim| prim.mode() != gltf::mesh::Mode::Points)
                    });
                    if has_meshes {
                        self.is_hybrid = true;
                        info!("Hybrid interactive scene detected: raster meshes + gaussian splats");
                    }
                }
            }

            // Create the renderer (unified via Renderer trait)
            let renderer_result: Result<Box<dyn Renderer>, String> = if self.use_splat && self.is_hybrid {
                // Hybrid: create both splat and raster renderers
                info!("Initializing hybrid splat + raster renderer...");
                let splat = splat_renderer::GaussianSplatRenderer::new(
                    &mut ctx,
                    &self.scene_source,
                    &self.pipeline_base,
                    self.width,
                    self.height,
                );
                match splat {
                    Ok(s) => {
                        let raster_pipeline = if std::path::Path::new("shadercache/gltf_pbr.vert.spv").exists() {
                            "shadercache/gltf_pbr".to_string()
                        } else if std::path::Path::new("examples/gltf_pbr.vert.spv").exists() {
                            "examples/gltf_pbr".to_string()
                        } else if std::path::Path::new("../examples/gltf_pbr.vert.spv").exists() {
                            "../examples/gltf_pbr".to_string()
                        } else {
                            "examples/gltf_pbr".to_string()
                        };
                        let raster = raster_renderer::PersistentRenderer::init(
                            &mut ctx,
                            &raster_pipeline,
                            &self.scene_source,
                            self.width,
                            self.height,
                            &self.ibl_name,
                            self.demo_lights,
                            self.sponza_lights,
                        );
                        match raster {
                            Ok(r) => {
                                self.hybrid_raster = Some(r);
                                self.hybrid_splat = Some(s);
                                // Create a dummy — hybrid_splat is the actual renderer
                                // We need something for the unified renderer slot;
                                // we'll handle hybrid specially in render_frame
                                // Return an error to skip the normal path
                                Err("__hybrid__".to_string())
                            }
                            Err(e) => {
                                // Fall back to splat-only
                                info!("Hybrid raster init failed ({}), falling back to splat-only", e);
                                self.is_hybrid = false;
                                Ok(Box::new(s) as Box<dyn Renderer>)
                            }
                        }
                    }
                    Err(e) => Err(e),
                }
            } else if self.use_splat {
                info!("Initializing gaussian splat renderer...");
                splat_renderer::GaussianSplatRenderer::new(
                    &mut ctx,
                    &self.scene_source,
                    &self.pipeline_base,
                    self.width,
                    self.height,
                ).map(|r| Box::new(r) as Box<dyn Renderer>)
            } else if self.use_rt {
                info!("Initializing RT renderer...");
                rt_renderer::RTRenderer::new(
                    &mut ctx,
                    &self.pipeline_base,
                    &self.scene_source,
                    self.width,
                    self.height,
                    &self.ibl_name,
                ).map(|r| Box::new(r) as Box<dyn Renderer>)
            } else if self.use_mesh {
                if !ctx.mesh_shader_supported {
                    Err("Mesh shader pipeline requires VK_EXT_mesh_shader".to_string())
                } else {
                    info!("Initializing mesh shader renderer...");
                    mesh_renderer::MeshShaderRenderer::new(
                        &mut ctx,
                        &self.scene_source,
                        &self.pipeline_base,
                        self.width,
                        self.height,
                        &self.ibl_name,
                        self.demo_lights,
                    ).map(|r| Box::new(r) as Box<dyn Renderer>)
                }
            } else if self.use_deferred {
                info!("Initializing deferred renderer...");
                deferred_renderer::DeferredRenderer::new(
                    &mut ctx,
                    &self.scene_source,
                    &self.pipeline_base,
                    self.width,
                    self.height,
                    &self.ibl_name,
                    self.demo_lights,
                    self.sponza_lights,
                ).map(|r| Box::new(r) as Box<dyn Renderer>)
            } else {
                info!("Initializing persistent renderer...");
                raster_renderer::PersistentRenderer::init(
                    &mut ctx,
                    &self.pipeline_base,
                    &self.scene_source,
                    self.width,
                    self.height,
                    &self.ibl_name,
                    self.demo_lights,
                    self.sponza_lights,
                ).map(|r| Box::new(r) as Box<dyn Renderer>)
            };

            // Handle hybrid case: renderer_result is Err("__hybrid__") when hybrid_splat/raster are set
            let renderer: Option<Box<dyn Renderer>> = match renderer_result {
                Ok(r) => Some(r),
                Err(ref e) if e == "__hybrid__" => None,
                Err(e) => {
                    error!("Failed to init renderer: {}", e);
                    unsafe {
                        ctx.device.destroy_semaphore(image_available, None);
                        ctx.device.destroy_semaphore(render_finished, None);
                        ctx.device.destroy_fence(fence, None);
                    }
                    ctx.destroy();
                    return;
                }
            };

            // Init orbit camera from auto-camera
            if let Some(ref r) = renderer {
                if r.has_scene_bounds() {
                    self.orbit.init_from_auto_camera(
                        r.auto_eye(),
                        r.auto_target(),
                        r.auto_up(),
                        r.auto_far(),
                    );
                }
            } else if let Some(ref s) = self.hybrid_splat {
                if s.has_scene_bounds() {
                    self.orbit.init_from_auto_camera(
                        s.auto_eye(),
                        s.auto_target(),
                        s.auto_up(),
                        s.auto_far(),
                    );
                }
            }

            self.image_available_sem = image_available;
            self.render_finished_sem = render_finished;
            self.in_flight_fence = fence;
            self.renderer = renderer;
            self.ctx = Some(ctx);
            self.initialized = true;

            info!("Interactive mode ready. Drag mouse to orbit, scroll to zoom, ESC to exit.");
        }

        fn render_frame(&mut self) {
            let ctx = match self.ctx.as_ref() {
                Some(c) => c,
                None => return,
            };

            // Wait for previous frame
            unsafe {
                let _ = ctx.device.wait_for_fences(&[self.in_flight_fence], true, u64::MAX);
                let _ = ctx.device.reset_fences(&[self.in_flight_fence]);
            }

            // Acquire swapchain image
            let (image_index, _suboptimal) = match ctx.acquire_next_image(self.image_available_sem) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    info!("Swapchain out of date, skipping frame");
                    return;
                }
                Err(e) => {
                    error!("Failed to acquire swapchain image: {:?}", e);
                    return;
                }
            };

            let extent = ctx.swapchain_extent;

            // --- Hybrid rendering path ---
            if self.is_hybrid && self.hybrid_splat.is_some() && self.hybrid_raster.is_some() {
                let aspect = extent.width as f32 / extent.height as f32;

                // Update both cameras
                let eye = self.orbit.eye();
                let target = self.orbit.target;
                let up = self.orbit.up;
                let fov_y = self.orbit.fov_y;
                let near = self.orbit.near_plane;
                let far = self.orbit.far_plane;

                let raster = self.hybrid_raster.as_mut().unwrap();
                raster.update_camera(eye, target, up, fov_y, aspect, near, far);

                // 1. Render raster scene
                let raster_cmd = match raster.render(ctx) {
                    Ok(c) => c,
                    Err(e) => {
                        error!("Hybrid raster render failed: {}", e);
                        return;
                    }
                };
                unsafe {
                    let _ = ctx.device.end_command_buffer(raster_cmd);
                }
                // Submit raster synchronously
                let raster_bufs = [raster_cmd];
                let raster_submit = vk::SubmitInfo::default().command_buffers(&raster_bufs);
                let raster_fence = unsafe {
                    ctx.device.create_fence(&vk::FenceCreateInfo::default(), None).unwrap()
                };
                unsafe {
                    let _ = ctx.device.queue_submit(ctx.graphics_queue, &[raster_submit], raster_fence);
                    let _ = ctx.device.wait_for_fences(&[raster_fence], true, u64::MAX);
                    ctx.device.destroy_fence(raster_fence, None);
                    ctx.device.free_command_buffers(ctx.command_pool, &raster_bufs);
                }

                // 2. Preload raster color + depth into splat buffers
                let raster_output = raster.output_image();
                let raster_depth = raster.depth_image();
                let raster_w = raster.width();
                let raster_h = raster.height();

                let splat = self.hybrid_splat.as_mut().unwrap();
                splat.update_camera(eye, target, up, fov_y, aspect, near, far);

                if let Err(e) = splat.preload_background(ctx, raster_output, raster_w, raster_h) {
                    error!("preload_background failed: {}", e);
                    return;
                }
                if let Err(e) = splat.preload_depth(ctx, raster_depth, raster_w, raster_h) {
                    error!("preload_depth failed: {}", e);
                    return;
                }

                // 3. Render splats on top (LOAD render pass preserves raster background)
                let cmd = match splat.render(ctx) {
                    Ok(c) => c,
                    Err(e) => {
                        error!("Hybrid splat render failed: {}", e);
                        return;
                    }
                };

                // Blit splat output to swapchain
                let swapchain_image = ctx.swapchain_images[image_index as usize];
                splat.blit_to_swapchain(&ctx.device, cmd, swapchain_image, extent);

                // End and submit
                unsafe {
                    let _ = ctx.device.end_command_buffer(cmd);
                }

                let wait_semaphores = [self.image_available_sem];
                let signal_semaphores = [self.render_finished_sem];
                let wait_stages = [vk::PipelineStageFlags::COMPUTE_SHADER];
                let cmd_bufs = [cmd];
                let submit_info = vk::SubmitInfo::default()
                    .wait_semaphores(&wait_semaphores)
                    .wait_dst_stage_mask(&wait_stages)
                    .command_buffers(&cmd_bufs)
                    .signal_semaphores(&signal_semaphores);

                unsafe {
                    let _ = ctx.device.queue_submit(
                        ctx.graphics_queue,
                        &[submit_info],
                        self.in_flight_fence,
                    );
                }

                // Present
                match ctx.queue_present(image_index, self.render_finished_sem) {
                    Ok(_) => {}
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        info!("Swapchain out of date during present");
                    }
                    Err(e) => {
                        error!("Present failed: {:?}", e);
                    }
                }

                // Free the command buffer
                unsafe {
                    let _ = ctx.device.device_wait_idle();
                    ctx.device.free_command_buffers(ctx.command_pool, &cmd_bufs);
                }
                return;
            }

            // --- Standard (non-hybrid) rendering path ---
            let renderer = match self.renderer.as_mut() {
                Some(r) => r,
                None => return,
            };

            // Animate Sponza torch light (light[1] orbits inside the courtyard)
            if self.sponza_lights {
                let elapsed = self.start_time.elapsed().as_secs_f32();
                let angle = elapsed * 0.3; // orbit speed
                let pos = glam::Vec3::new(800.0 * angle.sin(), 600.0, 400.0 * angle.cos());
                let dir = (glam::Vec3::new(0.0, -200.0, 0.0) - pos).normalize();
                let mut lights = crate::setup_sponza_lights();
                lights[1].position = pos;
                lights[1].direction = dir;
                renderer.update_lights(&lights);
            }

            // Update camera
            let aspect = extent.width as f32 / extent.height as f32;
            renderer.update_camera(
                self.orbit.eye(),
                self.orbit.target,
                self.orbit.up,
                self.orbit.fov_y,
                aspect,
                self.orbit.near_plane,
                self.orbit.far_plane,
            );

            // Render (returns a command buffer that is still recording)
            let cmd = match renderer.render(ctx) {
                Ok(c) => c,
                Err(e) => {
                    error!("Render failed: {}", e);
                    return;
                }
            };

            // Blit to swapchain image
            let swapchain_image = ctx.swapchain_images[image_index as usize];
            renderer.blit_to_swapchain(&ctx.device, cmd, swapchain_image, extent);

            // End and submit
            unsafe {
                let _ = ctx.device.end_command_buffer(cmd);
            }

            let wait_semaphores = [self.image_available_sem];
            let signal_semaphores = [self.render_finished_sem];
            let wait_stages = [renderer.wait_stage()];
            let cmd_bufs = [cmd];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmd_bufs)
                .signal_semaphores(&signal_semaphores);

            unsafe {
                let _ = ctx.device.queue_submit(
                    ctx.graphics_queue,
                    &[submit_info],
                    self.in_flight_fence,
                );
            }

            // Present
            match ctx.queue_present(image_index, self.render_finished_sem) {
                Ok(_) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    info!("Swapchain out of date during present");
                }
                Err(e) => {
                    error!("Present failed: {:?}", e);
                }
            }

            // Free the command buffer after GPU is done
            unsafe {
                let _ = ctx.device.device_wait_idle();
                ctx.device.free_command_buffers(ctx.command_pool, &cmd_bufs);
            }
        }

        fn cleanup_vulkan(&mut self) {
            if let Some(ref ctx) = self.ctx {
                unsafe { let _ = ctx.device.device_wait_idle(); }
            }

            if let Some(ref mut renderer) = self.renderer {
                if let Some(ref mut ctx) = self.ctx {
                    renderer.destroy(ctx);
                }
            }

            // Destroy hybrid renderers
            if let Some(ref mut splat) = self.hybrid_splat {
                if let Some(ref mut ctx) = self.ctx {
                    splat.cleanup(ctx);
                }
            }
            self.hybrid_splat = None;
            if let Some(ref mut raster) = self.hybrid_raster {
                if let Some(ref mut ctx) = self.ctx {
                    raster.cleanup(ctx);
                }
            }
            self.hybrid_raster = None;

            if let Some(ref ctx) = self.ctx {
                unsafe {
                    if self.image_available_sem != vk::Semaphore::null() {
                        ctx.device.destroy_semaphore(self.image_available_sem, None);
                    }
                    if self.render_finished_sem != vk::Semaphore::null() {
                        ctx.device.destroy_semaphore(self.render_finished_sem, None);
                    }
                    if self.in_flight_fence != vk::Fence::null() {
                        ctx.device.destroy_fence(self.in_flight_fence, None);
                    }
                }
            }

            if let Some(mut ctx) = self.ctx.take() {
                ctx.destroy();
            }
        }
    }

    impl ApplicationHandler for App {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_none() {
                let window_attrs = Window::default_attributes()
                    .with_title("Lux Playground")
                    .with_inner_size(winit::dpi::LogicalSize::new(self.width, self.height))
                    .with_resizable(false);

                match event_loop.create_window(window_attrs) {
                    Ok(window) => {
                        info!("Window created: {}x{}", self.width, self.height);
                        self.window = Some(window);
                        self.init_vulkan();
                    }
                    Err(e) => {
                        error!("Failed to create window: {}", e);
                        event_loop.exit();
                    }
                }
            }
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => {
                    info!("Window close requested");
                    self.cleanup_vulkan();
                    event_loop.exit();
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                        info!("Escape pressed, closing");
                        self.cleanup_vulkan();
                        event_loop.exit();
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        self.orbit.dragging = state == ElementState::Pressed;
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if self.orbit.dragging {
                        let dx = position.x - self.orbit.last_x;
                        let dy = position.y - self.orbit.last_y;
                        self.orbit.yaw += dx as f32 * 0.005;
                        self.orbit.pitch += dy as f32 * 0.005;
                        self.orbit.pitch = self.orbit.pitch.clamp(-1.5, 1.5);
                        if let Some(window) = &self.window {
                            window.request_redraw();
                        }
                    }
                    self.orbit.last_x = position.x;
                    self.orbit.last_y = position.y;
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                    };
                    self.orbit.distance *= 1.0 - scroll * 0.1;
                    self.orbit.distance = self.orbit.distance.clamp(0.01, 1000.0);
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                WindowEvent::RedrawRequested => {
                    if self.initialized {
                        self.render_frame();
                    }
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                _ => {}
            }
        }
    }

    let event_loop = EventLoop::new().map_err(|e| format!("Failed to create event loop: {}", e))?;

    let mut app = App {
        window: None,
        pipeline_base: pipeline_base.to_string(),
        scene_source: scene_source.to_string(),
        width,
        height,
        use_rt,
        use_mesh,
        use_splat,
        use_deferred,
        ibl_name: ibl_name.to_string(),
        force_validation,
        demo_lights,
        sponza_lights,
        start_time: std::time::Instant::now(),
        ctx: None,
        renderer: None,
        hybrid_splat: None,
        hybrid_raster: None,
        orbit: OrbitCamera::new(),
        image_available_sem: vk::Semaphore::null(),
        render_finished_sem: vk::Semaphore::null(),
        in_flight_fence: vk::Fence::null(),
        initialized: false,
        is_hybrid: false,
    };

    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("Event loop error: {}", e))?;

    Ok(())
}
