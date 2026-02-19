//! CLI entry point for the Lux shader playground.
//!
//! Supports scene/pipeline architecture with auto-detection of render paths.
//! Can run headless (offscreen render to PNG) or interactively (winit window).

mod camera;
pub mod gltf_loader;
mod raster_renderer;
pub mod reflected_pipeline;
mod rt_renderer;
mod scene;
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
    #[arg(long, value_parser = ["triangle", "fullscreen", "pbr", "rt"])]
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
        Ok("examples/gltf_pbr".to_string())
    } else if scene == "fullscreen" {
        Err("--pipeline required for fullscreen scenes".to_string())
    } else if scene == "triangle" {
        Ok("examples/hello_triangle".to_string())
    } else {
        Ok("examples/pbr_basic".to_string())
    }
}

/// Auto-detect the render path from which .spv files exist for a pipeline base.
fn detect_render_path(pipeline_base: &str) -> &'static str {
    if Path::new(&format!("{}.rgen.spv", pipeline_base)).exists() {
        "rt"
    } else if Path::new(&format!("{}.vert.spv", pipeline_base)).exists()
        && Path::new(&format!("{}.frag.spv", pipeline_base)).exists()
    {
        "raster"
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

    let pipeline_base = if let Some(p) = &args.pipeline {
        p.clone()
    } else if let Some(shader) = &args.shader {
        shader.clone()
    } else {
        resolve_default_pipeline(&scene_source).unwrap_or_else(|e| panic!("{}", e))
    };

    let render_path = detect_render_path(&pipeline_base);

    info!("Scene: {}", scene_source);
    info!("Pipeline: {}", pipeline_base);
    info!("Render path: {}", render_path);
    info!("Resolution: {}x{}", args.width, args.height);
    info!("Output: {}", args.output);

    if args.interactive && !args.headless {
        run_interactive(&pipeline_base, &scene_source, render_path, args.width, args.height)?;
    } else {
        run_headless(&pipeline_base, &scene_source, render_path, args.width, args.height, &args.output)?;
    }

    Ok(())
}

/// Run in headless mode: create context, render offscreen, save PNG.
fn run_headless(
    pipeline_base: &str,
    scene_source: &str,
    render_path: &str,
    width: u32,
    height: u32,
    output: &str,
) -> Result<(), String> {
    let enable_rt = render_path == "rt";

    info!("Creating Vulkan context (RT: {})...", enable_rt);
    let mut ctx = vulkan_context::VulkanContext::new(enable_rt)?;

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
            rt_renderer::render_rt(&mut ctx, pipeline_base, scene_source, width, height, output_path)
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

/// Run in interactive mode with a winit window.
fn run_interactive(
    pipeline_base: &str,
    scene_source: &str,
    render_path: &str,
    width: u32,
    height: u32,
) -> Result<(), String> {
    use winit::application::ApplicationHandler;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, EventLoop};
    use winit::keyboard::{KeyCode, PhysicalKey};
    use winit::window::{Window, WindowId};

    struct App {
        window: Option<Window>,
        pipeline_base: String,
        scene_source: String,
        render_path: String,
        width: u32,
        height: u32,
        rendered: bool,
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
                    event_loop.exit();
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                        info!("Escape pressed, closing");
                        event_loop.exit();
                    }
                }
                WindowEvent::RedrawRequested => {
                    if !self.rendered {
                        self.rendered = true;

                        // In interactive mode, we do an offscreen render and display
                        // the result info. A full swapchain-based interactive loop
                        // would require surface creation and presentation.
                        info!(
                            "Interactive mode: rendering scene '{}' with pipeline '{}' ({})",
                            self.scene_source, self.pipeline_base, self.render_path
                        );

                        let enable_rt = self.render_path == "rt";
                        match vulkan_context::VulkanContext::new(enable_rt) {
                            Ok(mut ctx) => {
                                let output_path = Path::new("interactive_output.png");
                                let result = match self.render_path.as_str() {
                                    "raster" => raster_renderer::render_raster(
                                        &mut ctx,
                                        &self.pipeline_base,
                                        &self.scene_source,
                                        self.width,
                                        self.height,
                                        output_path,
                                    ),
                                    "fullscreen" => raster_renderer::render_fullscreen(
                                        &mut ctx,
                                        &self.pipeline_base,
                                        self.width,
                                        self.height,
                                        output_path,
                                    ),
                                    "rt" => rt_renderer::render_rt(
                                        &mut ctx,
                                        &self.pipeline_base,
                                        &self.scene_source,
                                        self.width,
                                        self.height,
                                        output_path,
                                    ),
                                    _ => Err(format!("Unknown render path: {}", self.render_path)),
                                };

                                unsafe {
                                    let _ = ctx.device.device_wait_idle();
                                }
                                ctx.destroy();

                                match result {
                                    Ok(()) => info!("Interactive render saved to {:?}", output_path),
                                    Err(e) => error!("Interactive render failed: {}", e),
                                }
                            }
                            Err(e) => {
                                error!("Failed to create Vulkan context: {}", e);
                            }
                        }
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
        render_path: render_path.to_string(),
        width,
        height,
        rendered: false,
    };

    event_loop
        .run_app(&mut app)
        .map_err(|e| format!("Event loop error: {}", e))?;

    Ok(())
}
