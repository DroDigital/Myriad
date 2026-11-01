//! Windowed game: winit event loop, input mapping, surface presentation.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use glam::Vec2;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::game::Controls;
use crate::gpu::Gpu;
use crate::run::RunConfig;
use crate::session::{Session, SessionConfig};

/// Mass granted per press of the developer key (H, with --dev).
const DEV_GRANT: f64 = 100.0;

#[derive(Clone, Debug)]
pub struct AppOptions {
    pub agents: u32,
    pub capacity: u32,
    pub seed: u64,
    /// Enables developer keys.
    pub dev: bool,
    /// Spawn threats and the Cleaner.
    pub threats: bool,
}

pub fn run(opts: AppOptions) -> Result<()> {
    let event_loop = EventLoop::new().context("creating event loop")?;
    let mut app = App {
        opts,
        running: None,
        error: None,
    };
    event_loop.run_app(&mut app)?;
    match app.error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Raw input state from keyboard, mouse and gamepad, turned into
/// [`Controls`] each frame.
#[derive(Default)]
struct Input {
    up: bool,
    down: bool,
    left: bool,
    right: bool,
    gather: bool,
    spread: bool,
    build_key: bool,
    mouse_px: Option<Vec2>,
    mouse_down: bool,
    /// Presses waiting to be consumed by a simulation step.
    form: [bool; 2],
    cycle: [bool; 2],
    start: bool,
    pad: PadState,
}

/// Gamepad state (gilrs): left stick steers, triggers Gather/Spread,
/// A/B fire the Form slots, X holds Build, bumpers cycle the slots.
#[derive(Default)]
struct PadState {
    stick: Vec2,
    gather: f32,
    spread: f32,
    build: bool,
}

const STICK_DEADZONE: f32 = 0.2;

impl Input {
    fn key(&mut self, code: KeyCode, pressed: bool, repeat: bool) {
        let fresh = pressed && !repeat;
        match code {
            KeyCode::KeyW | KeyCode::ArrowUp => self.up = pressed,
            KeyCode::KeyS | KeyCode::ArrowDown => self.down = pressed,
            KeyCode::KeyA | KeyCode::ArrowLeft => self.left = pressed,
            KeyCode::KeyD | KeyCode::ArrowRight => self.right = pressed,
            KeyCode::Space => self.gather = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.spread = pressed,
            KeyCode::KeyB | KeyCode::KeyL => self.build_key = pressed,
            KeyCode::KeyZ | KeyCode::KeyJ => self.form[0] |= fresh,
            KeyCode::KeyX | KeyCode::KeyK => self.form[1] |= fresh,
            KeyCode::KeyQ => self.cycle[0] |= fresh,
            KeyCode::KeyE => self.cycle[1] |= fresh,
            KeyCode::Enter | KeyCode::NumpadEnter => self.start |= fresh,
            _ => {}
        }
    }

    fn gamepad(&mut self, gilrs: &mut gilrs::Gilrs) {
        use gilrs::{Axis, Button, EventType};
        while let Some(ev) = gilrs.next_event() {
            match ev.event {
                EventType::ButtonPressed(b, _) => match b {
                    Button::South => self.form[0] = true,
                    Button::Start => self.start = true,
                    Button::East => self.form[1] = true,
                    Button::West => self.pad.build = true,
                    Button::LeftTrigger => self.cycle[0] = true,
                    Button::RightTrigger => self.cycle[1] = true,
                    _ => {}
                },
                EventType::ButtonReleased(Button::West, _) => self.pad.build = false,
                EventType::ButtonChanged(Button::RightTrigger2, v, _) => self.pad.gather = v,
                EventType::ButtonChanged(Button::LeftTrigger2, v, _) => self.pad.spread = v,
                EventType::AxisChanged(Axis::LeftStickX, v, _) => self.pad.stick.x = v,
                EventType::AxisChanged(Axis::LeftStickY, v, _) => self.pad.stick.y = v,
                EventType::Disconnected => self.pad = PadState::default(),
                _ => {}
            }
        }
    }

    fn controls(&self, session: &Session, viewport: Vec2) -> Controls {
        let axis = |neg: bool, pos: bool| (pos as i32 - neg as i32) as f32;
        let cursor = self
            .mouse_px
            .map(|px| session.run.game.camera.screen_to_world(px, viewport));
        let stick = if self.pad.stick.length() > STICK_DEADZONE {
            self.pad.stick
        } else {
            Vec2::ZERO
        };
        let keys = Vec2::new(axis(self.left, self.right), axis(self.down, self.up));
        Controls {
            move_dir: keys + stick,
            gather: (axis(self.spread, self.gather) + self.pad.gather - self.pad.spread)
                .clamp(-1.0, 1.0),
            pointer: cursor.filter(|_| self.mouse_down),
            form: self.form,
            cycle: self.cycle,
            start: self.start,
            build: self.build_key || self.pad.build,
            // Aim with the stick if it is pushed, else at the cursor.
            aim: if stick != Vec2::ZERO {
                session.run.game.swarm_center.map(|c| c + stick * 10.0)
            } else {
                cursor
            },
        }
    }

    /// Clears presses once a simulation step has consumed them.
    fn consume_presses(&mut self) {
        self.form = [false; 2];
        self.cycle = [false; 2];
        self.start = false;
    }
}

struct Running {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    session: Session,
    input: Input,
    /// Gamepad support; `None` if gilrs could not start (e.g. no udev).
    gilrs: Option<gilrs::Gilrs>,
    dev: bool,
    last_frame: Instant,
    fps_window_start: Instant,
    fps_frames: u32,
}

impl Running {
    fn new(event_loop: &ActiveEventLoop, opts: &AppOptions) -> Result<Self> {
        let attrs = Window::default_attributes()
            .with_title("Myriad")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let window = Arc::new(event_loop.create_window(attrs)?);
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        desc.display = Some(Box::new(event_loop.owned_display_handle()));
        let instance = wgpu::Instance::new(desc);
        let surface = instance.create_surface(window.clone())?;
        let gpu = pollster::block_on(Gpu::from_instance(instance, Some(&surface)))?;
        log::info!("adapter: {}", gpu.adapter_summary());

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1))
            .context("surface is not supported by the adapter")?;
        // Prefer an sRGB format; the shaders output linear colour.
        let caps = surface.get_capabilities(&gpu.adapter);
        if let Some(srgb) = caps.formats.iter().copied().find(|f| f.is_srgb()) {
            config.format = srgb;
        }
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&gpu.device, &config);

        let cfg = SessionConfig {
            run: RunConfig {
                initial_critters: opts.agents,
                capacity: opts.capacity.max(opts.agents),
                threats: opts.threats,
                title: true,
                ..Default::default()
            },
            seed: opts.seed,
        };
        let aspect = config.width as f32 / config.height as f32;
        let session = Session::new(&gpu, config.format, aspect, &cfg);
        let now = Instant::now();
        Ok(Self {
            window,
            surface,
            config,
            gpu,
            session,
            input: Input::default(),
            gilrs: gilrs::Gilrs::new()
                .inspect_err(|e| log::warn!("gamepads unavailable: {e}"))
                .ok(),
            dev: opts.dev,
            last_frame: now,
            fps_window_start: now,
            fps_frames: 0,
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
        self.session.run.game.camera.aspect = width as f32 / height as f32;
    }

    fn frame(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        let viewport = Vec2::new(self.config.width as f32, self.config.height as f32);
        if let Some(gilrs) = &mut self.gilrs {
            self.input.gamepad(gilrs);
        }
        let controls = self.input.controls(&self.session, viewport);
        if self.session.advance(&self.gpu, &controls, dt) > 0 {
            self.input.consume_presses();
        }

        let texture = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.gpu.device, &self.config);
                return;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
            wgpu::CurrentSurfaceTexture::Validation => {
                log::error!("surface validation error");
                return;
            }
        };
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.session
            .draw(&self.gpu, &view, self.config.width, self.config.height);
        self.window.pre_present_notify();
        self.gpu.queue.present(texture);

        self.fps_frames += 1;
        let elapsed = (now - self.fps_window_start).as_secs_f32();
        if elapsed >= 0.5 {
            let fps = self.fps_frames as f32 / elapsed;
            self.window.set_title(&format!(
                "Myriad | mass {} | {fps:.0} fps | WASD/mouse steer, Space gather, Shift spread, Z/X Forms, B build, Enter start, R restart",
                self.session.run.colony.mass()
            ));
            self.fps_window_start = now;
            self.fps_frames = 0;
        }
    }
}

struct App {
    opts: AppOptions,
    running: Option<Running>,
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.running.is_some() {
            return;
        }
        match Running::new(event_loop, &self.opts) {
            Ok(r) => self.running = Some(r),
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(r) = self.running.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => r.resize(size.width, size.height),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    let pressed = event.state == ElementState::Pressed;
                    if code == KeyCode::Escape && pressed {
                        event_loop.exit();
                    } else if code == KeyCode::KeyR && pressed && !event.repeat {
                        r.session.restart(&r.gpu);
                    } else if code == KeyCode::KeyH && pressed && !event.repeat && r.dev {
                        r.session.grant(DEV_GRANT);
                    } else {
                        r.input.key(code, pressed, event.repeat);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                r.input.mouse_px = Some(Vec2::new(position.x as f32, position.y as f32));
            }
            WindowEvent::CursorLeft { .. } => r.input.mouse_px = None,
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => r.input.mouse_down = state == ElementState::Pressed,
            WindowEvent::RedrawRequested => r.frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(r) = &self.running {
            r.window.request_redraw();
        }
    }
}
