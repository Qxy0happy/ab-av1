use crate::command::args::PixelFormat;
use anyhow::{Context, ensure};
use clap::Parser;
use std::{borrow::Cow, fmt::Display, sync::Arc, thread};

const DEFAULT_VMAF_FPS: f32 = 25.0;

/// Common vmaf options.
#[derive(Debug, Parser, Clone)]
pub struct Vmaf {
    /// Set to calculate vmaf when it would otherwise not be, e.g. when calculating xpsnr.
    /// So using this allows both vmaf & xpsnr to be calculated at the same time.
    // TODO: nicer if named "--vmaf"
    #[arg(long, num_args=0..=1, default_missing_value = "true")]
    pub and_vmaf: Option<bool>,

    /// Additional vmaf arg(s). E.g. --vmaf n_threads=8 --vmaf n_subsample=4
    ///
    /// By default `n_threads` is set to available system threads.
    ///
    /// Also see https://ffmpeg.org/ffmpeg-filters.html#libvmaf.
    #[arg(long = "vmaf", value_parser = parse_vmaf_arg)]
    pub vmaf_args: Vec<Arc<str>>,

    /// Calculate VMAF on the GPU using ffmpeg's `libvmaf_cuda` filter.
    ///
    /// Frames are uploaded to the GPU with `hwupload_cuda` after all other filters, so
    /// scaling and pixel format behave the same as the default CPU `libvmaf` path.
    /// VMAF is computed on 8-bit `yuv420p` frames in this mode, regardless of `--pix-format`.
    ///
    /// Falls back to CPU `libvmaf` when the GPU run fails. Requires a CUDA capable GPU and
    /// ffmpeg built with `--enable-libvmaf` & CUDA support (ffmpeg `libvmaf_cuda` filter).
    #[arg(long)]
    pub cuda_vmaf: bool,

    /// Additional ffmpeg input arg(s), applied to both vmaf inputs.
    /// These are added as ffmpeg input file options. E.g.
    /// `--vmaf-input hwaccel=cuda --vmaf-input hwaccel_output_format=cuda`
    ///
    /// Hardware frames from `hwaccel_output_format` are downloaded inside the filter graph,
    /// so decoding can use the GPU while filters, scaling & scores stay identical.
    ///
    /// Both inputs must be hardware decodable when `hwaccel_output_format` is set, otherwise
    /// ffmpeg falls back to software for that input and the upload/download graph fails.
    /// Using `hwaccel` alone decodes on the GPU and falls back per input, which is safer.
    #[arg(long = "vmaf-input", allow_hyphen_values = true, value_parser = parse_vmaf_input_arg)]
    pub vmaf_input_args: Vec<Arc<str>>,

    /// Video resolution scale to use in VMAF analysis. If set, video streams will be bicubic
    /// scaled to this during VMAF analysis. `auto` (default) automatically sets
    /// based on the model and input video resolution. `none` disables any scaling.
    /// `WxH` format may be used to specify custom scaling, e.g. `1920x1080`.
    ///
    /// auto behaviour:
    /// * 1k model (default for resolutions <= 2560x1440) if width and height
    ///   are less than 1728 & 972 respectively upscale to 1080p. Otherwise no scaling.
    /// * 4k model (default for resolutions > 2560x1440) if width and height
    ///   are less than 3456 & 1944 respectively upscale to 4k. Otherwise no scaling.
    ///
    /// The auto behaviour is based on the distorted video dimensions, equivalent
    /// to post input/reference vfilter dimensions.
    ///
    /// Scaling happens after any input/reference vfilters.
    #[arg(long, default_value_t, value_parser = parse_vmaf_scale)]
    pub vmaf_scale: VmafScale,

    /// Frame rate override used to analyse both reference & distorted videos.
    /// Maps to ffmpeg `-r` input arg.
    ///
    /// Setting to 0 disables use.
    #[arg(long, default_value_t = DEFAULT_VMAF_FPS)]
    pub vmaf_fps: f32,
}

impl Default for Vmaf {
    fn default() -> Self {
        Self {
            and_vmaf: None,
            vmaf_args: <_>::default(),
            cuda_vmaf: false,
            vmaf_input_args: <_>::default(),
            vmaf_scale: <_>::default(),
            vmaf_fps: DEFAULT_VMAF_FPS,
        }
    }
}

impl std::hash::Hash for Vmaf {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.and_vmaf.hash(state);
        self.vmaf_args.hash(state);
        // GPU scores are 8-bit while CPU scores follow --pix-format, keep the cache entries apart
        self.cuda_vmaf.hash(state);
        self.vmaf_input_args.hash(state);
        self.vmaf_scale.hash(state);
        self.vmaf_fps.to_ne_bytes().hash(state);
    }
}

fn parse_vmaf_arg(arg: &str) -> anyhow::Result<Arc<str>> {
    Ok(arg.to_owned().into())
}

fn parse_vmaf_input_arg(arg: &str) -> anyhow::Result<Arc<str>> {
    let mut arg = arg.trim().to_owned();
    ensure!(!arg.is_empty(), "vmaf-input arg is empty");
    if !arg.starts_with('-') {
        arg.insert(0, '-');
    }
    Ok(arg.into())
}

/// Software frame formats used to download hardware frames before the vmaf filters.
#[derive(Debug, Clone, Copy)]
struct HwFrames {
    distorted: &'static str,
    reference: &'static str,
}

/// Software format to download hardware frames as, from the input's pixel format.
fn hw_download_format(pix_fmt: Option<PixelFormat>) -> &'static str {
    match pix_fmt {
        // hardware decoders output nv12 for 8-bit & p010le for 10-bit
        Some(PixelFormat::Yuv420p) | None => "nv12",
        Some(_) => "p010le",
    }
}

impl Vmaf {
    pub fn fps(&self) -> Option<f32> {
        Some(self.vmaf_fps).filter(|r| *r > 0.0)
    }

    /// ffmpeg input options for both vmaf inputs, split into individual args.
    pub fn input_args(&self) -> Vec<Arc<str>> {
        self.vmaf_input_args
            .iter()
            .flat_map(|arg| match arg.split_once('=') {
                Some((opt, val)) => vec![Arc::from(opt), Arc::from(val)],
                None => vec![arg.clone()],
            })
            .collect()
    }

    /// Hardware pixel format requested by `--vmaf-input hwaccel_output_format=<fmt>`.
    fn hw_output_format(&self) -> Option<Arc<str>> {
        let args = self.input_args();
        let idx = args
            .iter()
            .position(|a| a.as_ref() == "-hwaccel_output_format")?;
        let val = args.get(idx + 1)?;
        (val.as_ref() != "none").then(|| val.clone())
    }

    /// Returns the ffmpeg `filter_complex` graph for calculating vmaf.
    ///
    /// `input_fmts` are the (distorted, reference) pixel formats, used to pick the
    /// `hwdownload` format when `--vmaf-input` requests hardware frames.
    /// Returns the default CPU graph and, when `--cuda-vmaf` is set, the `libvmaf_cuda`
    /// graph to try first.
    pub fn ffmpeg_lavfi_attempts(
        &self,
        distorted_res: Option<(u32, u32)>,
        pix_fmt: Option<PixelFormat>,
        ref_vfilter: Option<&str>,
        input_fmts: (Option<PixelFormat>, Option<PixelFormat>),
    ) -> (String, Option<String>) {
        let hw = self.hw_frames(input_fmts);
        let cpu = match hw {
            None => self.ffmpeg_lavfi(distorted_res, pix_fmt, ref_vfilter),
            Some(hw) => self.lavfi(distorted_res, pix_fmt, ref_vfilter, false, Some(hw)),
        };
        let cuda = self.cuda_vmaf.then(|| {
            self.lavfi(
                distorted_res,
                Some(PixelFormat::Yuv420p),
                ref_vfilter,
                true,
                hw,
            )
        });
        (cpu, cuda)
    }

    /// Returns ffmpeg `filter_complex`/`lavfi` value for calculating vmaf.
    pub fn ffmpeg_lavfi(
        &self,
        distorted_res: Option<(u32, u32)>,
        pix_fmt: Option<PixelFormat>,
        ref_vfilter: Option<&str>,
    ) -> String {
        self.lavfi(distorted_res, pix_fmt, ref_vfilter, false, None)
    }

    /// Hardware frame formats to download, when `--vmaf-input` uses `hwaccel_output_format`.
    fn hw_frames(
        &self,
        (distorted, reference): (Option<PixelFormat>, Option<PixelFormat>),
    ) -> Option<HwFrames> {
        self.hw_output_format().map(|_| HwFrames {
            distorted: hw_download_format(distorted),
            reference: hw_download_format(reference),
        })
    }

    fn lavfi(
        &self,
        distorted_res: Option<(u32, u32)>,
        pix_fmt: Option<PixelFormat>,
        ref_vfilter: Option<&str>,
        cuda: bool,
        hw: Option<HwFrames>,
    ) -> String {
        let (filter, upload) = match cuda {
            false => ("libvmaf", ""),
            true => ("libvmaf_cuda", ",hwupload_cuda"),
        };
        let mut args = self.vmaf_args.clone();
        // libvmaf_cuda crashes ffmpeg when given n_threads, so it's only defaulted for cpu
        if !cuda && !args.iter().any(|a| a.contains("n_threads")) {
            // default n_threads to all cores
            args.push(
                format!(
                    "n_threads={}",
                    thread::available_parallelism().map_or(1, |p| p.get())
                )
                .into(),
            );
        }
        let mut lavfi = format!("{filter}=shortest=true:ts_sync_mode=nearest");
        if !args.is_empty() {
            lavfi.push(':');
            lavfi.push_str(&args.join(":"));
        }

        let mut model = VmafModel::from_args(&args);
        if let (None, Some((w, h))) = (model, distorted_res)
            && w > 2560
            && h > 1440
        {
            // for >2k resolutions use 4k model
            lavfi.push_str(":model=version=vmaf_4k_v0.6.1");
            model = Some(VmafModel::V0_6_1_4k);
        }

        let ref_vf: Cow<_> = match ref_vfilter {
            None => "".into(),
            Some(vf) if vf.ends_with(',') => vf.into(),
            Some(vf) => format!("{vf},").into(),
        };
        let (dl_dis, dl_ref) = match hw {
            None => (String::new(), String::new()),
            Some(hw) => (
                format!("hwdownload,format={},", hw.distorted),
                format!("hwdownload,format={},", hw.reference),
            ),
        };
        let format = pix_fmt.map(|v| format!("format={v},")).unwrap_or_default();
        let scale = self
            .vf_scale(model.unwrap_or_default(), distorted_res)
            .map(|(w, h)| format!("scale={w}:{h}:flags=bicubic,"))
            .unwrap_or_default();

        // prefix:
        // * download hardware frames if `--vmaf-input` asked for them
        // * Add reference-vfilter if any
        // * convert both streams to common pixel format
        // * scale to vmaf width if necessary
        // * sync presentation timestamp
        // * upload to the GPU as the last step (cuda only)
        let prefix = format!(
            "[0:v]{dl_dis}{format}{scale}setpts=PTS-STARTPTS,settb=AVTB{upload}[dis];\
             [1:v]{dl_ref}{format}{ref_vf}{scale}setpts=PTS-STARTPTS,settb=AVTB{upload}[ref];\
             [dis][ref]"
        );

        lavfi.insert_str(0, &prefix);
        lavfi
    }

    fn vf_scale(&self, model: VmafModel, distorted_res: Option<(u32, u32)>) -> Option<(i32, i32)> {
        match (self.vmaf_scale, distorted_res) {
            (VmafScale::Auto, Some((w, h))) => match model {
                m if m.wants_1080_scale() && w < 1728 && h < 972 => {
                    Some(minimally_scale((w, h), (1920, 1080)))
                }
                m if m.wants_2160_scale() && w < 3456 && h < 1944 => {
                    Some(minimally_scale((w, h), (3840, 2160)))
                }
                _ => None,
            },
            (VmafScale::Custom { width, height }, Some((w, h))) => {
                Some(minimally_scale((w, h), (width, height)))
            }
            (VmafScale::Custom { width, height }, None) => Some((width as _, height as _)),
            _ => None,
        }
    }
}

/// Return the smallest ffmpeg vf `(w, h)` scale values so that at least one of the
/// `target_w` or `target_h` bounds are met.
fn minimally_scale((from_w, from_h): (u32, u32), (target_w, target_h): (u32, u32)) -> (i32, i32) {
    let w_factor = from_w as f64 / target_w as f64;
    let h_factor = from_h as f64 / target_h as f64;
    if h_factor > w_factor {
        (-1, target_h as _) // scale vertically
    } else {
        (target_w as _, -1) // scale horizontally
    }
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VmafScale {
    None,
    #[default]
    Auto,
    Custom {
        width: u32,
        height: u32,
    },
}

fn parse_vmaf_scale(vs: &str) -> anyhow::Result<VmafScale> {
    const ERR: &str = "vmaf-scale must be 'none', 'auto' or WxH format e.g. '1920x1080'";
    match vs {
        "none" => Ok(VmafScale::None),
        "auto" => Ok(VmafScale::Auto),
        _ => {
            let (w, h) = vs.split_once('x').context(ERR)?;
            let (width, height) = (w.parse().context(ERR)?, h.parse().context(ERR)?);
            Ok(VmafScale::Custom { width, height })
        }
    }
}

impl Display for VmafScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => "none".fmt(f),
            Self::Auto => "auto".fmt(f),
            Self::Custom { width, height } => write!(f, "{width}x{height}"),
        }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
enum VmafModel {
    /// Default 1080p model.
    #[default]
    V0_6_1,
    /// 4k model.
    V0_6_1_4k,
    V1_0_16_3d0h,
    V1_0_16_5d0h,
    V1_0_16_1d5h4k,
    V1_0_16_3d0h4k,
    /// Some other user specified model.
    Custom,
}

impl VmafModel {
    fn from_args(args: &[Arc<str>]) -> Option<Self> {
        let mut using_custom_model: Vec<_> = args.iter().filter(|v| v.contains("model")).collect();

        match using_custom_model.len() {
            0 => None,
            1 => Some(match using_custom_model.remove(0) {
                v if v.ends_with("version=vmaf_v0.6.1") => Self::V0_6_1,
                v if v.ends_with("version=vmaf_4k_v0.6.1") => Self::V0_6_1_4k,
                v if v.ends_with("version=vmaf_v1.0.16_3d0h") => Self::V1_0_16_3d0h,
                v if v.ends_with("version=vmaf_v1.0.16_hfr_3d0h") => Self::V1_0_16_3d0h,
                v if v.ends_with("version=vmaf_v1.0.16_5d0h") => Self::V1_0_16_5d0h,
                v if v.ends_with("version=vmaf_v1.0.16_hfr_5d0h") => Self::V1_0_16_5d0h,
                v if v.ends_with("version=vmaf_v1.0.16_1d5h_2160") => Self::V1_0_16_1d5h4k,
                v if v.ends_with("version=vmaf_v1.0.16_hfr_1d5h_2160") => Self::V1_0_16_1d5h4k,
                v if v.ends_with("version=vmaf_v1.0.16_3d0h_2160") => Self::V1_0_16_3d0h4k,
                v if v.ends_with("version=vmaf_v1.0.16_hfr_3d0h_2160") => Self::V1_0_16_3d0h4k,
                _ => Self::Custom,
            }),
            _ => Some(Self::Custom),
        }
    }

    fn wants_1080_scale(self) -> bool {
        matches!(self, Self::V0_6_1 | Self::V1_0_16_3d0h | Self::V1_0_16_5d0h)
    }

    fn wants_2160_scale(self) -> bool {
        matches!(
            self,
            Self::V0_6_1_4k | Self::V1_0_16_1d5h4k | Self::V1_0_16_3d0h4k
        )
    }
}

#[test]
fn vmaf_lavfi() {
    let vmaf = Vmaf {
        vmaf_args: vec!["n_threads=5".into(), "n_subsample=4".into()],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(
            None,
            Some(PixelFormat::Yuv420p),
            Some("scale=1280:-1,fps=24")
        ),
        "[0:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,scale=1280:-1,fps=24,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads=5:n_subsample=4"
    );
}

#[test]
fn vmaf_lavfi_default() {
    let vmaf = Vmaf::default();
    let expected = format!(
        "[0:v]setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads={}",
        thread::available_parallelism().map_or(1, |p| p.get())
    );
    assert_eq!(vmaf.ffmpeg_lavfi(None, None, None), expected);
}

#[test]
fn vmaf_lavfi_default_pix_fmt() {
    let vmaf = Vmaf::default();
    let expected = format!(
        "[0:v]format=yuv420p10le,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p10le,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads={}",
        thread::available_parallelism().map_or(1, |p| p.get())
    );
    assert_eq!(
        vmaf.ffmpeg_lavfi(None, Some(PixelFormat::Yuv420p10le), None),
        expected
    );
}

/// GPU graph keeps the cpu filters & scaling, uploading as the last step before `libvmaf_cuda`
#[test]
fn vmaf_lavfi_cuda() {
    let vmaf = Vmaf {
        cuda_vmaf: true,
        ..<_>::default()
    };
    let expected = "[0:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB,hwupload_cuda[dis];\
         [1:v]format=yuv420p,scale=1280:-1,fps=24,setpts=PTS-STARTPTS,settb=AVTB,hwupload_cuda[ref];\
         [dis][ref]libvmaf_cuda=shortest=true:ts_sync_mode=nearest";
    let (_, cuda) =
        vmaf.ffmpeg_lavfi_attempts(None, None, Some("scale=1280:-1,fps=24"), (None, None));
    assert_eq!(cuda.expect("cuda graph"), expected);
}

/// CUDA is opt-in, and the cpu graph is unchanged by the flag
#[test]
fn vmaf_lavfi_attempts_cuda_opt_in() {
    let (_, cuda) = Vmaf::default().ffmpeg_lavfi_attempts(None, None, None, (None, None));
    assert!(cuda.is_none());

    let vmaf = Vmaf {
        cuda_vmaf: true,
        ..<_>::default()
    };
    let (cpu, cuda) = vmaf.ffmpeg_lavfi_attempts(None, None, None, (None, None));
    assert_eq!(cpu, Vmaf::default().ffmpeg_lavfi(None, None, None));
    let cuda = cuda.expect("cuda graph");
    assert!(cuda.contains("libvmaf_cuda="));
    assert!(!cpu.contains("libvmaf_cuda="));
}

/// `--vmaf-input hwaccel_output_format` frames are downloaded before the common filters
#[test]
fn vmaf_lavfi_hw_input() {
    let vmaf = Vmaf {
        vmaf_input_args: vec!["-hwaccel=cuda".into(), "-hwaccel_output_format=cuda".into()],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.input_args().join(" "),
        "-hwaccel cuda -hwaccel_output_format cuda"
    );

    // 8-bit input downloads as nv12
    let lavfi = vmaf
        .ffmpeg_lavfi_attempts(None, Some(PixelFormat::Yuv420p), None, (None, None))
        .0;
    assert!(
        lavfi.starts_with(
            "[0:v]hwdownload,format=nv12,format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[dis];"
        ),
        "{lavfi}"
    );

    // each input is downloaded using its own bit depth
    let lavfi = vmaf
        .ffmpeg_lavfi_attempts(
            None,
            Some(PixelFormat::Yuv420p10le),
            None,
            (Some(PixelFormat::Yuv420p10le), Some(PixelFormat::Yuv420p)),
        )
        .0;
    assert!(
        lavfi.starts_with(
            "[0:v]hwdownload,format=p010le,format=yuv420p10le,setpts=PTS-STARTPTS,settb=AVTB[dis];"
        ),
        "{lavfi}"
    );
    assert!(
        lavfi.contains(
            "[1:v]hwdownload,format=nv12,format=yuv420p10le,setpts=PTS-STARTPTS,settb=AVTB[ref];"
        ),
        "{lavfi}"
    );

    // combined with --cuda-vmaf frames go back up after the cpu filters
    let vmaf_cuda = Vmaf {
        cuda_vmaf: true,
        ..vmaf
    };
    let (_, cuda) = vmaf_cuda.ffmpeg_lavfi_attempts(None, None, None, (None, None));
    let cuda = cuda.expect("cuda graph");
    assert!(
        cuda.contains(
            "hwdownload,format=nv12,format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB,hwupload_cuda[dis]"
        ),
        "{cuda}"
    );

    // hwaccel_output_format=none keeps frames in software
    let vmaf = Vmaf {
        vmaf_input_args: vec!["-hwaccel_output_format=none".into()],
        ..<_>::default()
    };
    assert!(!vmaf.ffmpeg_lavfi(None, None, None).contains("hwdownload"));
}

#[test]
fn vmaf_lavfi_include_n_threads() {
    let vmaf = Vmaf {
        vmaf_args: vec!["log_path=output.xml".into()],
        ..<_>::default()
    };
    let expected = format!(
        "[0:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:log_path=output.xml:n_threads={}",
        thread::available_parallelism().map_or(1, |p| p.get())
    );
    assert_eq!(
        vmaf.ffmpeg_lavfi(None, Some(PixelFormat::Yuv420p), None),
        expected
    );
}

/// Low resolution videos should be upscaled to 1080p
#[test]
fn vmaf_lavfi_small_width_upscale_to_1k() {
    let vmaf = Vmaf {
        vmaf_args: vec!["n_threads=5".into(), "n_subsample=4".into()],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((1280, 720)), Some(PixelFormat::Yuv420p), None),
        "[0:v]format=yuv420p,scale=1920:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,scale=1920:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads=5:n_subsample=4"
    );
}

#[test]
fn vmaf_lavfi_v1_0_16_small_width_upscale_to_1k() {
    for v in [
        "vmaf_v1.0.16_3d0h",
        "vmaf_v1.0.16_hfr_3d0h",
        "vmaf_v1.0.16_5d0h",
        "vmaf_v1.0.16_hfr_5d0h",
    ] {
        let vmaf = Vmaf {
            vmaf_args: vec!["n_threads=5".into(), format!("model=version={v}").into()],
            ..<_>::default()
        };
        assert_eq!(
            vmaf.ffmpeg_lavfi(Some((1280, 720)), Some(PixelFormat::Yuv420p), None),
            format!(
                "[0:v]format=yuv420p,scale=1920:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[dis];\
                 [1:v]format=yuv420p,scale=1920:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[ref];\
                 [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads=5:model=version={v}"
            ),
            "{v}: {vmaf:#?}",
        );
    }
}

/// 4k videos should use 4k model
#[test]
fn vmaf_lavfi_4k() {
    let vmaf = Vmaf {
        vmaf_args: vec!["n_threads=5".into(), "n_subsample=4".into()],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((3840, 2160)), Some(PixelFormat::Yuv420p), None),
        "[0:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads=5:n_subsample=4:model=version=vmaf_4k_v0.6.1"
    );
}

/// >2k videos should be upscaled to 4k & use 4k model
#[test]
fn vmaf_lavfi_3k_upscale_to_4k() {
    let vmaf = Vmaf {
        vmaf_args: vec!["n_threads=5".into()],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((3008, 1692)), Some(PixelFormat::Yuv420p), None),
        "[0:v]format=yuv420p,scale=3840:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,scale=3840:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads=5:model=version=vmaf_4k_v0.6.1"
    );
}

#[test]
fn vmaf_lavfi_v1_0_16_3k_upscale_to_4k() {
    for v in [
        "vmaf_v1.0.16_1d5h_2160",
        "vmaf_v1.0.16_hfr_1d5h_2160",
        "vmaf_v1.0.16_3d0h_2160",
        "vmaf_v1.0.16_hfr_3d0h_2160",
    ] {
        let vmaf = Vmaf {
            vmaf_args: vec!["n_threads=5".into(), format!("model=version={v}").into()],
            ..<_>::default()
        };
        assert_eq!(
            vmaf.ffmpeg_lavfi(Some((3008, 1692)), Some(PixelFormat::Yuv420p), None),
            format!(
                "[0:v]format=yuv420p,scale=3840:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[dis];\
                 [1:v]format=yuv420p,scale=3840:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[ref];\
                 [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads=5:model=version={v}"
            ),
            "{v}: {vmaf:#?}",
        );
    }
}

/// If user has overridden the model, don't default a vmaf width
#[test]
fn vmaf_lavfi_small_width_custom_model() {
    let vmaf = Vmaf {
        vmaf_args: vec![
            "model=version=foo".into(),
            "n_threads=5".into(),
            "n_subsample=4".into(),
        ],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((1280, 720)), Some(PixelFormat::Yuv420p), None),
        "[0:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:model=version=foo:n_threads=5:n_subsample=4"
    );
}

#[test]
fn vmaf_lavfi_custom_model_and_width() {
    let vmaf = Vmaf {
        vmaf_args: vec![
            "model=version=foo".into(),
            "n_threads=5".into(),
            "n_subsample=4".into(),
        ],
        // if specified just do it
        vmaf_scale: VmafScale::Custom {
            width: 123,
            height: 720,
        },
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((1280, 720)), Some(PixelFormat::Yuv420p), None),
        "[0:v]format=yuv420p,scale=123:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,scale=123:-1:flags=bicubic,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:model=version=foo:n_threads=5:n_subsample=4"
    );
}

#[test]
fn vmaf_lavfi_1080p() {
    let vmaf = Vmaf {
        vmaf_args: vec!["n_threads=5".into(), "n_subsample=4".into()],
        ..<_>::default()
    };
    assert_eq!(
        vmaf.ffmpeg_lavfi(Some((1920, 1080)), Some(PixelFormat::Yuv420p), None),
        "[0:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[dis];\
         [1:v]format=yuv420p,setpts=PTS-STARTPTS,settb=AVTB[ref];\
         [dis][ref]libvmaf=shortest=true:ts_sync_mode=nearest:n_threads=5:n_subsample=4"
    );
}
