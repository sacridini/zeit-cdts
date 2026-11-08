//! Sentinel-2 Level 1C for the atmospheric correction: what FORCE's `force-l2ps` does before the correction to turn a
//! granule into the digital numbers, the metadata and the quality bits `Atc::allocate` and `radiometric_correction`
//! take. Ported from FORCE 3.11.00 (Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later,
//! https://github.com/davidfrantz/force):
//!
//! * [`parse_product`] and [`parse_tile`]: `parse_metadata_sentinel2` of `meta-ll.c` (`parse_metadata` for a Sentinel-2
//!   granule), with `interpolate_sunview_grid` and `collapse_view_grid`. The C reads two XML files line by line and
//!   splits every line into words at ` ="<>`, then looks at each pair of neighbouring words (a tag and its value); the
//!   functions do the same on strings, so that what the C accepts and how it reads it is the same. [`parse_metadata`]
//!   runs both and builds the [`Meta`], the geometry of the image and the [`Dn`] of the C's `meta_t` and `DN` brick.
//!   The product file (`MTD_MSIL1C.xml`) gives the spacecraft (the response function bands), the start time, the
//!   quantification value and, since processing baseline 04.00, the additive offset of each band (`RADIO_ADD_OFFSET`);
//!   the tile file (`MTD_TL.xml`) gives the size, resolution and upper left corner of the image and the grids of the sun
//!   and view angles, from which the image is clipped to the valid cells.
//! * [`bounds_level1`]: `bounds_level1` of `read-ll.c` (the initial QAI: no-data and saturation, with the optional
//!   buffer of the no-data pixels).
//! * [`convert_level1`]: the Sentinel-2 branch of `convert_level1` of `read-ll.c` (DN to the `short` TOA reflectance).
//! * [`L1c::read_window`]: the window `read_level1` reads from a band of a given resolution.
//!
//! Not here: the reading of the rasters (`read_level1` opens the 13 JPEG 2000 files with GDAL, reads the window of each
//! band and resamples it to the grid of the finest band by nearest neighbour; the functions take the bands already
//! there), the projection of the image (the C gets it from the first band file; the EPSG code of the tile file is in
//! [`TileMeta::epsg`]), and the Landsat branch.
//!
//! What the C leaves out: the offset of baseline 04.00 is added to the DN (`dn + radd`) but the result is not clipped,
//! so a DN below 1000 becomes a negative TOA reflectance (the `short` is negative; FORCE does not flag it), and the
//! DN 0 that Sentinel-2 uses for no-data is still the no-data of the QAI. The C does not check the baseline: any
//! `RADIO_ADD_OFFSET` is read, an older product has none (the offset is 0).

use crate::atc::{Dn, FORCE_NO_DATA, SVGRID};
use crate::l2::c_short;
use crate::meta::{Cal, Meta, S2Meta};
use crate::parallax;
use crate::qai;
use crate::radtran::{self, md2doy, Date};

/// Number of bands of Sentinel-2 and of detectors (`nb`, `nd` of `parse_metadata_sentinel2`).
pub const NB: usize = 13;
pub const ND: usize = 12;
/// The saturation value of the digital numbers (`meta->sat`) and their size in bits (`meta->dtype`).
pub const SATURATION: i32 = 65535;
pub const DTYPE: i32 = 16;
/// The scale of the TOA reflectance (`set_brick_scale(TOA, b, 10000)`).
pub const TOA_SCALE: f32 = 10000.0;
/// The domain of each band, in the order of the C.
pub const DOMAINS: [&str; NB] = [
    "ULTRABLUE",
    "BLUE",
    "GREEN",
    "RED",
    "REDEDGE1",
    "REDEDGE2",
    "REDEDGE3",
    "BROADNIR",
    "NIR",
    "VAPOR",
    "CIRRUS",
    "SWIR1",
    "SWIR2",
];
/// The band names of the original files (`orig_band`).
pub const ORIG_BANDS: [&str; NB] = [
    "01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "10", "11", "12",
];

/// An error of the metadata or of the arrays.
#[derive(Debug, Clone, PartialEq)]
pub enum S2L1cError {
    /// The FORCE `return FAILURE` (or a crash of the C on a truncated line) with its message.
    Metadata(String),
    /// "no valid cell after subsetting": FORCE stops the run (`exit(SUCCESS)`).
    NoValidCell,
    /// An array of the wrong size.
    Shape(String),
}

impl std::fmt::Display for S2L1cError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            S2L1cError::Metadata(m) => write!(f, "Sentinel-2 metadata: {m}"),
            S2L1cError::NoValidCell => write!(f, "no valid cell after subsetting the angle grids"),
            S2L1cError::Shape(m) => write!(f, "{m}"),
        }
    }
}
impl std::error::Error for S2L1cError {}

fn bad<T>(m: &str) -> Result<T, S2L1cError> {
    Err(S2L1cError::Metadata(m.to_string()))
}

// ---------------------------------------------------------------------------------------------------------------
// the C's string functions
// ---------------------------------------------------------------------------------------------------------------

/// The lines as `fgets(buffer, NPOW_13, fp)` reads them: at most 8191 bytes, ending after a newline.
fn c_lines(xml: &[u8]) -> impl Iterator<Item = &[u8]> {
    const NPOW_13: usize = 8192;
    let mut pos = 0usize;
    std::iter::from_fn(move || {
        if pos >= xml.len() {
            return None;
        }
        let end = (pos + NPOW_13 - 1).min(xml.len());
        let stop = xml[pos..end]
            .iter()
            .position(|&c| c == b'\n')
            .map_or(end, |k| pos + k + 1);
        let line = &xml[pos..stop];
        pos = stop;
        Some(line)
    })
}

/// `strtok_r` with the separators ` ="<>` (a newline is not one: it ends up in the last word of the line).
fn words(line: &[u8]) -> Vec<&[u8]> {
    line.split(|c| matches!(c, b' ' | b'=' | b'"' | b'<' | b'>'))
        .filter(|w| !w.is_empty())
        .collect()
}

/// `atoi`.
fn atoi(s: &[u8]) -> i32 {
    let mut i = 0;
    while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
        i += 1;
    }
    let mut neg = false;
    if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
        neg = s[i] == b'-';
        i += 1;
    }
    let mut v: i64 = 0;
    while i < s.len() && s[i].is_ascii_digit() {
        v = v.wrapping_mul(10).wrapping_add((s[i] - b'0') as i64);
        i += 1;
    }
    (if neg { -v } else { v }) as i32
}

/// `atof`: the longest prefix that is a number (`strtod`), 0 if there is none.
fn atof(s: &[u8]) -> f64 {
    let mut i = 0;
    while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
        i += 1;
    }
    let start = i;
    if i < s.len() && (s[i] == b'-' || s[i] == b'+') {
        i += 1;
    }
    let lower: Vec<u8> = s[i..].iter().take(8).map(|c| c.to_ascii_lowercase()).collect();
    if lower.starts_with(b"infinity") {
        i += 8;
    } else if lower.starts_with(b"inf") || lower.starts_with(b"nan") {
        i += 3;
    } else {
        let digits = |mut k: usize| {
            while k < s.len() && s[k].is_ascii_digit() {
                k += 1;
            }
            k
        };
        let int_end = digits(i);
        let mut end = int_end;
        let mut any = int_end > i;
        if end < s.len() && s[end] == b'.' {
            let frac_end = digits(end + 1);
            any |= frac_end > end + 1;
            if any {
                end = frac_end;
            }
        }
        if !any {
            return 0.0;
        }
        if end < s.len() && (s[end] == b'e' || s[end] == b'E') {
            let mut k = end + 1;
            if k < s.len() && (s[k] == b'-' || s[k] == b'+') {
                k += 1;
            }
            let e_end = digits(k);
            if e_end > k {
                end = e_end;
            }
        }
        i = end;
    }
    std::str::from_utf8(&s[start..i])
        .ok()
        .and_then(|t| t.parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// `char_to_int` (`strtol` in base 0 on the whole word): `None` if it fails.
fn char_to_int(s: &[u8]) -> Option<i32> {
    let t = std::str::from_utf8(s).ok()?;
    let (neg, u) = match t.as_bytes().first()? {
        b'-' => (true, &t[1..]),
        b'+' => (false, &t[1..]),
        _ => (false, t),
    };
    let (radix, digits) = if let Some(h) = u.strip_prefix("0x").or_else(|| u.strip_prefix("0X")) {
        (16, h)
    } else if u.len() > 1 && u.starts_with('0') {
        (8, &u[1..])
    } else {
        (10, u)
    };
    let v = i64::from_str_radix(digits, radix).ok()?;
    i32::try_from(if neg { -v } else { v }).ok()
}

/// `char_to_float` (`strtof` on the whole word): `None` if it fails.
fn char_to_float(s: &[u8]) -> Option<f32> {
    std::str::from_utf8(s)
        .ok()?
        .parse::<f32>()
        .ok()
        .filter(|v| v.is_finite())
}

/// The words of a line and the position of the C's `tokenptr`.
struct Words<'a> {
    w: Vec<&'a [u8]>,
    pos: usize,
}

impl<'a> Words<'a> {
    /// `strtok_r(NULL, separator, &saveptr)`.
    fn next(&mut self) -> Option<&'a [u8]> {
        let r = self.w.get(self.pos).copied();
        self.pos += 1;
        r
    }
}

/// The word of the C's `tokenptr` where the C does not test for NULL (and would crash).
fn need<'a>(w: Option<&'a [u8]>) -> Result<&'a [u8], S2L1cError> {
    w.map_or_else(|| bad("a line ends where a value was expected"), Ok)
}

// ---------------------------------------------------------------------------------------------------------------
// the product file
// ---------------------------------------------------------------------------------------------------------------

/// What the top-level file `MTD_MSIL1C.xml` gives.
#[derive(Clone, Debug, PartialEq)]
pub struct ProductMeta {
    /// `"SEN2A"`, `"SEN2B"`, `"SEN2C"` or `"SEN2D"` (`set_brick_sensor`).
    pub sensor: String,
    /// The first band of the spectral response table for this sensor (`b_rsr`, `_RSR_START_SEN2*_`).
    pub rsr_start: usize,
    /// `PROCESSING_BASELINE` (not read by FORCE), e.g. `"05.09"`, if the file has it.
    pub baseline: Option<String>,
    /// `QUANTIFICATION_VALUE`: the scale of the digital numbers (`set_brick_scale(DN, b, ...)`, 10000 or, in the earliest
    /// products, 1000).
    pub dn_scale: f32,
    /// `RADIO_ADD_OFFSET` of each band (`cal[b].radd`; 0 for a file without them, i.e. before baseline 04.00).
    pub radd: [f32; NB],
    /// The start of the product (`PRODUCT_START_TIME`; the time zone is 0).
    pub date: Date,
}

/// `parse_metadata_sentinel2`, top-level file (`MTD_MSIL1C.xml`).
pub fn parse_product(xml: &[u8]) -> Result<ProductMeta, S2L1cError> {
    let mut sensor: Option<(String, usize)> = None;
    let mut baseline: Option<String> = None;
    let mut dn_scale: Option<f32> = None;
    let mut radd = [0.0f32; NB];
    let mut date: Option<Date> = None;

    for line in c_lines(xml) {
        let mut ws = Words { w: words(line), pos: 0 };
        let Some(first) = ws.next() else { continue };
        let mut tag = first;
        let mut cur = Some(first);
        while cur.is_some() {
            cur = ws.next();
            if tag == b"SPACECRAFT_NAME" {
                let value = need(cur)?;
                // strtok_r(tokenptr, "-"): the second word is the number of the spacecraft
                let mut parts = value.split(|&c| c == b'-').filter(|p| !p.is_empty());
                let _ = parts.next();
                let id = parts.next().map_or_else(|| bad("SPACECRAFT_NAME has no number"), Ok)?;
                if atoi(id) != 2 {
                    return bad("unknown/unsupported sensor ID!");
                }
                let name = format!("SEN{}", String::from_utf8_lossy(id));
                let start = match name.as_str() {
                    "SEN2A" => radtran::RSR_START_SEN2A,
                    "SEN2B" => radtran::RSR_START_SEN2B,
                    "SEN2C" => radtran::RSR_START_SEN2C,
                    "SEN2D" => radtran::RSR_START_SEN2D,
                    _ => return bad("unknown/unsupported Sentinel-2!"),
                };
                sensor = Some((name, start));
            } else if tag == b"PROCESSING_LEVEL" {
                if !need(cur)?.windows(2).any(|w| w == b"1C") {
                    return bad("Unknown processing level.");
                }
            } else if tag == b"PROCESSING_BASELINE" {
                baseline = cur.map(|c| String::from_utf8_lossy(c).into_owned());
            } else if tag == b"PRODUCT_START_TIME" {
                let value = need(cur)?;
                let mut p = value
                    .split(|c| matches!(c, b'-' | b'T' | b':' | b'.'))
                    .filter(|p| !p.is_empty());
                let mut field = || {
                    p.next()
                        .map(atoi)
                        .map_or_else(|| bad("PRODUCT_START_TIME is incomplete"), Ok)
                };
                let (year, month, day) = (field()?, field()?, field()?);
                let (hh, mm, ss) = (field()?, field()?, field()?);
                date = Some(Date {
                    year,
                    month,
                    day,
                    doy: md2doy(month, day),
                    hh,
                    mm,
                    ss,
                    tz: 0,
                });
            } else if tag == b"QUANTIFICATION_VALUE" {
                let mut value = need(cur)?;
                dn_scale = Some(atoi(value) as f32);
                while value != b"/QUANTIFICATION_VALUE" {
                    if atoi(value) != 0 {
                        dn_scale = Some(atoi(value) as f32);
                    }
                    value = need(ws.next())?;
                }
                cur = Some(value);
            } else if tag == b"RADIO_ADD_OFFSET" {
                let mut value = need(cur)?;
                let mut b: i32 = -1;
                while value != b"/RADIO_ADD_OFFSET" {
                    if b < 0 {
                        if let Some(v) = char_to_int(value) {
                            b = v;
                        }
                    } else if (b as usize) < NB {
                        // the C writes to radd[b] whatever b is
                        if let Some(v) = char_to_float(value) {
                            radd[b as usize] = v;
                        }
                    }
                    value = need(ws.next())?;
                }
                cur = Some(value);
            }
            // in case tag (key words) is not the first word in a line
            tag = match cur {
                Some(c) => c,
                None => break,
            };
        }
    }

    let Some((sensor, rsr_start)) = sensor else {
        return bad("no SPACECRAFT_NAME");
    };
    // a scale of 0 would divide by zero in `convert_level1`
    let Some(dn_scale) = dn_scale.filter(|&s| s != 0.0) else {
        return bad("no QUANTIFICATION_VALUE");
    };
    let Some(date) = date else {
        return bad("no PRODUCT_START_TIME");
    };
    Ok(ProductMeta {
        sensor,
        rsr_start,
        baseline,
        dn_scale,
        radd,
        date,
    })
}

// ---------------------------------------------------------------------------------------------------------------
// the tile file
// ---------------------------------------------------------------------------------------------------------------

/// What the granule file `MTD_TL.xml` gives, after the image has been clipped to the valid cells of the angle grids.
#[derive(Clone, Debug)]
pub struct TileMeta {
    /// Resolution of the finest band in metres (`get_brick_res`) and the size of the whole tile at that resolution
    /// (`NCOLS`, `NROWS` of the finest resolution).
    pub res: f64,
    pub tile_ncols: i32,
    pub tile_nrows: i32,
    /// The upper left corner (`ULX`, `ULY`, the last `Geoposition` of the file), of the tile and of the clipped image.
    pub tile_ulx: f64,
    pub tile_uly: f64,
    /// The image after the clipping: columns, rows and upper left corner (`set_brick_ncols`, `_nrows`, `_ulx`, `_uly`).
    pub ncols: i32,
    pub nrows: i32,
    pub ulx: f64,
    pub uly: f64,
    /// The window of the tile that is the image, in pixels of the finest band (`s2.left`, `top`, `right`, `bottom`).
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
    /// The grids of the view angles (`s2.vzen`, `vazi`, `nx`, `ny`, `nodata`: all bands and detectors averaged) for the
    /// atmospheric correction.
    pub s2: S2Meta,
    /// The grids of the sun angles on the same cells (`s2.szen`, `sazi`).
    pub szen: Vec<f32>,
    pub sazi: Vec<f32>,
    /// `HORIZONTAL_CS_CODE` of the file (e.g. 32723 for `EPSG:32723`; not read by FORCE, which takes the projection from
    /// the band files) and the tile identifier (`TILE_ID`, e.g. `S2B_OPER_MSI_L1C_TL_..._T23LKC_N05.09`).
    pub epsg: Option<u32>,
    pub tile_id: Option<String>,
}

/// `interpolate_sunview_grid`: the grid at the corners of the cells (`int_nx` x `int_ny`, the C indexes it with
/// `int_ny` as the width) to one value per cell, the mean of the corners that are positive (`nodata` if none is).
fn interpolate_sunview_grid(grid: &mut Vec<f32>, int_nx: usize, int_ny: usize, nodata: f32) {
    let (cell_nx, cell_ny) = (int_nx - 1, int_ny - 1);
    let mut cells = vec![0.0f32; cell_ny * cell_nx];
    let mut p = 0;
    for i in 0..cell_ny {
        for j in 0..cell_nx {
            let (mut sum, mut num) = (0.0f32, 0.0f32);
            for ii in 0..=1 {
                for jj in 0..=1 {
                    let v = grid[(i + ii) * int_ny + (j + jj)];
                    if v > 0.0 {
                        sum += v;
                        num += 1.0;
                    }
                }
            }
            cells[p] = if num > 0.0 { sum / num } else { nodata };
            p += 1;
        }
    }
    *grid = cells;
}

/// `collapse_view_grid`: the mean of the grids of all bands and detectors in the first one (`grid[0][0]`).
fn collapse_view_grid(grid: &mut [Vec<f32>], nx: usize, ny: usize, nodata: f32) {
    for p in 0..nx * ny {
        let (mut sum, mut num) = (0.0f32, 0.0f32);
        for g in grid.iter() {
            if g[p] != nodata {
                sum += g[p];
                num += 1.0;
            }
        }
        grid[0][p] = if num > 0.0 { sum / num } else { nodata };
    }
}

/// `parse_metadata_sentinel2`, granule file (`MTD_TL.xml`). `subset` is `doreproj || dotile`: the image is clipped to the
/// valid cells of the view grid (rounded out to multiples of 60 m), else it is the whole tile.
pub fn parse_tile(xml: &[u8], subset: bool) -> Result<TileMeta, S2L1cError> {
    let svgrid = SVGRID;
    // the DN brick: columns, rows, resolution, corner (`allocate_brick`, `set_brick_*`)
    let (mut b_nx, mut b_ny, mut b_res) = (1i32, 1i32, i32::MAX as f64);
    let (mut b_ulx, mut b_uly) = (0.0f64, 0.0f64);
    let nodata = i16::MIN as f32; // meta->s2.nodata = meta->fill

    let (mut sv_nx, mut sv_ny) = (0i32, 0i32);
    let mut grids: Option<(Vec<f32>, Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>)> = None;
    let (mut s, mut v, mut z, mut a) = (false, false, false, false);
    let (mut i, mut b, mut d) = (0i32, 0i32, 0i32);
    let mut epsg: Option<u32> = None;
    let mut tile_id: Option<String> = None;

    for line in c_lines(xml) {
        let mut ws = Words { w: words(line), pos: 0 };
        let Some(first) = ws.next() else { continue };
        let mut tag = first;
        let mut cur = Some(first);
        while cur.is_some() {
            cur = ws.next();
            if tag == b"resolution" {
                let r = atoi(need(cur)?);
                if (r as f64) < b_res {
                    b_res = r as f64;
                }
            } else if tag == b"NROWS" {
                let n = atoi(need(cur)?);
                if n > b_ny {
                    b_ny = n;
                }
            } else if tag == b"NCOLS" {
                let n = atoi(need(cur)?);
                if n > b_nx {
                    b_nx = n;
                }
            } else if tag == b"ULX" {
                b_ulx = atof(need(cur)?);
            } else if tag == b"ULY" {
                b_uly = atof(need(cur)?);
            } else if tag == b"HORIZONTAL_CS_CODE" {
                epsg = cur
                    .and_then(|c| c.strip_prefix(b"EPSG:"))
                    .and_then(|c| std::str::from_utf8(c).ok()?.parse().ok());
            } else if tag == b"TILE_ID" {
                // the element has an attribute (`metadataLevel="Brief"`): the value is the last word before the end tag
                let mut value = need(cur)?;
                while let Some(w) = ws.next() {
                    if w == b"/TILE_ID" {
                        break;
                    }
                    value = w;
                }
                cur = Some(value);
                tile_id = Some(String::from_utf8_lossy(value).into_owned());
            } else if tag == b"COL_STEP" {
                let mut value = need(cur)?;
                while atoi(value) == 0 {
                    value = need(ws.next())?;
                }
                cur = Some(value);
                if svgrid != atoi(value) {
                    return bad("SUN_VIEW_GRID is incompatible with Sentinel-2 metadata.");
                }
                let width = b_nx as f64 * b_res;
                let height = b_ny as f64 * b_res;
                sv_nx = (width / (svgrid as f32) as f64).ceil() as i32 + 1;
                sv_ny = (height / (svgrid as f32) as f64).ceil() as i32 + 1;
                if grids.is_none() {
                    let n = (sv_ny * sv_nx) as usize;
                    grids = Some((
                        vec![0.0; n],
                        vec![0.0; n],
                        vec![vec![0.0; n]; NB * ND],
                        vec![vec![0.0; n]; NB * ND],
                    ));
                }
            } else if tag == b"Sun_Angles_Grid" {
                s = true;
            } else if tag == b"/Sun_Angles_Grid" {
                s = false;
            } else if tag == b"Viewing_Incidence_Angles_Grids" {
                v = true;
                b = 0;
                d = 0;
                while let Some(mut value) = cur {
                    if value == b"bandId" {
                        value = need(ws.next())?;
                        b = atoi(value);
                    }
                    if value == b"detectorId" {
                        value = need(ws.next())?;
                        d = atoi(value) - 1;
                    }
                    let _ = value;
                    cur = ws.next();
                }
            } else if tag == b"/Viewing_Incidence_Angles_Grids" {
                v = false;
            } else if tag == b"Zenith" {
                z = true;
                i = 0;
            } else if tag == b"/Zenith" {
                z = false;
            } else if tag == b"Azimuth" {
                a = true;
                i = 0;
            } else if tag == b"/Azimuth" {
                a = false;
            } else if tag == b"VALUES" && i < sv_ny {
                let mut value = need(cur)?;
                let mut j = 0i32;
                while value != b"/VALUES" && j < sv_nx {
                    let Some((s_sz, s_sa, s_vz, s_va)) = grids.as_mut() else {
                        return bad("VALUES before COL_STEP");
                    };
                    let k = (i * sv_ny + j) as usize;
                    if value != b"NaN" {
                        let x = atof(value) as f32;
                        if s {
                            if z {
                                s_sz[k] = x;
                            } else if a {
                                s_sa[k] = x;
                            }
                        } else if v {
                            if !(0..NB as i32).contains(&b) || !(0..ND as i32).contains(&d) {
                                return bad("bandId or detectorId out of range");
                            }
                            let g = (b as usize) * ND + d as usize;
                            if z {
                                s_vz[g][k] += x;
                            } else if a {
                                s_va[g][k] += x;
                            }
                        }
                    }
                    value = need(ws.next())?;
                    j += 1;
                }
                cur = Some(value);
                i += 1;
            }
            // in case tag (key words) is not the first word in a line
            tag = match cur {
                Some(c) => c,
                None => break,
            };
        }
    }

    let Some((mut s_sz, mut s_sa, mut s_vz, mut s_va)) = grids else {
        return bad("no COL_STEP: no angle grids");
    };
    if sv_nx != sv_ny {
        return bad("the angle grids are not square, which the indexing of FORCE does not support");
    }
    let tile_ncols = b_nx;
    let tile_nrows = b_ny;

    // interpolate the grids: angles are given at grid intersections, we need one value per cell
    let (gx, gy) = (sv_nx as usize, sv_ny as usize);
    interpolate_sunview_grid(&mut s_sz, gx, gy, nodata);
    interpolate_sunview_grid(&mut s_sa, gx, gy, nodata);
    for g in s_vz.iter_mut() {
        interpolate_sunview_grid(g, gx, gy, nodata);
    }
    for g in s_va.iter_mut() {
        interpolate_sunview_grid(g, gx, gy, nodata);
    }
    sv_nx -= 1;
    sv_ny -= 1;
    let (cx, cy) = (sv_nx as usize, sv_ny as usize);
    collapse_view_grid(&mut s_vz, cx, cy, nodata);
    collapse_view_grid(&mut s_va, cx, cy, nodata);
    let (vz, va) = (&s_vz[0], &s_va[0]);

    // get image subset
    let (mut left, mut right, mut top, mut bottom) = (sv_nx - 1, 0i32, sv_ny - 1, 0i32);
    for i in 0..sv_ny {
        for j in 0..sv_nx {
            let k = (i * sv_ny + j) as usize;
            if vz[k] != nodata && va[k] != nodata {
                left = left.min(j);
                right = right.max(j);
                top = top.min(i);
                bottom = bottom.max(i);
            }
        }
    }
    right += 1; // lower-right corner of cell
    bottom += 1;
    if left > 0 {
        left -= 1; // one to the left to fill the missing left edge
    }
    while (left * svgrid) % 60 != 0 && left > 0 {
        left -= 1;
    }
    while (top * svgrid) % 60 != 0 && top > 0 {
        top -= 1;
    }
    while (right * svgrid) % 60 != 0 && right < (sv_nx - 1) {
        right += 1;
    }
    while (bottom * svgrid) % 60 != 0 && bottom < (sv_nx - 1) {
        bottom += 1;
    }
    if !subset {
        right = sv_nx;
        left = 0;
        bottom = sv_ny;
        top = 0;
    }

    let (nx, ny) = (right - left, bottom - top);
    if nx <= 0 || ny <= 0 {
        return Err(S2L1cError::NoValidCell);
    }
    let (nxu, nyu) = (nx as usize, ny as usize);
    let mut vzen = Vec::with_capacity(nxu * nyu);
    let mut vazi = Vec::with_capacity(nxu * nyu);
    let mut szen = Vec::with_capacity(nxu * nyu);
    let mut sazi = Vec::with_capacity(nxu * nyu);
    for i in 0..ny {
        for j in 0..nx {
            let k = ((i + top) * sv_ny + (j + left)) as usize;
            szen.push(s_sz[k]);
            sazi.push(s_sa[k]);
            vzen.push(vz[k]);
            vazi.push(va[k]);
        }
    }

    // the window in pixels (int * int / double, converted to int)
    let px = |cells: i32| (cells as f64 * svgrid as f64 / b_res) as i32;
    let (px_left, px_top) = (px(left), px(top));
    let mut px_right = px(right);
    let mut px_bottom = px(bottom);
    if px_right > b_nx {
        px_right = b_nx;
    }
    if px_bottom > b_ny {
        px_bottom = b_ny;
    }
    Ok(TileMeta {
        res: b_res,
        tile_ncols,
        tile_nrows,
        tile_ulx: b_ulx,
        tile_uly: b_uly,
        ncols: px_right - px_left,
        nrows: px_bottom - px_top,
        ulx: b_ulx + px_left as f64 * b_res,
        uly: b_uly + px_top as f64 * -b_res,
        left: px_left,
        top: px_top,
        right: px_right,
        bottom: px_bottom,
        s2: S2Meta {
            nodata,
            vzen,
            vazi,
            nx: nxu,
            ny: nyu,
        },
        szen,
        sazi,
        epsg,
        tile_id,
    })
}

/// A Sentinel-2 L1C granule as the correction takes it.
#[derive(Clone, Debug)]
pub struct L1c {
    pub product: ProductMeta,
    pub tile: TileMeta,
    /// `meta_t`: the response function band and the offset of each band, and the view angle grids.
    pub meta: Meta,
    /// The identifier of the tile in the name of the granule (`meta->refsys_id`, e.g. `T23LKC`).
    pub refsys_id: String,
}

/// `meta->refsys_id` from the name of the granule directory: characters 49 to 54 of the old, long name
/// (`S2B_OPER_MSI_L1C_TL_VGS2_20210728T151918_A022944_T23LKC_N03.01`), 4 to 9 of the compact one
/// (`L1C_T23LKC_A033753_20230823T095553`).
pub fn refsys_id(granule_name: &str) -> String {
    let b = granule_name.as_bytes();
    let from = if b.len() > 50 { 49 } else { 4 };
    let id = b.get(from..).unwrap_or(&[]);
    String::from_utf8_lossy(&id[..id.len().min(6)]).into_owned()
}

/// `parse_metadata` for a Sentinel-2 granule: the product file, the granule file and the name of the granule directory
/// (`b_level1`); `subset` is `doreproj || dotile` of the parameters.
pub fn parse_metadata(
    product_xml: &[u8],
    tile_xml: &[u8],
    granule_name: &str,
    subset: bool,
) -> Result<L1c, S2L1cError> {
    let product = parse_product(product_xml)?;
    let tile = parse_tile(tile_xml, subset)?;
    let cal = (0..NB)
        .map(|b| Cal {
            rsr_band: product.rsr_start + b,
            radd: product.radd[b],
        })
        .collect();
    let meta = Meta {
        cal,
        s2: tile.s2.clone(),
    };
    Ok(L1c {
        refsys_id: refsys_id(granule_name),
        product,
        tile,
        meta,
    })
}

impl L1c {
    /// The part of the DN brick that `Atc::allocate` reads: the size and resolution of the image, and the wavelength
    /// (um) and domain of each band.
    pub fn dn(&self) -> Dn {
        Dn {
            nx: self.tile.ncols as usize,
            ny: self.tile.nrows as usize,
            res: self.tile.res,
            wavelength: self.meta.cal.iter().map(|c| radtran::wavelength(c.rsr_band)).collect(),
            domain: DOMAINS.iter().map(|d| d.to_string()).collect(),
        }
    }

    /// The window `read_level1` reads from a band of resolution `band_res` (metres): `(xoff, yoff, nx, ny)` in its
    /// pixels, which GDAL resamples to the `ncols` x `nrows` of the image (nearest neighbour).
    pub fn read_window(&self, band_res: f64) -> (i32, i32, i32, i32) {
        let t = &self.tile;
        let f = |v: i32| (v as f64 * t.res / band_res).floor() as i32;
        (f(t.left), f(t.top), f(t.ncols), f(t.nrows))
    }

    /// The Level 1 steps on the bands in memory (`dn[b]`: `ncols * nrows` values of band `b`, on the grid of the image):
    /// the initial QAI and the TOA reflectance (`bounds_level1`, then `convert_level1`).
    pub fn bounds_and_convert(
        &self,
        dn: &[&[u16]],
        buffer_nodata: bool,
        n_jobs: i32,
    ) -> Result<(Vec<i16>, Vec<Vec<i16>>), S2L1cError> {
        let (nx, ny) = (self.tile.ncols as usize, self.tile.nrows as usize);
        let qai = bounds_level1(dn, nx, ny, buffer_nodata)?;
        let radd: Vec<f32> = self.meta.cal.iter().map(|c| c.radd).collect();
        let toa = convert_level1(dn, &qai, &radd, self.product.dn_scale, n_jobs)?;
        Ok((qai, toa))
    }
}

// ---------------------------------------------------------------------------------------------------------------
// read-ll.c
// ---------------------------------------------------------------------------------------------------------------

fn check_bands(dn: &[&[u16]], nc: usize) -> Result<(), S2L1cError> {
    if dn.len() != NB || dn.iter().any(|b| b.len() != nc) {
        return Err(S2L1cError::Shape(format!("expected {NB} bands of {nc} values")));
    }
    Ok(())
}

/// `bounds_level1` for Sentinel-2: the initial QAI (`short`, [`crate::qai`]'s bits) of an image of `nx` x `ny` pixels. A
/// pixel is off (no-data) where any band but the cirrus band is 0, and saturated where one is at least [`SATURATION`]
/// (checked band by band in order, and the search of a pixel stops at the first of either); the off flag is buffered
/// by one pixel with FORCE's midpoint circle (`buffer_`, which loses the off pixels of the border) if `buffer_nodata`.
/// The impulse noise filter is for the 8 bit data of Landsat and does not apply.
pub fn bounds_level1(dn: &[&[u16]], nx: usize, ny: usize, buffer_nodata: bool) -> Result<Vec<i16>, S2L1cError> {
    let nc = nx * ny;
    check_bands(dn, nc)?;
    let b_cirrus = DOMAINS.iter().position(|&d| d == "CIRRUS");
    let mut off = vec![false; nc];
    let mut qai = vec![0i16; nc];
    for p in 0..nc {
        for (b, band) in dn.iter().enumerate() {
            // if any layer void --> boundary
            if Some(b) != b_cirrus && band[p] == 0 {
                off[p] = true;
                break;
            }
            // if any layer saturated
            if band[p] as i32 >= SATURATION {
                qai::set_saturation(&mut qai[p], true);
                break;
            }
        }
    }
    // buffer one pixel (pixels are somehow contaminated. due to resampling?)
    if buffer_nodata {
        off = parallax::buffer(&off, nx, ny, 1, true, 1);
    }
    for (q, &o) in qai.iter_mut().zip(&off) {
        qai::set_off(q, o);
    }
    Ok(qai)
}

/// `convert_level1` for Sentinel-2: the TOA reflectance of each band, as `short`s scaled by [`TOA_SCALE`]: the no-data
/// value (-9999) where the QAI is off, else `(dn + radd) / dn_scale * 10000` in single precision, converted to `short` as
/// gcc does on x86-64 (the truncation; the low 16 bits of the 32 bit integer outside the range, see [`c_short`]). `radd`
/// is the offset of each band (0 before baseline 04.00, -1000 since) and `dn_scale` the quantification value.
pub fn convert_level1(
    dn: &[&[u16]],
    qai: &[i16],
    radd: &[f32],
    dn_scale: f32,
    n_jobs: i32,
) -> Result<Vec<Vec<i16>>, S2L1cError> {
    let nc = qai.len();
    check_bands(dn, nc)?;
    if radd.len() != NB {
        return Err(S2L1cError::Shape(format!("expected {NB} offsets")));
    }
    let nodata = FORCE_NO_DATA as i16;
    let mut toa = vec![vec![0i16; nc]; NB];
    for (b, out) in toa.iter_mut().enumerate() {
        let (band, r) = (dn[b], radd[b]);
        crate::parallel::for_each_row(
            out,
            4096,
            n_jobs,
            || (),
            |_, k, row| {
                let base = k * 4096;
                for (q, o) in row.iter_mut().enumerate() {
                    let p = base + q;
                    *o = if qai::off(qai[p]) {
                        nodata
                    } else {
                        c_short(((band[p] as f32 + r) / dn_scale * TOA_SCALE) as f64)
                    };
                }
            },
        );
    }
    Ok(toa)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A product file with the lines of a real one that FORCE reads.
    fn product(baseline: &str, craft: &str, offsets: bool) -> String {
        let mut s = String::new();
        s += "<?xml version=\"1.0\"?>\n<n1:Level-1C_User_Product>\n  <n1:General_Info>\n    <Product_Info>\n";
        s += "      <PRODUCT_START_TIME>2023-07-28T13:22:39.024Z</PRODUCT_START_TIME>\n";
        s += "      <PROCESSING_LEVEL>Level-1C</PROCESSING_LEVEL>\n";
        s += &format!("      <PROCESSING_BASELINE>{baseline}</PROCESSING_BASELINE>\n");
        s += &format!("      <Datatake datatakeIdentifier=\"GS2B_x\">\n  <SPACECRAFT_NAME>{craft}</SPACECRAFT_NAME>\n</Datatake>\n");
        s += "    </Product_Info>\n    <Product_Image_Characteristics>\n";
        s += "      <QUANTIFICATION_VALUES_LIST>\n        <QUANTIFICATION_VALUE unit=\"none\">10000</QUANTIFICATION_VALUE>\n";
        if offsets {
            s += "      <Radiometric_Offset_List>\n";
            for b in 0..13 {
                s += &format!("        <RADIO_ADD_OFFSET band_id=\"{b}\">-1000</RADIO_ADD_OFFSET>\n");
            }
            s += "      </Radiometric_Offset_List>\n";
        }
        s += "      </QUANTIFICATION_VALUES_LIST>\n    </Product_Image_Characteristics>\n  </n1:General_Info>\n</n1:Level-1C_User_Product>\n";
        s
    }

    /// A tile file of `n` x `n` pixels at 10 m with grids of `g` x `g` corners; the view zenith of the corner
    /// (`i`, `j`) is `vz(i, j)` (0 for none), the sun zenith 40 + i + j / 10.
    fn tile(n: i32, vz: &dyn Fn(usize, usize) -> f32, bands: &[(usize, usize)]) -> String {
        let g = ((n as f64 * 10.0 / 5000.0).ceil() as usize) + 1;
        let row = |f: &dyn Fn(usize, usize) -> f32, i: usize| {
            let v: Vec<String> = (0..g)
                .map(|j| {
                    let x = f(i, j);
                    if x == 0.0 {
                        "NaN".to_string()
                    } else {
                        format!("{x}")
                    }
                })
                .collect();
            format!("            <VALUES>{}</VALUES>\n", v.join(" "))
        };
        let grid = |f: &dyn Fn(usize, usize) -> f32| {
            let mut s = String::from("          <COL_STEP unit=\"m\">5000</COL_STEP>\n          <ROW_STEP unit=\"m\">5000</ROW_STEP>\n          <Values_List>\n");
            for i in 0..g {
                s += &row(f, i);
            }
            s + "          </Values_List>\n"
        };
        let mut s = String::from("<n1:Level-1C_Tile_ID>\n  <n1:General_Info>\n    <TILE_ID metadataLevel=\"Brief\">S2B_OPER_MSI_L1C_TL_2BPS_20230728T150149_A033383_T23LKC_N05.09</TILE_ID>\n  </n1:General_Info>\n  <n1:Geometric_Info>\n    <Tile_Geocoding>\n      <HORIZONTAL_CS_CODE>EPSG:32723</HORIZONTAL_CS_CODE>\n");
        for (r, k) in [(10, 1), (20, 2), (60, 6)] {
            s += &format!("      <Size resolution=\"{r}\">\n        <NROWS>{}</NROWS>\n        <NCOLS>{}</NCOLS>\n      </Size>\n", n / k, n / k);
        }
        for r in [10, 20, 60] {
            s += &format!("      <Geoposition resolution=\"{r}\">\n        <ULX>199980</ULX>\n        <ULY>8300020</ULY>\n        <XDIM>{r}</XDIM>\n        <YDIM>-{r}</YDIM>\n      </Geoposition>\n");
        }
        s += "    </Tile_Geocoding>\n    <Tile_Angles>\n      <Sun_Angles_Grid>\n        <Zenith>\n";
        s += &grid(&|i, j| 40.0 + i as f32 + j as f32 / 10.0);
        s += "        </Zenith>\n        <Azimuth>\n";
        s += &grid(&|i, j| 100.0 + i as f32 + j as f32 / 10.0);
        s += "        </Azimuth>\n      </Sun_Angles_Grid>\n";
        for &(b, d) in bands {
            s += &format!(
                "      <Viewing_Incidence_Angles_Grids bandId=\"{b}\" detectorId=\"{}\">\n        <Zenith>\n",
                d + 1
            );
            s += &grid(&|i, j| vz(i, j) + b as f32 * 0.01);
            s += "        </Zenith>\n        <Azimuth>\n";
            s += &grid(&|i, j| if vz(i, j) == 0.0 { 0.0 } else { 200.0 + j as f32 });
            s += "        </Azimuth>\n      </Viewing_Incidence_Angles_Grids>\n";
        }
        s + "    </Tile_Angles>\n  </n1:Geometric_Info>\n</n1:Level-1C_Tile_ID>\n"
    }

    #[test]
    fn the_product_file_gives_the_offsets_of_baseline_04_and_later() {
        let new = parse_product(product("05.09", "Sentinel-2B", true).as_bytes()).unwrap();
        assert_eq!(
            (new.sensor.as_str(), new.rsr_start, new.dn_scale),
            ("SEN2B", radtran::RSR_START_SEN2B, 10000.0)
        );
        assert_eq!(new.baseline.as_deref(), Some("05.09"));
        assert_eq!(new.radd, [-1000.0; NB]);
        assert_eq!(
            new.date,
            Date {
                year: 2023,
                month: 7,
                day: 28,
                doy: 209,
                hh: 13,
                mm: 22,
                ss: 39,
                tz: 0
            }
        );
        let old = parse_product(product("03.01", "Sentinel-2A", false).as_bytes()).unwrap();
        assert_eq!(
            (old.sensor.as_str(), old.rsr_start, old.radd),
            ("SEN2A", radtran::RSR_START_SEN2A, [0.0; NB])
        );
        assert_eq!(
            parse_product(product("04.00", "Sentinel-2C", true).as_bytes())
                .unwrap()
                .sensor,
            "SEN2C"
        );
    }

    #[test]
    fn the_product_file_is_refused_as_in_the_c() {
        assert!(parse_product(product("05.09", "Sentinel-1A", true).as_bytes()).is_err());
        assert!(parse_product(product("05.09", "Landsat-8", true).as_bytes()).is_err());
        assert!(parse_product(
            product("05.09", "Sentinel-2B", true)
                .replace("Level-1C", "Level-2A")
                .as_bytes()
        )
        .is_err());
        assert!(parse_product(product("05.09", "Sentinel-2B", true).replace("10000", "").as_bytes()).is_err());
    }

    #[test]
    fn the_tile_file_gives_the_geometry_and_the_clipped_image() {
        // a 10980 x 10980 tile at 10 m: 23 corners per side (22 cells of 5000 m, the last one partial); three (band,
        // detector) view grids, none with an angle in the corner column 3
        let all = [(0usize, 0usize), (3, 4), (12, 11)];
        let vz = |_i: usize, j: usize| if j == 3 { 0.0 } else { 5.0 + j as f32 };
        let whole = parse_tile(tile(10980, &vz, &all).as_bytes(), false).unwrap();
        assert_eq!((whole.tile_ncols, whole.tile_nrows, whole.res), (10980, 10980, 10.0));
        assert_eq!(
            (whole.tile_ulx, whole.tile_uly, whole.epsg),
            (199980.0, 8300020.0, Some(32723))
        );
        assert!(whole.tile_id.as_deref().unwrap().contains("T23LKC"));
        assert_eq!(
            (
                whole.s2.nx,
                whole.s2.ny,
                whole.ncols,
                whole.nrows,
                whole.left,
                whole.right
            ),
            (22, 22, 10980, 10980, 0, 10980)
        );
        assert_eq!(
            (whole.s2.nodata, whole.s2.vzen.len(), whole.szen.len()),
            (-32768.0, 22 * 22, 22 * 22)
        );
        // the sun zenith of cell (0, 0) is the mean of its 4 corners: 40, 40.1, 41, 41.1
        assert_eq!(whole.szen[0], (40.0f32 + 40.1 + 41.0 + 41.1) / 4.0);
        assert_eq!(whole.sazi[0], (100.0f32 + 100.1 + 101.0 + 101.1) / 4.0);
        // the view zenith is the mean of the three (band, detector) grids, each a mean of its corners
        assert!((whole.s2.vzen[0] - 5.5).abs() < 0.2, "{}", whole.s2.vzen[0]);

        // the image clipped to the valid cells: no angle from the corner column 21 on, so the cells 21 (and beyond) have
        // none and are nodata
        let hole = |_i: usize, j: usize| if j >= 21 { 0.0 } else { 5.0 };
        let clip = parse_tile(tile(10980, &hole, &all).as_bytes(), true).unwrap();
        // 21 cells of 5000 m = 10500 pixels
        assert_eq!(clip.s2.nx, 21);
        assert_eq!((clip.left, clip.top, clip.right, clip.bottom), (0, 0, 10500, 10980));
        assert_eq!(
            (clip.ncols, clip.nrows, clip.ulx, clip.uly),
            (10500, 10980, 199980.0, 8300020.0)
        );
        assert_eq!(clip.s2.vzen.len(), 21 * 22);
        // without the clipping, the whole tile
        let full = parse_tile(tile(10980, &hole, &all).as_bytes(), false).unwrap();
        assert_eq!((full.ncols, full.s2.nx), (10980, 22));
    }

    #[test]
    fn a_tile_without_valid_cells_is_refused() {
        let none = |_: usize, _: usize| 0.0;
        assert_eq!(
            parse_tile(tile(10980, &none, &[(0, 0)]).as_bytes(), true).unwrap_err(),
            S2L1cError::NoValidCell
        );
        assert!(parse_tile(b"<a>1</a>\n", false).is_err());
    }

    #[test]
    fn the_clipping_moves_the_corner() {
        // no view angle in the first 8 columns and 5 rows of corners: the first valid cell is (row 5, column 8) of the
        // 22 x 22 cells, minus one to the left, rounded down to a multiple of 60 m (3 cells), 7 -> 6 cells = 3000 m (the
        // top: 5 -> 3 cells = 1500 m); the same window FORCE gives to the real granule with these cells blanked
        // (benchmarks/vs_originals/force_l1.py)
        let vz = |i: usize, j: usize| if i < 5 || j < 8 { 0.0 } else { 7.0 };
        let c = parse_tile(tile(10980, &vz, &[(0, 0)]).as_bytes(), true).unwrap();
        assert_eq!((c.left, c.top, c.right, c.bottom), (3000, 1500, 10980, 10980));
        assert_eq!((c.ncols, c.nrows), (7980, 9480));
        assert_eq!((c.ulx, c.uly), (199980.0 + 3000.0 * 10.0, 8300020.0 - 1500.0 * 10.0));
        assert_eq!((c.s2.nx, c.s2.ny), (16, 19));
        // the cell (0, 0) of the clipped grid is the cell (3, 6) of the tile: its sun zenith is the mean of its corners
        assert!(
            (c.szen[0] - (43.6 + 43.7 + 44.6 + 44.7) / 4.0).abs() < 1e-4,
            "{}",
            c.szen[0]
        );
        let w = parse_tile(tile(10980, &vz, &[(0, 0)]).as_bytes(), false).unwrap();
        assert_eq!(
            (w.left, w.top, w.ncols, w.nrows, w.ulx, w.uly),
            (0, 0, 10980, 10980, 199980.0, 8300020.0)
        );
    }

    #[test]
    fn the_granule_name_gives_the_reference_system() {
        assert_eq!(
            refsys_id("S2B_OPER_MSI_L1C_TL_VGS2_20210728T151918_A022944_T23LKC_N03.01"),
            "T23LKC"
        );
        assert_eq!(refsys_id("L1C_T34UCF_A033753_20230823T095553"), "T34UCF");
        assert_eq!(refsys_id("x"), "");
    }

    #[test]
    fn the_words_are_the_cs_and_the_numbers_too() {
        assert_eq!(
            words(b"  <a b=\"1\">2</a>\n"),
            vec![&b"a"[..], b"b", b"1", b"2", b"/a", b"\n"]
        );
        assert_eq!((atoi(b" -12x"), atoi(b"x"), atoi(b"2B"), atoi(b"007")), (-12, 0, 2, 7));
        assert_eq!(
            (atof(b"3.5e2xyz"), atof(b".5"), atof(b"-1e"), atof(b"abc")),
            (350.0, 0.5, -1.0, 0.0)
        );
        assert_eq!(
            (
                char_to_int(b"13"),
                char_to_int(b"band_id"),
                char_to_int(b"08"),
                char_to_int(b"-1")
            ),
            (Some(13), None, None, Some(-1))
        );
        assert_eq!((char_to_float(b"-1000"), char_to_float(b"10x")), (Some(-1000.0), None));
        // fgets cuts at 8191 bytes
        let long = vec![b'a'; 20000];
        assert_eq!(
            c_lines(&long).map(|l| l.len()).collect::<Vec<_>>(),
            vec![8191, 8191, 3618]
        );
    }

    #[test]
    fn the_quality_bits_and_the_reflectance_of_the_digital_numbers() {
        let (nx, ny) = (6usize, 5usize);
        let nc = nx * ny;
        let mut bands: Vec<Vec<u16>> = (0..NB)
            .map(|b| (0..nc).map(|p| (1500 + 100 * b + p) as u16).collect())
            .collect();
        bands[0][7] = 0; // off
        bands[10][8] = 0; // the cirrus band may be 0
        bands[3][9] = 65535; // saturated
        bands[2][10] = 0; // off before the saturated band 3: off only
        bands[3][10] = 65535;
        bands[3][11] = 65535; // saturated, then a 0 that is not looked at
        bands[5][11] = 0;
        let view: Vec<&[u16]> = bands.iter().map(|b| b.as_slice()).collect();
        let qai = bounds_level1(&view, nx, ny, false).unwrap();
        let flags: Vec<(bool, bool)> = qai.iter().map(|&q| (qai::off(q), qai::saturation(q))).collect();
        assert_eq!(
            (flags[7], flags[8], flags[9], flags[10], flags[11]),
            (
                (true, false),
                (false, false),
                (false, true),
                (true, false),
                (false, true)
            )
        );
        assert_eq!(flags.iter().filter(|f| f.0 || f.1).count(), 4);
        // with the buffer, each off pixel (rows 1, columns 1 and 4: inside the border) takes its 4 neighbours
        let buf = bounds_level1(&view, nx, ny, true).unwrap();
        let off_buf: Vec<usize> = (0..nc).filter(|&p| qai::off(buf[p])).collect();
        assert_eq!(off_buf, vec![1, 4, 6, 7, 8, 9, 10, 11, 13, 16]);

        // the reflectance: baseline 03: (dn + 0) / 10000 * 10000; baseline 04: (dn - 1000)
        let toa3 = convert_level1(&view, &qai, &[0.0; NB], 10000.0, 1).unwrap();
        let toa5 = convert_level1(&view, &qai, &[-1000.0; NB], 10000.0, 2).unwrap();
        assert_eq!((toa3[0][0], toa5[0][0], toa3[1][0], toa5[1][0]), (1500, 500, 1600, 600));
        assert_eq!((toa3[0][7], toa5[3][10]), (-9999, -9999)); // off
        assert_eq!(toa5[3][9], c_short(((65535.0f32 - 1000.0) / 10000.0 * 10000.0) as f64)); // saturated: converted, not flagged off
                                                                                             // below the offset: negative
        bands[0][0] = 500;
        let view: Vec<&[u16]> = bands.iter().map(|b| b.as_slice()).collect();
        assert_eq!(
            convert_level1(&view, &qai, &[-1000.0; NB], 10000.0, 1).unwrap()[0][0],
            -500
        );
        // a scale of 1000 (the earliest products)
        assert_eq!(convert_level1(&view, &qai, &[0.0; NB], 1000.0, 1).unwrap()[1][0], 16000);
        assert!(convert_level1(&view[..12], &qai, &[0.0; NB], 1000.0, 1).is_err());
    }

    #[test]
    fn the_granule_feeds_the_atc() {
        let all = [(0usize, 0usize)];
        let l1 = parse_metadata(
            product("05.09", "Sentinel-2A", true).as_bytes(),
            tile(10980, &|_, _| 6.0, &all).as_bytes(),
            "L1C_T23LKC_A033753_x",
            false,
        )
        .unwrap();
        assert_eq!(l1.refsys_id, "T23LKC");
        let dn = l1.dn();
        assert_eq!(
            (dn.nx, dn.ny, dn.res, dn.wavelength.len(), dn.domain[8].as_str()),
            (10980, 10980, 10.0, 13, "NIR")
        );
        assert_eq!(
            (l1.meta.cal[0].rsr_band, l1.meta.cal[12].rsr_band, l1.meta.cal[3].radd),
            (30, 42, -1000.0)
        );
        assert!((dn.wavelength[3] - 0.665).abs() < 0.01, "{}", dn.wavelength[3]);
        assert_eq!(l1.read_window(20.0), (0, 0, 5490, 5490));
        assert_eq!(l1.read_window(60.0), (0, 0, 1830, 1830));
        assert_eq!((l1.meta.s2.nx, l1.meta.s2.ny, l1.meta.s2.vzen.len()), (22, 22, 484));
        let atc = crate::atc::Atc::allocate(&dn, &l1.meta).unwrap();
        assert_eq!((atc.nx, atc.ny), (22, 22));
    }
}
