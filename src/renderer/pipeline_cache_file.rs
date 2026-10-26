const MAGIC: [u8; 4] = *b"SSPC";

const FORMAT: u32 = 1;

const HEADER_LEN: usize = 4 + 4 + 8 + 8 + KEY_LEN;
const KEY_LEN: usize = 4 + 4 + 4 + 16 + 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheKey {
    pub vendor_id: u32,
    pub device_id: u32,
    pub driver_version: u32,

    pub cache_uuid: [u8; 16],

    pub build: u64,
}

impl CacheKey {
    fn bytes(&self) -> [u8; KEY_LEN] {
        let mut k = [0u8; KEY_LEN];
        k[0..4].copy_from_slice(&self.vendor_id.to_le_bytes());
        k[4..8].copy_from_slice(&self.device_id.to_le_bytes());
        k[8..12].copy_from_slice(&self.driver_version.to_le_bytes());
        k[12..28].copy_from_slice(&self.cache_uuid);
        k[28..36].copy_from_slice(&self.build.to_le_bytes());
        k
    }
}

fn checksum(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn encode(key: &CacheKey, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_LEN + data.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT.to_le_bytes());
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&checksum(data).to_le_bytes());
    out.extend_from_slice(&key.bytes());
    out.extend_from_slice(data);
    out
}

#[derive(Debug, PartialEq, Eq)]
pub enum Refused {
    NotACacheFile,
    OtherFormat(u32),

    WrongLength { said: u64, has: u64 },
    Corrupt,

    OtherKey,
}

pub fn decode<'a>(key: &CacheKey, file: &'a [u8]) -> Result<&'a [u8], Refused> {
    if file.len() < HEADER_LEN || file[0..4] != MAGIC {
        return Err(Refused::NotACacheFile);
    }
    let u32_at = |i: usize| u32::from_le_bytes(file[i..i + 4].try_into().unwrap());
    let u64_at = |i: usize| u64::from_le_bytes(file[i..i + 8].try_into().unwrap());
    let format = u32_at(4);
    if format != FORMAT {
        return Err(Refused::OtherFormat(format));
    }
    let data = &file[HEADER_LEN..];
    let said = u64_at(8);
    if said != data.len() as u64 {
        return Err(Refused::WrongLength { said, has: data.len() as u64 });
    }
    if u64_at(16) != checksum(data) {
        return Err(Refused::Corrupt);
    }
    if file[24..24 + KEY_LEN] != key.bytes() {
        return Err(Refused::OtherKey);
    }
    Ok(data)
}

pub fn write_whole(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

pub fn build_identity() -> u64 {
    loaded_from().map_or(0, |path| identity_of(&path))
}

fn identity_of(path: &str) -> u64 {

    let file = path.split('!').next().unwrap_or(path);
    let mut id = path.as_bytes().to_vec();
    if let Ok(meta) = std::fs::metadata(file) {
        id.extend_from_slice(&meta.len().to_le_bytes());
        if let Ok(t) = meta.modified() {
            if let Ok(d) = t.duration_since(std::time::UNIX_EPOCH) {
                id.extend_from_slice(&d.as_nanos().to_le_bytes());
            }
        }
    }
    checksum(&id)
}

fn loaded_from() -> Option<String> {
    #[repr(C)]
    struct DlInfo {
        dli_fname: *const std::ffi::c_char,
        dli_fbase: *mut std::ffi::c_void,
        dli_sname: *const std::ffi::c_char,
        dli_saddr: *mut std::ffi::c_void,
    }
    extern "C" {
        fn dladdr(addr: *const std::ffi::c_void, info: *mut DlInfo) -> std::ffi::c_int;
    }
    let mut info = DlInfo {
        dli_fname: std::ptr::null(),
        dli_fbase: std::ptr::null_mut(),
        dli_sname: std::ptr::null(),
        dli_saddr: std::ptr::null_mut(),
    };
    let ok = unsafe { dladdr(build_identity as *const std::ffi::c_void, &raw mut info) };
    if ok == 0 || info.dli_fname.is_null() {
        return None;
    }
    Some(unsafe { std::ffi::CStr::from_ptr(info.dli_fname) }.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> CacheKey {
        CacheKey { vendor_id: 0x5143, device_id: 0x4300_0001, driver_version: 7, cache_uuid: [9; 16], build: 42 }
    }

    #[test]
    fn a_written_cache_reads_back_as_the_drivers_bytes() {
        let data: Vec<u8> = (0..5000u32).map(|i| (i * 7) as u8).collect();
        assert_eq!(decode(&key(), &encode(&key(), &data)), Ok(&data[..]));
        assert_eq!(decode(&key(), &encode(&key(), &[])), Ok(&[][..]));
    }

    #[test]
    fn a_file_cut_short_or_damaged_is_refused() {
        let file = encode(&key(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(decode(&key(), &file[..file.len() - 3]), Err(Refused::WrongLength { said: 8, has: 5 }));
        assert_eq!(decode(&key(), &file[..10]), Err(Refused::NotACacheFile));
        let mut flipped = file.clone();
        *flipped.last_mut().unwrap() ^= 1;
        assert_eq!(decode(&key(), &flipped), Err(Refused::Corrupt));
        assert_eq!(decode(&key(), b"not a cache at all, just some text"), Err(Refused::NotACacheFile));
    }

    #[test]
    fn another_gpu_driver_or_build_starts_from_nothing() {
        let file = encode(&key(), &[1, 2, 3]);
        for other in [
            CacheKey { driver_version: 8, ..key() },
            CacheKey { device_id: 1, ..key() },
            CacheKey { cache_uuid: [8; 16], ..key() },
            CacheKey { build: 43, ..key() },
        ] {
            assert_eq!(decode(&other, &file), Err(Refused::OtherKey), "{other:?}");
        }
    }

    #[test]
    fn a_whole_write_replaces_the_file_and_leaves_no_temporary() {
        let dir = std::env::temp_dir().join(format!("sspc_{}_{:?}", std::process::id(), std::thread::current().id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pipelines.cache");
        write_whole(&path, b"first").unwrap();
        write_whole(&path, b"second").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"second");
        assert!(!path.with_extension("tmp").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn this_build_has_an_identity_and_it_is_stable() {
        let a = build_identity();
        assert_ne!(a, 0, "dladdr found no file for the running code");
        assert_eq!(a, build_identity());

        assert_eq!(identity_of("/nonexistent/base.apk!/lib/arm64-v8a/libx.so"), identity_of("/nonexistent/base.apk!/lib/arm64-v8a/libx.so"));
        assert_ne!(identity_of("/a/base.apk!/lib/x.so"), identity_of("/b/base.apk!/lib/x.so"));
    }
}
