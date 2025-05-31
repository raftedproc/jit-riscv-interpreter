#[repr(C)]
#[derive(Default)]
pub struct Cpu {
    pub regs: [u32; 32],
    pub pc:   u32,
    pub mem:  Vec<u8>,
}
impl Cpu {
    pub fn new(code: &[u8]) -> Self {
        let mut m = vec![0; 64 * 1024];
        m[..code.len()].copy_from_slice(code);
        Self { mem: m, ..Default::default() }
    }
    pub fn load32(&self, addr: u32) -> u32 {
        let i = addr as usize;
        // println!("load32 addr {} val {:x}", i, u32::from_le_bytes(self.mem[i..i + 4].try_into().unwrap()));
        u32::from_le_bytes(self.mem[i..i + 4].try_into().unwrap())
    }
    pub fn store32(&mut self, addr: u32, val: u32) {
        let i = addr as usize;
        // println!("store32 addr {} val {:x}", i, val);
        self.mem[i..i + 4].copy_from_slice(&val.to_le_bytes());
    }
}