mod cpu;
mod translator;

use cranelift_jit::{JITBuilder, JITModule};
use libc::{madvise, MADV_DONTNEED};
use memmap2::{MmapMut, MmapOptions};
use translator::{mem_load32, mem_store32};
use std::{ptr, thread::sleep, time::{Duration, Instant}};

use crate::cpu::Cpu;
use crate::translator::compile_tb;

/// Размер "виртуального" кэша: 4 ГиБ (2^30).
const LEN: usize = 1 << 32;
/// Размер «страницы» из точки зрения нашего кэша — 4 К (дробить меньше смысла).
const PAGE: usize = 64 * 1024;

// fn main() -> anyhow::Result<()> {
//     // 1) Резервируем 1 ГиБ VA. Физические страницы пока НЕ выделены.
//     let mut map: MmapMut = MmapOptions::new().len(LEN).map_anon()?;

//     println!("mmap created: {} MiB virtual ⇒ 0 MiB RAM (пока)", LEN / (1 << 20));

//     // 2) Пишем в ~16 случайных страниц; ОС «материализует» их лениво.
//     // let mut rng = SmallRng::seed_from_u64(0xCAFE_BABE);
//     let before = Instant::now();

//     for i in 0..16 {
//         let off = i * PAGE + 1;
//         map[off] = 42; // первая запись ⇒ page-fault ⇒ ядро отдаёт 64 К
//         println!("Touched page {:5} ⇒ commit +4 / 64 K", off / PAGE);
//         // sleep(Duration::from_secs(10));
//     }

//     println!("Done in {:.2?}", before.elapsed());
//     // sleep(Duration::from_secs(120));

//     Ok(())
// }

fn main() {
    // 1) tiny RISC-V программа: x1 = 1; x2 = 2; x3 = x1 + x2
    let code = [
        0x13, 0x05, 0x10, 0x00,     // addi x10, x0, 1
        0x13, 0x0a, 0x20, 0x00,     // addi x20, x0, 2
        0x33, 0x0e, 0xaa, 0x00,     // add  x28, x20, x10
        0x67, 0x80, 0x00, 0x00,     // jalr x0, 0(x0)   -> stop
    ];

    let mut cpu = Cpu::new(&code);
    let mut builder = JITBuilder::new(cranelift_module::default_libcall_names()).expect("failed to create JITBuilder");

    // регистрируем helpers
    builder.symbol("mem_load32",  mem_load32 as *const u8);
    builder.symbol("mem_store32", mem_store32 as *const u8);

    let mut jit     = JITModule::new(builder);

    // исполняем, переводя TB максимум по 16 инструкций
    loop {
        let (fn_ptr, insns) = compile_tb(&mut jit, &cpu, 16);
        let executor: extern "C" fn(*mut Cpu) -> u32 =
            unsafe { std::mem::transmute(fn_ptr) };
        let next_pc = executor(&mut cpu);
        println!("insns {}", insns);
        println!("x10 {}", cpu.regs[10]);
        println!("x20 {}", cpu.regs[20]);
        println!("x28 {}", cpu.regs[28]);
        if insns == 0 { break; }      // нет декодированных инструкций
        cpu.pc = next_pc;
        if next_pc == 0 { break; }    // наш demo JALR x0,0
    }

    println!("x28 = {}", cpu.regs[28]); // печатает 3
}