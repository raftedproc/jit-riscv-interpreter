mod cpu;
mod translator;

use cranelift_jit::{JITBuilder, JITModule};
use translator::{mem_load32, mem_store32};

use crate::cpu::Cpu;
use crate::translator::compile_tb;

const RANDOM_MATH: [u32; 104] = [
        0x13, 0x05, 0x10, 0x00, // addi x10, x0, 1
        0x13, 0x06, 0x20, 0x00, // addi x12, x0, 2
        0x13, 0x07, 0x30, 0x00, // addi x14, x0, 3
        0x13, 0x08, 0x40, 0x00, // addi x16, x0, 4
        0x13, 0x09, 0x50, 0x00, // addi x18, x0, 5
        // Math operations start here (this is instruction 5)
        0x33, 0x0e, 0xc5, 0x00, // add  x28, x10, x12
        0x33, 0x0f, 0xe5, 0x00, // add  x30, x10, x14
        0x33, 0x00, 0xc7, 0x01, // add  x0, x14, x12   // x0 always remains 0
        0x33, 0x01, 0x06, 0x01, // add  x2, x12, x16
        0x33, 0x81, 0xc6, 0x01, // sub  x3, x13, x12
        // Instructions 10-14
        0x33, 0x42, 0xd7, 0x01, // xor  x4, x15, x13
        0x33, 0x63, 0xc8, 0x00, // or   x6, x16, x12
        0x33, 0x74, 0xe9, 0x00, // and  x8, x18, x14
        0x33, 0x15, 0x17, 0x01, // sll  x10, x14, x1   // shift left logical
        0x33, 0x55, 0x27, 0x01, // srl  x10, x14, x2   // shift right logical
        // Instructions 15-19
        0x33, 0xa5, 0x27, 0x40, // slt  x10, x14, x2   // set less than
        0x13, 0x0b, 0x30, 0x00, // addi x22, x0, 3
        0x13, 0x8b, 0x3b, 0x00, // addi x23, x23, 3
        0x13, 0x0c, 0x4c, 0x00, // addi x24, x24, 4
        0x33, 0x8c, 0xdc, 0x01, // sub  x25, x25, x13
        // Instructions 20-24
        0x33, 0x0d, 0xdb, 0x01, // add  x26, x23, x13
        0x33, 0x0d, 0xeb, 0x01, // add  x26, x23, x14
        0x33, 0x0d, 0x0c, 0x02, // add  x26, x24, x16
        0x33, 0x0d, 0x1c, 0x02, // add  x26, x24, x17
        0x33, 0x8d, 0xdb, 0x01, // sub  x27, x23, x13
        0x83, 0x2c, 0x00, 0x02, // lw   0x20 -> x25
    ];

fn main() {
    let code = [
        0x13, 0x05, 0x10, 0x00, // addi x10, x0, 1
        0x13, 0x06, 0x20, 0x00, // addi x12, x0, 2
        0x13, 0x07, 0x30, 0x00, // addi x14, x0, 3
        0x13, 0x08, 0x40, 0x00, // addi x16, x0, 4
        0x13, 0x09, 0x50, 0x00, // addi x18, x0, 5
        // Math operations start here (this is instruction 5)
        0x33, 0x0e, 0xc5, 0x00, // add  x28, x10, x12
        0x33, 0x0f, 0xe5, 0x00, // add  x30, x10, x14
        0x33, 0x00, 0xc7, 0x01, // add  x0, x14, x12   // x0 always remains 0
        0x33, 0x01, 0x06, 0x01, // add  x2, x12, x16
        0x33, 0x81, 0xc6, 0x01, // sub  x3, x13, x12
        // Instructions 10-14
        0x33, 0x42, 0xd7, 0x01, // xor  x4, x15, x13
        0x33, 0x63, 0xc8, 0x00, // or   x6, x16, x12
        0x33, 0x74, 0xe9, 0x00, // and  x8, x18, x14
        0x33, 0x15, 0x17, 0x01, // sll  x10, x14, x1   // shift left logical
        0x33, 0x55, 0x27, 0x01, // srl  x10, x14, x2   // shift right logical
        // Instructions 15-19
        0x33, 0xa5, 0x27, 0x40, // slt  x10, x14, x2   // set less than
        0x13, 0x0b, 0x30, 0x00, // addi x22, x0, 3
        0x13, 0x8b, 0x3b, 0x00, // addi x23, x23, 3
        0x13, 0x0c, 0x4c, 0x00, // addi x24, x24, 4
        0x33, 0x8c, 0xdc, 0x01, // sub  x25, x25, x13
        // Instructions 20-24
        0x33, 0x0d, 0xdb, 0x01, // add  x26, x23, x13
        0x33, 0x0d, 0xeb, 0x01, // add  x26, x23, x14
        0x33, 0x0d, 0x0c, 0x02, // add  x26, x24, x16
        0x33, 0x0d, 0x1c, 0x02, // add  x26, x24, x17
        0x33, 0x8d, 0xdb, 0x01, // sub  x27, x23, x13
        0x83, 0x2c, 0x00, 0x01, // lw   0x20 -> x25
    ];

    let mut cpu = Cpu::new(&code);
    let mut builder = JITBuilder::new(cranelift_module::default_libcall_names()).expect("failed to create JITBuilder");

    // регистрируем helpers
    builder.symbol("mem_load32",  mem_load32 as *const u8);
    builder.symbol("mem_store32", mem_store32 as *const u8);

    let mut jit     = JITModule::new(builder);

    // for i in 0..32 {
    //     println!("x{} = {}", i, cpu.regs[i]);
    // }

    // исполняем, переводя TB максимум по 16 инструкций
    loop {
        let (fn_ptr, insns) = compile_tb(&mut jit, &cpu, 16);
        let executor: extern "C" fn(*mut Cpu) -> u32 =
            unsafe { std::mem::transmute(fn_ptr) };
        let next_pc = executor(&mut cpu);
        println!("insns {}", insns);
        if insns == 0 { break; }      // нет декодированных инструкций
        cpu.pc = next_pc;
        if next_pc == 0 { break; }    // наш demo JALR x0,0
    }

    for i in 0..32 {
        println!("x{} = {}", i, cpu.regs[i]);
    }
}