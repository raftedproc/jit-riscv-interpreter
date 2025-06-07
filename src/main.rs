mod cpu;
mod reg;
mod translator;

use cranelift_jit::{JITBuilder, JITModule};
use translator::{mem_load32, mem_store32};

use crate::cpu::Cpu;
use crate::translator::compile_tb;

use log::{debug, info, trace, warn, error, log_enabled};
use env_logger::Env;

/// A set of random instructions for testing.
const RANDOM_MATH: [u8; 104] = [
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

    const SIMPLE_ADDITION: [u8; 16] = [
        0x13, 0x05, 0x10, 0x00,     // addi x10, x0, 1
        0x13, 0x0a, 0x20, 0x00,     // addi x20, x0, 2
        0x33, 0x0e, 0xaa, 0x00,     // add  x28, x20, x10Add commentMore actions
        0x67, 0x80, 0x00, 0x00,     // jalr x0, 0(x0)   -> stop
    ];

fn main() {
    // Initialize logger from environment variables
    // Use RUST_LOG=debug,memap=trace to set debug level for all crates and trace level for this crate
    // Example: RUST_LOG=debug ./memap
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    
    info!("Starting RISC-V interpreter");
    debug!("Initializing with test code");
    let code = RANDOM_MATH;

    trace!("Creating CPU instance");
    let mut cpu = Cpu::new(&code);
    trace!("Creating JIT builder");
    let mut builder = JITBuilder::new(cranelift_module::default_libcall_names()).expect("failed to create JITBuilder");

    // регистрируем helpers
    debug!("Registering helper functions");
    builder.symbol("mem_load32",  mem_load32 as *const u8);
    builder.symbol("mem_store32", mem_store32 as *const u8);

    trace!("Creating JIT module");
    let mut jit = JITModule::new(builder);

    info!("Starting execution loop");
    loop {
        trace!("Compiling translation block at PC: {:#x}", cpu.pc);
        let (fn_ptr, insns) = compile_tb(&mut jit, &cpu, 16);
        
        debug!("Compiled {} instructions", insns);
        if insns == 0 { 
            info!("No more instructions to decode, exiting");
            break; 
        }
        
        trace!("Executing compiled code at address: {:p}", fn_ptr);
        let executor: extern "C" fn(*mut Cpu) -> u32 =
            unsafe { std::mem::transmute(fn_ptr) };
        let next_pc = executor(&mut cpu);
        
        debug!("Execution completed, next PC: {:#x}, instructions executed: {}", next_pc, insns);
        cpu.pc = next_pc;
        if next_pc == 0 { 
            info!("Hit JALR x0,0 instruction, stopping execution");
            break; 
        }
    }

    // Dump register state at trace level
    if log_enabled!(log::Level::Trace) {
        trace!("Final register state:");
        for i in 0..32 {
            trace!("Register x{} = {}", i, cpu.regs[i]);
        }
    }
    
    info!("Execution completed successfully");
}