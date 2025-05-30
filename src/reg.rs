use cranelift_codegen::ir::{ExtFuncData, ExternalName, InstBuilder, MemFlags, Signature};
use cranelift_codegen::ir::{types, AbiParam, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_frontend::Variable;

pub fn load_two_regs(
    b: &mut FunctionBuilder<'_>,
    cpu_ptr: Value,
    regs: &[Variable; 32],
    regs_read_so_far: &mut [bool; 32],
    rs1: Option<usize>,
    rs2: Option<usize>,
) -> (usize, usize) {
    let rs1 = load_reg_if_needed(b, cpu_ptr, rs1.unwrap(), regs_read_so_far, regs);
    let rs2 = load_reg_if_needed(b, cpu_ptr, rs2.unwrap(), regs_read_so_far, regs);
    (rs1, rs2)
}

pub fn define_rd_and_mark_dirty(
    b: &mut FunctionBuilder<'_>,
    regs: &[Variable; 32],
    dirty_regs: &mut [bool; 32],
    rd: Option<usize>,
    r: Value,
) {
    let rd_idx = rd.unwrap();
    b.def_var(regs[rd_idx], r);
    println!("define_rd_and_mark_dirty def rd {}", regs[rd_idx]);
    dirty_regs[rd_idx] = true;
}

pub fn load_reg_if_needed(
    b: &mut FunctionBuilder<'_>,
    cpu_ptr: Value,
    reg: usize,
    regs_read_so_far: &mut [bool; 32],
    regs: &[Variable],
) -> usize {
    println!("load_reg_if_needed try to load reg {}", reg);
    if !regs_read_so_far[reg] {
      println!("load_reg_if_needed loading reg {}", reg);
        load_register_from_cpu(b, cpu_ptr, reg, &regs[reg]);
        regs_read_so_far[reg] = true;
    }
    reg
}

pub fn load_register_from_cpu(
    b: &mut FunctionBuilder<'_>,
    cpu_ptr: Value,
    reg: usize,
    reg_var: &Variable,
) {
    // TODO convention first comes regs array.
    println!("loading reg {} to {}", reg, reg_var);
    let off = (reg * 4) as i64;
    let addr = b.ins().iadd_imm(cpu_ptr, off);
    let val = b.ins().load(types::I32, MemFlags::new(), addr, 0);
    b.def_var(*reg_var, val);
}

pub fn store_registers_to_cpu(
    b: &mut FunctionBuilder<'_>,
    cpu_ptr: Value,
    regs: &[Variable],
    dirty_regs: &[bool],
) {
    // Only store registers that have been modified
    for i in 0..32 {
        // Skip registers that haven't been modified
        if !dirty_regs[i] {
            continue;
        }
        let reg_val = b.use_var(regs[i]);
        println!("Register {}: retrieved value = {:?}", i, reg_val);
        let off = (i * 4) as i64;

        // Calculate pointer to CPU's regs[i]
        let addr = b.ins().iadd_imm(cpu_ptr, off);

        // Store the register value back to CPU memory
        b.ins().store(MemFlags::new(), reg_val, addr, 0);
        println!("Stored reg {} value back to CPU at offset {}", i, off);
    }
}

/// helper-ы для доступа к памяти: вызываем обычные Rust-функции
pub fn call_mem_load(b: &mut FunctionBuilder, cpu_ptr: Value, addr: Value) -> Value {
    let call_conv = b.func.signature.call_conv;
    let sig = {
        let sig = b.func.import_signature(Signature {
            params: vec![AbiParam::new(types::I64), AbiParam::new(types::I32)],
            returns: vec![AbiParam::new(types::I32)],
            call_conv: call_conv,
        });
        b.func.import_function(ExtFuncData {
            name: ExternalName::testcase("mem_load32"),
            signature: sig,
            colocated: false,
        })
    };
    let call = b.ins().call(sig, &[cpu_ptr, addr]);
    b.inst_results(call)[0]
}

pub fn call_mem_store(b: &mut FunctionBuilder, cpu_ptr: Value, addr: Value, val: Value) {
    let call_conv = b.func.signature.call_conv;
    let sig = {
        let sig = b.func.import_signature(Signature {
            params: vec![
                AbiParam::new(types::I64),
                AbiParam::new(types::I32),
                AbiParam::new(types::I32),
            ],
            returns: vec![],
            call_conv: call_conv,
        });
        b.func.import_function(ExtFuncData {
            name: ExternalName::testcase("mem_store32"),
            signature: sig,
            colocated: false,
        })
    };
    b.ins().call(sig, &[cpu_ptr, addr, val]);
}
