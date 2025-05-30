use cranelift_codegen::ir::{condcodes::IntCC, *};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::JITModule;
use cranelift_module::Module;
use raki::{BaseIOpcode, Decode, Instruction, Isa, OpcodeKind};

use crate::cpu::Cpu;
use crate::reg::*;

pub extern "C" fn mem_load32(cpu: &mut Cpu, addr: u32) -> u32 {
    cpu.load32(addr)
}
pub extern "C" fn mem_store32(cpu: &mut Cpu, addr: u32, val: u32) {
    cpu.store32(addr, val)
}

/// Компилирует до `max_insns` или до первой JALR (эмуляция границы TB)
pub fn compile_tb(jit: &mut JITModule, cpu: &Cpu, max_insns: usize) -> (*const u8, usize) {
    let mut ctx = jit.make_context();
    ctx.func.signature.params.push(AbiParam::new(types::I64)); // *mut Cpu
    ctx.func.signature.returns.push(AbiParam::new(types::I32)); // next PC

    let mut fctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);

    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);

    let cpu_ptr = b.block_params(entry)[0]; // *mut Cpu as i64
    let regs: [Variable; 32] = core::array::from_fn(|i| Variable::from_u32(i as u32));
    // Track which registers have been modified during this TB
    let mut regs_read_or_changed_so_far = [false; 32];
    let mut dirty_regs = [false; 32];

    // объявим x0..x31 как переменные
    // use liveness analysis
    for i in 0..32 {
        b.declare_var(regs[i], types::I32);
    }

    let mut pc = cpu.pc;

    let mut cnt = 0;
    let mut term_was_added = false;
    while cnt < max_insns {
        let raw = cpu.load32(pc);
        println!("raw {:x}", raw);

        // Handle the case where we might be reading beyond the program
        // (memory might be zeroed or have invalid instruction patterns)
        let inst = match raw.decode(Isa::Rv32) {
            Ok(inst) => {
                println!("inst {:?}", inst);
                inst
            }
            Err(e) => {
                println!("Failed to decode instruction at pc={}: {:?}", pc, e);
                // We've reached the end of the program, so break the loop
                break;
            }
        };
        match inst.opc {
            OpcodeKind::BaseI(BaseIOpcode::ADDI) => {
                let Instruction { rd, rs1, imm, .. } = inst;
                // todo not needed to actually load rs1 if it is x0
                let rs1 = load_reg_if_needed(&mut b, cpu_ptr, rs1.unwrap(), &mut regs_read_or_changed_so_far, &regs);

                let v1 = b.use_var(regs[rs1]);
                let r = b.ins().iadd_imm(v1, imm.unwrap() as i64);
                println!("ADDI {} r {}", v1, r);
                
                regs_read_or_changed_so_far[rd.unwrap()] = true;
                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, r);
            }
            OpcodeKind::BaseI(BaseIOpcode::ADD) => {
                let Instruction { rd, rs1, rs2, .. } = inst;
                let (rs1, rs2) = load_two_regs(&mut b, cpu_ptr, &regs, &mut regs_read_or_changed_so_far, rs1, rs2);

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);
                let v = b.ins().iadd(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            OpcodeKind::BaseI(BaseIOpcode::SUB) => {
                let Instruction { rd, rs1, rs2, .. } = inst;
                let (rs1, rs2) = load_two_regs(&mut b, cpu_ptr, &regs, &mut regs_read_or_changed_so_far, rs1, rs2);

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);
                let v = b.ins().isub(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            OpcodeKind::BaseI(BaseIOpcode::XOR) => {
                let Instruction { rd, rs1, rs2, .. } = inst;
                let (rs1, rs2) = load_two_regs(&mut b, cpu_ptr, &regs, &mut regs_read_or_changed_so_far, rs1, rs2);

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);
                let v = b.ins().bxor(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            OpcodeKind::BaseI(BaseIOpcode::OR) => {
                let Instruction { rd, rs1, rs2, .. } = inst;
                let (rs1, rs2) = load_two_regs(&mut b, cpu_ptr, &regs, &mut regs_read_or_changed_so_far, rs1, rs2);

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);

                let v = b.ins().bor(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            OpcodeKind::BaseI(BaseIOpcode::AND) => {
                let Instruction { rd, rs1, rs2, .. } = inst;
                let (rs1, rs2) = load_two_regs(&mut b, cpu_ptr, &regs, &mut regs_read_or_changed_so_far, rs1, rs2);

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);

                let v = b.ins().band(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            OpcodeKind::BaseI(BaseIOpcode::SLL) => {
                let Instruction { rd, rs1, rs2, .. } = inst;
                let (rs1, rs2) = load_two_regs(&mut b, cpu_ptr, &regs, &mut regs_read_or_changed_so_far, rs1, rs2);

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);

                let v = b.ins().ishl(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            OpcodeKind::BaseI(BaseIOpcode::SRL) => {
                let Instruction { rd, rs1, rs2, .. } = inst;
                let (rs1, rs2) = load_two_regs(&mut b, cpu_ptr, &regs, &mut regs_read_or_changed_so_far, rs1, rs2);

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);

                let v = b.ins().ushr(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            OpcodeKind::BaseI(BaseIOpcode::SLT) => {
                let Instruction { rd, rs1, rs2, .. } = inst;
                let (rs1, rs2) = load_two_regs(&mut b, cpu_ptr, &regs, &mut regs_read_or_changed_so_far, rs1, rs2);

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);

                // Use icmp_slt to compare if v1 < v2 (signed comparison)
                let cond = b.ins().icmp(IntCC::SignedLessThan, v1, v2);
                let zero = b.ins().iconst(types::I32, 0);
                let one = b.ins().iconst(types::I32, 1);
                let v = b.ins().select(cond, one, zero);
                b.def_var(regs[rd.unwrap()], v);
            }
            OpcodeKind::BaseI(BaseIOpcode::LW) => {
                let Instruction { rd, rs1, imm, .. } = inst;
                let rs1 = load_reg_if_needed(&mut b, cpu_ptr, rs1.unwrap(), &mut regs_read_or_changed_so_far, &regs);

                let base = b.use_var(regs[rs1]);
                let addr = b.ins().iadd_imm(base, imm.unwrap() as i64);
                let val = call_mem_load(&mut b, cpu_ptr, addr);
                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, val);
            }
            OpcodeKind::BaseI(BaseIOpcode::SW) => {
                let Instruction { rs2, rs1, imm, .. } = inst;
                let (rs1, rs2) = load_two_regs(&mut b, cpu_ptr, &regs, &mut regs_read_or_changed_so_far, rs1, rs2);

                let base = b.use_var(regs[rs1]);
                let addr = b.ins().iadd_imm(base, imm.unwrap() as i64);
                let val = b.use_var(regs[rs2]);

                call_mem_store(&mut b, cpu_ptr, addr, val);
            }
            OpcodeKind::BaseI(BaseIOpcode::JALR) => {
                let Instruction { rd, rs1, imm, .. } = inst;
                let target = b.use_var(regs[rs1.unwrap()]);
                let next = b.ins().iadd_imm(target, imm.unwrap() as i64);
                let const_pc = b.ins().iconst(types::I32, (pc + 4) as i64);
                b.def_var(regs[rd.unwrap()], const_pc);

                store_registers_to_cpu(&mut b, cpu_ptr, &regs, &dirty_regs);

                b.ins().return_(&[next]);
                term_was_added = true;
                break;
            }
            OpcodeKind::M(raki::MOpcode::MUL) => {
                let Instruction { rd, rs1, rs2, .. } = inst;
                let (rs1, rs2) = load_two_regs(&mut b, cpu_ptr, &regs, &mut regs_read_or_changed_so_far, rs1, rs2);

                let v1 = b.use_var(regs[rs1]);
                let v2 = b.use_var(regs[rs2]);
                let v = b.ins().imul(v1, v2);

                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, v);
            }
            _ => unimplemented!("demo supports few instrs"),
        }
        pc += 4;
        cnt += 1;
    }

    // если дошли до лимита, вернуть следующий PC
    if cnt == max_insns || !term_was_added {
        store_registers_to_cpu(&mut b, cpu_ptr, &regs, &dirty_regs);
        let rvals = &[b.ins().iconst(types::I32, pc as i64)];
        b.ins().return_(rvals);
    }

    b.seal_all_blocks();
    // replace with ctx.func.signature ?
    let sign = b.func.signature.clone();
    b.finalize();

    println!("----- Cranelift IR for translation block -----");
    println!("{}", ctx.func.display());
    println!("----- End of Cranelift IR -----");

    let id = jit.declare_anonymous_function(&sign).unwrap();
    jit.define_function(id, &mut ctx).unwrap();
    jit.clear_context(&mut ctx);
    jit.finalize_definitions().expect("must be ok");
    (jit.get_finalized_function(id), cnt)
}
