use cranelift_codegen::ir::{condcodes::IntCC, *};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::JITModule;
use cranelift_module::Module;
use log::{error, trace};
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
        // Handle the case where we might be reading beyond the program
        // (memory might be zeroed or have invalid instruction patterns)
        let inst = match raw.decode(Isa::Rv32) {
            Ok(inst) => {
                trace!("inst {:?}", inst);
                inst
            }
            Err(e) => {
                error!("Failed to decode instruction at pc={}: {:?}", pc, e);
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
                let rd = rd.unwrap();
                b.def_var(regs[rd], v);
                dirty_regs[rd] = true;
            }
            OpcodeKind::BaseI(BaseIOpcode::LW) => {

                let Instruction { rd, rs1, imm, .. } = inst;

                let rs1 = load_reg_if_needed(&mut b, cpu_ptr, rs1.unwrap(), &mut regs_read_or_changed_so_far, &regs);

                let base = b.use_var(regs[rs1]);
                let addr = b.ins().iadd_imm(base, imm.unwrap() as i64);
                let val = call_mem_load(jit, &mut b, cpu_ptr, addr);
                define_rd_and_mark_dirty(&mut b, &regs, &mut dirty_regs, rd, val);
            }
            OpcodeKind::BaseI(BaseIOpcode::SW) => {
                let Instruction { rs2, rs1, imm, .. } = inst;
                let (rs1, rs2) = load_two_regs(&mut b, cpu_ptr, &regs, &mut regs_read_or_changed_so_far, rs1, rs2);

                let base = b.use_var(regs[rs1]);
                let addr = b.ins().iadd_imm(base, imm.unwrap() as i64);
                let val = b.use_var(regs[rs2]);
 
                call_mem_store( jit, &mut b, cpu_ptr, addr, val);
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

    let id = jit.declare_anonymous_function(&sign).unwrap();
    jit.define_function(id, &mut ctx).unwrap();
    jit.clear_context(&mut ctx);
    jit.finalize_definitions().expect("must be ok");
    (jit.get_finalized_function(id), cnt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::default_libcall_names;
    use crate::cpu::Cpu;

    // Helper function to setup test environment
    fn setup_test_env(program: &[u8]) -> (Cpu, JITModule, usize, u32) {
        // Initialize CPU with the test program
        let cpu = Cpu::new(program);
        
        setup_test_env_with_cpu(program, cpu)
    }

    fn setup_test_env_with_cpu(program: &[u8], mut cpu: Cpu) -> (Cpu, JITModule, usize, u32) {
        println!("setup_test_env_with_cpu {} ", program.len());

        // Setup JIT module
        let mut builder = JITBuilder::new(default_libcall_names())
            .expect("failed to create JITBuilder");
        
        // Register helper functions
        builder.symbol("mem_load32", mem_load32 as *const u8);
        builder.symbol("mem_store32", mem_store32 as *const u8);
        
        let mut jit = JITModule::new(builder);
        
        // Compile the translation block
        let (fn_ptr, insns) = compile_tb(&mut jit, &cpu, program.len() / 4); // Max instructions based on program size
        
        // Execute the compiled code
        let executor: extern "C" fn(*mut Cpu) -> u32 = unsafe { std::mem::transmute(fn_ptr) };
        let next_pc = executor(&mut cpu);
        
        (cpu, jit, insns, next_pc)
    }

    #[test]
    fn test_add_instruction() {
        // Define a simple program with ADD instruction
        let test_program = [
            0x13, 0x05, 0x10, 0x00,     // addi x10, x0, 1
            0x13, 0x0a, 0x20, 0x00,     // addi x20, x0, 2
            0x33, 0x0e, 0xaa, 0x00,     // add  x28, x20, x10
        ];

        let (cpu, _, insns, next_pc) = setup_test_env(&test_program);
        
        // Verify the results
        assert_eq!(insns, 3, "Should have translated all 3 instructions");
        assert_eq!(next_pc, 12, "PC should be 12 after execution");
        assert_eq!(cpu.regs[10], 1, "Register x10 should be 1");
        assert_eq!(cpu.regs[20], 2, "Register x20 should be 2");
        assert_eq!(cpu.regs[28], 3, "Register x28 should be 3 (result of add)");
    }

    #[test]
    fn test_sub_instruction() {
        // Define a simple program with SUB instruction
        let test_program = [
            0x13, 0x05, 0x30, 0x00,     // addi x10, x0, 7
            0x13, 0x0a, 0x70, 0x00,     // addi x20, x0, 3
            0x33, 0x0e, 0x45, 0x41,     // sub  x28, x20, x10
        ];

        let (cpu, _, insns, next_pc) = setup_test_env(&test_program);
        
        // Verify the results
        assert_eq!(insns, 3, "Should have translated all 3 instructions");
        assert_eq!(next_pc, 12, "PC should be 12 after execution");
        assert_eq!(cpu.regs[10], 3, "Register x10 should be 3");
        assert_eq!(cpu.regs[20], 7, "Register x20 should be 7");
        assert_eq!(cpu.regs[28], -4i32 as u32, "Register x28 should be -4 (result of 3-7)");
    }

    #[test]
    fn test_xor_instruction() {
        // Define a simple program with XOR instruction
        let test_program = [
            0x13, 0x05, 0x30, 0x00,     // addi x10, x0, 3
            0x13, 0x0a, 0x50, 0x00,     // addi x20, x0, 5
            0x33, 0x4e, 0xaa, 0x00,     // xor  x28, x20, x10
        ];

        let (cpu, _, insns, next_pc) = setup_test_env(&test_program);
        
        // Verify the results
        assert_eq!(insns, 3, "Should have translated all 3 instructions");
        assert_eq!(next_pc, 12, "PC should be 12 after execution");
        assert_eq!(cpu.regs[10], 3, "Register x10 should be 3");
        assert_eq!(cpu.regs[20], 5, "Register x20 should be 5");
        assert_eq!(cpu.regs[28], 6, "Register x28 should be 6 (result of 5 XOR 3)");
    }

    #[test]
    fn test_or_instruction() {
        // Define a simple program with OR instruction
        let test_program = [
            0x13, 0x05, 0x90, 0x00,     // addi x10, x0, 9
            0x13, 0x0a, 0x60, 0x00,     // addi x20, x0, 6
            0x33, 0x6e, 0xaa, 0x00,     // or   x28, x20, x10
        ];

        let (cpu, _, insns, next_pc) = setup_test_env(&test_program);
        
        // Verify the results
        assert_eq!(insns, 3, "Should have translated all 3 instructions");
        assert_eq!(next_pc, 12, "PC should be 12 after execution");
        assert_eq!(cpu.regs[10], 9, "Register x10 should be 9");
        assert_eq!(cpu.regs[20], 6, "Register x20 should be 6");
        assert_eq!(cpu.regs[28], 15, "Register x28 should be 15 (result of 6 OR 9)");
    }

    #[test]
    fn test_and_instruction() {
        // Define a simple program with AND instruction
        let test_program = [
            0x13, 0x05, 0xF0, 0x00,     // addi x10, x0, 15
            0x13, 0x0a, 0x60, 0x00,     // addi x20, x0, 6
            0x33, 0x7e, 0xaa, 0x00,     // and  x28, x20, x10
        ];

        let (cpu, _, insns, next_pc) = setup_test_env(&test_program);
        
        // Verify the results
        assert_eq!(insns, 3, "Should have translated all 3 instructions");
        assert_eq!(next_pc, 12, "PC should be 12 after execution");
        assert_eq!(cpu.regs[10], 15, "Register x10 should be 15");
        assert_eq!(cpu.regs[20], 6, "Register x20 should be 6");
        assert_eq!(cpu.regs[28], 6, "Register x28 should be 6 (result of 6 AND 15)");
    }

    #[test]
    fn test_sll_instruction() {
        // Define a simple program with SLL (Shift Left Logical) instruction
        let test_program = [
            0x13, 0x05, 0x10, 0x00,     // addi x10, x0, 1
            0x13, 0x0a, 0x20, 0x00,     // addi x20, x0, 2
            0x33, 0x1e, 0xaa, 0x00,     // sll  x28, x20, x10
        ];

        let (cpu, _, insns, next_pc) = setup_test_env(&test_program);
        
        // Verify the results
        assert_eq!(insns, 3, "Should have translated all 3 instructions");
        assert_eq!(next_pc, 12, "PC should be 12 after execution");
        assert_eq!(cpu.regs[10], 1, "Register x10 should be 1");
        assert_eq!(cpu.regs[20], 2, "Register x20 should be 2");
        assert_eq!(cpu.regs[28], 4, "Register x28 should be 4 (result of 2 << 1)");
    }

    #[test]
    fn test_srl_instruction() {
        // Define a simple program with SRL (Shift Right Logical) instruction
        let test_program = [
            0x13, 0x05, 0x10, 0x00,     // addi x10, x0, 1
            0x13, 0x0a, 0x80, 0x00,     // addi x20, x0, 8
            0x33, 0x5e, 0xaa, 0x00,     // srl  x28, x20, x10
        ];

        let (cpu, _, insns, next_pc) = setup_test_env(&test_program);
        
        // Verify the results
        assert_eq!(insns, 3, "Should have translated all 3 instructions");
        assert_eq!(next_pc, 12, "PC should be 12 after execution");
        assert_eq!(cpu.regs[10], 1, "Register x10 should be 1");
        assert_eq!(cpu.regs[20], 8, "Register x20 should be 8");
        assert_eq!(cpu.regs[28], 4, "Register x28 should be 4 (result of 8 >> 1)");
    }

    #[test]
    fn test_slt_instruction() {
        // Define a simple program with SLT (Set Less Than) instruction
        // RISC-V R-type instruction format for SLT: funct7(0000000) | rs2 | rs1 | funct3(010) | rd | opcode(0110011)
        let test_program = [
            0x13, 0x05, 0x50, 0x00,     // addi x10, x0, 5
            0x13, 0x0a, 0x30, 0x00,     // addi x20, x0, 3
            // For SLT, we need rs1=x20(3), rs2=x10(5), rd=x28
            // SLT performs rd = (rs1 < rs2) ? 1 : 0, and since 3 < 5, we expect 1
            0x33, 0x2e, 0xaa, 0x00,     // slt  x28, x20, x10
        ];

        let (cpu, _, insns, next_pc) = setup_test_env(&test_program);
        
        // Verify the results
        assert_eq!(insns, 3, "Should have translated all 3 instructions");
        assert_eq!(next_pc, 12, "PC should be 12 after execution");

        assert_eq!(cpu.regs[10], 5, "Register x10 should be 5");
        assert_eq!(cpu.regs[20], 3, "Register x20 should be 3");
        assert_eq!(cpu.regs[28], 1, "Register x28 should be 1 (result of 3 < 5)");
        
    }

    #[test]
    fn test_lw_instruction() {
        // Define a simple program with LW instruction
        // This program will:
        // 1. Set x10 to an address (0) 
        // 2. Store the value 42 at memory address (0)
        // 3. Load from that address into x28
        let test_program = [
            0x13, 0x05, 0x00, 0x04,     // addi x10, x0, 0      # set x10 to address 0
            0x93, 0x0F, 0xA0, 0x02,     // addi x31, x0, 42     # set x31 to value 42
            // 0x23, 0x20, 0xFF, 0x00,     // sw   x31, 0(x10)     # store 42 at address 0
            0x83, 0x2E, 0x05, 0x00,     // lw   x29, 0(x10)     # load from address 0 into x29
        ];

        let mut cpu = Cpu::new(&test_program);
        cpu.mem[64] = 42;
        let (cpu, _, insns, next_pc) = setup_test_env_with_cpu(&test_program, cpu);
        
        // Verify the results
        assert_eq!(insns, 3, "Should have translated all 4 instructions");
        assert_eq!(next_pc, 12, "PC should be 16 after execution");
        assert_eq!(cpu.regs[10], 64, "Register x10 should be 64");
        assert_eq!(cpu.regs[29], 42, "Register x29 should be 42 (loaded from memory)");
    }

    #[test]
    fn test_sw_instruction() {
        // Define a simple program with SW instruction
        // This program will:
        // 1. Set x10 to an address (4)
        // 2. Set x20 to a value (123)
        // 3. Store x20 to the address in x10 + 4
        let test_program = [
            0x13, 0x05, 0x40, 0x00,     // addi x10, x0, 4      # set x10 to address 4
            0x13, 0x0A, 0xB0, 0x07,     // addi x20, x0, 123    # set x20 to value 123
            0x23, 0x22, 0x45, 0x01,     // sw   x20, 4(x10)     # store 123 at address 8
        ];

        let (cpu, _, insns, next_pc) = setup_test_env(&test_program);
        
        // Verify the results
        assert_eq!(insns, 3, "Should have translated all 3 instructions");
        assert_eq!(next_pc, 12, "PC should be 12 after execution");
        assert_eq!(cpu.regs[10], 4, "Register x10 should be 4");
        assert_eq!(cpu.regs[20], 123, "Register x20 should be 123");
        assert_eq!(cpu.mem[8], 123, "Register x28 should be 123 (loaded from memory)");
    }

    #[test]
    fn test_jalr_instruction() {
        // Define a simple program with JALR instruction
        // This program will:
        // 1. Set x10 to an address (20)
        // 2. Jump to that address and link the return address to x1
        // We expect the TB to end at the JALR, so we'll check the next_pc value
        // Additionally, x1 should contain the return address (PC+4)
        let test_program = [
            0x13, 0x05, 0x40, 0x01,     // addi x10, x0, 20     # set x10 to address 20
            0x67, 0x80, 0x05, 0x00,     // jalr x1, 0(x10)      # jump to address in x10
            // The following should not be executed:
            0x13, 0x0F, 0x10, 0x00,     // addi x30, x0, 1      # set x30 to 1
        ];

        let (cpu, _, insns, next_pc) = setup_test_env(&test_program);
        
        // Verify the results
        assert_eq!(insns, 1, "Should have translated 2 instructions (until JALR)");
        assert_eq!(next_pc, 20, "PC should be 20 after execution (jumped to x10)");
        assert_eq!(cpu.regs[10], 20, "Register x10 should be 20");
        assert_eq!(cpu.regs[1], 8, "Register x1 should be 8 (return address: PC+4)");
        assert_eq!(cpu.regs[30], 0, "Register x30 should be 0 (instruction after JALR not executed)");
    }

    #[test]
    fn test_mul_instruction() {
        // Define a simple program with MUL instruction
        let test_program = [
            0x13, 0x05, 0x70, 0x00,     // addi x10, x0, 7      # set x10 to 7
            0x13, 0x0a, 0x40, 0x00,     // addi x20, x0, 4      # set x20 to 4
            0x33, 0x0e, 0x45, 0x03,     // mul  x28, x10, x20   # x28 = x10 * x20
        ];

        let (cpu, _, insns, next_pc) = setup_test_env(&test_program);
        
        // Verify the results
        assert_eq!(insns, 3, "Should have translated all 3 instructions");
        assert_eq!(next_pc, 12, "PC should be 12 after execution");
        assert_eq!(cpu.regs[10], 7, "Register x10 should be 7");
        assert_eq!(cpu.regs[20], 4, "Register x20 should be 4");
        assert_eq!(cpu.regs[28], 28, "Register x28 should be 28 (result of 7 * 4)");
    }
}
