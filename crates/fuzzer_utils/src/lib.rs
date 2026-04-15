use std::sync::Mutex;
use lazy_static::lazy_static;
use openvm_stark_backend::p3_field::{Field, PrimeField32};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;
use rand::seq::IndexedRandom;

use openvm_rv32im_transpiler::{
    BaseAluOpcode,
    ShiftOpcode,
    LessThanOpcode,
    Rv32LoadStoreOpcode,
    BranchEqualOpcode,
    BranchLessThanOpcode,
    Rv32JalLuiOpcode,
    Rv32JalrOpcode,
    Rv32AuipcOpcode,
    MulOpcode,
    MulHOpcode,
    DivRemOpcode,
    Rv32HintStoreOpcode,
};

use openvm_instructions::{
    VmOpcode,
    LocalOpcode,
    SystemOpcode,
    instruction::Instruction,
};


////////////////
// GLOBAL STATE
/////////

/// Global state machine that controls fault injection and trace logging.
///
/// A single `lazy_static` instance is shared across all threads via `Mutex`.
/// The Python fuzzer configures this via the host binary's CLI flags
/// (`--inject`, `--seed`, `--trace`, etc.) and then the patched OpenVM code
/// reads it from anywhere in the executor without plumbing changes.
#[derive(Debug, Clone)]
pub struct GlobalState {
    /// When `true`, `print_trace_info()` and `print_injection_info()` emit
    /// XML-tagged JSON lines to stdout (`<trace>...</trace>`, `<fault>...</fault>`)
    /// so the Python fuzzer can parse the instruction trace.
    /// Set via the host binary's `--trace` flag.
    pub trace_logging: bool,

    /// Master on/off switch for fault injection. When `false`, all injection
    /// hooks stay dormant regardless of `injection_step`/`injection_kind`.
    /// Set via the host binary's `--inject` flag.
    pub injection: bool,

    /// Controls whether `fuzzer_assert!` / `fuzzer_assert_eq!` / `fuzzer_assert_ne!`
    /// panic on failure (like standard `assert!`) or just print a warning and
    /// continue. Disabled during injection so that a corrupted execution can
    /// reach the prover instead of crashing on internal invariant checks.
    pub assertions: bool,

    /// RNG seed stored for logging/reproducibility. A finding can be replayed
    /// by re-running with the same `(seed, injection_step, injection_kind)`.
    pub seed: u64,

    /// Global counter of instructions executed so far. Incremented once per
    /// instruction by `inc_step()`. Used as the "clock" for injection timing
    /// and also as an endless-loop guard (panics at > 1,000,000 steps).
    pub step: u64,

    /// Target injection type (e.g. `"INSTR_WORD_MOD"`, `"BASE_ALU_RANDOM_OUTPUT"`).
    /// Each patched code site checks its own specific kind string, so only
    /// the matching injection fires. Selects *what* to mutate.
    pub injection_kind: String,

    /// Target step number at which the injection should fire. Compared against
    /// `step` to decide "is this the moment to inject?". Selects *when* to
    /// mutate (the Nth dynamic instruction, not a static PC).
    pub injection_step: u64,

    /// Seeded RNG that drives all mutation randomness
    /// (`random_mutate_instruction`, `random_mod_of_u32`, etc.). Seeded from
    /// `seed` so mutations are deterministic and findings are reproducible.
    pub rng: StdRng,

    /// Human-readable opcode name of the current instruction (e.g. `"ADD"`,
    /// `"LOADW"`). Attached to every `<trace>` / `<fault>` log so Python can
    /// reconstruct what was executing. Refreshed each cycle by `update_hints()`.
    pub hint_instruction: String,

    /// Debug-formatted full instruction (e.g.
    /// `"Instruction { opcode: ADD_RV32, a: 4, b: 8, c: 12, ... }"`). Included
    /// in trace/fault logs for richer reproduction context.
    pub hint_assembly: String,

    /// Program counter of the current instruction, captured before any
    /// mutation. Tags every trace/fault log line so the fuzzer can correlate
    /// steps with static code locations.
    pub hint_pc: u32,

    /// Counts how many fault injections actually fired during the current run.
    /// Incremented by `print_injection_info()` (called once per fired hook).
    ///
    /// The orchestrator resets this to 0 before each run via
    /// `reset_fault_count()`, and reads it after via `get_fault_count()`.
    ///
    /// Expected values after a single injection run:
    ///   - `0`: target step never reached (program terminated early) or the
    ///     requested `injection_kind` didn't match any hook at that step
    ///   - `1`: injection fired as intended (the normal case)
    ///   - `>1`: impossible by design — `step` is strictly monotonic so
    ///     `is_injection_at_step` can match at most once per run. If this is
    ///     ever observed, it indicates a bug (e.g., two hook sites for the
    ///     same kind in the same cycle).
    pub fault_count: u64,
}

impl GlobalState {
    fn new() -> Self {
        Self {
            trace_logging: false,
            injection: false,
            assertions: true,
            seed: 0,
            injection_kind: String::new(),
            step: 0,
            injection_step: 0,
            rng: StdRng::seed_from_u64(0),
            hint_instruction: String::new(),
            hint_assembly: String::new(),
            hint_pc: 0,
            fault_count: 0,
        }
    }
}

lazy_static! {
    static ref GLOBAL_STATE: Mutex<GlobalState> = Mutex::new(GlobalState::new());
}

pub fn is_trace_logging() -> bool {
    GLOBAL_STATE.lock().unwrap().trace_logging
}

pub fn set_trace_logging(value: bool) {
    let mut state = GLOBAL_STATE.lock().unwrap();
    state.trace_logging = value;
}

pub fn enable_trace_logging() {
    set_trace_logging(true);
}

pub fn disable_trace_logging() {
    set_trace_logging(false);
}

pub fn is_injection() -> bool {
    GLOBAL_STATE.lock().unwrap().injection
}

pub fn set_injection(value: bool) {
    let mut state = GLOBAL_STATE.lock().unwrap();
    state.injection = value;
}

pub fn enable_injection() {
    set_injection(true);
}

pub fn disable_injection() {
    set_injection(false);
}

pub fn is_assertions() -> bool {
    GLOBAL_STATE.lock().unwrap().assertions
}

pub fn set_assertions(value: bool) {
    let mut state = GLOBAL_STATE.lock().unwrap();
    state.assertions = value;
}

pub fn enable_assertions() {
    set_assertions(true);
}

pub fn disable_assertions() {
    set_assertions(false);
}

pub fn set_seed(value: u64) {
    let mut state = GLOBAL_STATE.lock().unwrap();
    state.rng = StdRng::seed_from_u64(value);
    state.seed = value;
}

pub fn get_seed() -> u64 {
    GLOBAL_STATE.lock().unwrap().seed
}

pub fn is_injection_kind(value: &str) -> bool {
    GLOBAL_STATE.lock().unwrap().injection_kind == value
}

pub fn set_injection_kind(value: String) {
    let mut state = GLOBAL_STATE.lock().unwrap();
    state.injection_kind = value.clone();
}

pub fn get_injection_kind() -> String {
    GLOBAL_STATE.lock().unwrap().injection_kind.clone()
}

pub fn inc_step() {
    let mut state = GLOBAL_STATE.lock().unwrap();
    state.step += 1;

    if state.step > 1000000 {
        panic!("Endless loop detection step bound triggered! Bound: 1000000 steps");
    }
}

pub fn get_step() -> u64 {
    GLOBAL_STATE.lock().unwrap().step
}

/// Reset the fault counter to 0. Call this before each run so the next
/// `get_fault_count()` reports only faults from the upcoming execution.
///
/// The counter itself is incremented inside `print_injection_info` (not
/// via a separate function) to avoid double-locking the global mutex.
pub fn reset_fault_count() {
    let mut state = GLOBAL_STATE.lock().unwrap();
    state.fault_count = 0;
}

/// Read the number of fault injections that fired during the current run.
/// The orchestrator (e.g. arguzz-plus) uses this to verify that injection
/// actually happened (expected value: 1 for a successful injection run,
/// 0 if the target step was never reached).
pub fn get_fault_count() -> u64 {
    GLOBAL_STATE.lock().unwrap().fault_count
}

pub fn get_injection_step() -> u64 {
    GLOBAL_STATE.lock().unwrap().injection_step
}

pub fn set_injection_step(value: u64) {
    let mut state = GLOBAL_STATE.lock().unwrap();
    state.injection_step = value;
}

pub fn is_injection_at_step(kind: &str) -> bool {
    let state = GLOBAL_STATE.lock().unwrap();
    state.injection &&
        state.step == state.injection_step &&
        state.injection_kind == kind
}

pub fn set_hint_instruction(value: &String) {
    let mut state = GLOBAL_STATE.lock().unwrap();
    state.hint_instruction = value.clone();
}

pub fn get_hint_instruction() -> String {
    let state = GLOBAL_STATE.lock().unwrap();
    state.hint_instruction.clone()
}

pub fn set_hint_assembly(value: &String) {
    let mut state = GLOBAL_STATE.lock().unwrap();
    state.hint_assembly = value.clone();
}

pub fn get_hint_assembly() -> String {
    let state = GLOBAL_STATE.lock().unwrap();
    state.hint_assembly.clone()
}

pub fn set_hint_pc(value: u32) {
    let mut state = GLOBAL_STATE.lock().unwrap();
    state.hint_pc = value;
}

pub fn get_hint_pc() -> u32 {
    let state = GLOBAL_STATE.lock().unwrap();
    state.hint_pc
}

pub fn update_hints(pc: u32, instruction: &String, assembly: &String) {
    let mut state = GLOBAL_STATE.lock().unwrap();
    state.hint_pc = pc;
    state.hint_instruction = instruction.clone();
    state.hint_assembly = assembly.clone();
}

////////////////
// CUSTOM ASSERTION MACROS
/////////

/// Custom assert! macro — binds condition to local to avoid double-evaluation.
#[macro_export]
macro_rules! fuzzer_assert {
    ($cond:expr $(,)?) => {{
        let cond_val = $cond;
        if $crate::is_assertions() {
            assert!(cond_val);
        } else if !cond_val {
            println!("Warning: fuzzer_assert! failed: {}", stringify!($cond));
        }
    }};
    ($cond:expr, $($arg:tt)+) => {{
        let cond_val = $cond;
        if $crate::is_assertions() {
            assert!(cond_val, $($arg)+);
        } else if !cond_val {
            println!("Warning: fuzzer_assert! failed: {}", format_args!($($arg)+));
        }
    }};
}

/// Custom assert_eq! macro — binds to locals to avoid double-move issues.
#[macro_export]
macro_rules! fuzzer_assert_eq {
    ($left:expr, $right:expr $(,)?) => {{
        let left_val = $left;
        let right_val = $right;
        if $crate::is_assertions() {
            assert_eq!(left_val, right_val);
        } else if left_val != right_val {
            println!(
                "Warning: fuzzer_assert_eq! failed: `{} != {}` (left: `{:?}`, right: `{:?}`)",
                stringify!($left),
                stringify!($right),
                &left_val,
                &right_val,
            );
        }
    }};
    ($left:expr, $right:expr, $($arg:tt)+) => {{
        let left_val = $left;
        let right_val = $right;
        if $crate::is_assertions() {
            assert_eq!(left_val, right_val, $($arg)+);
        } else if left_val != right_val {
            println!(
                "Warning: fuzzer_assert_eq! failed: `{} != {}` (left: `{:?}`, right: `{:?}`): {}",
                stringify!($left),
                stringify!($right),
                &left_val,
                &right_val,
                format_args!($($arg)+),
            );
        }
    }};
}

/// Custom assert_ne! macro — binds to locals to avoid double-move issues.
#[macro_export]
macro_rules! fuzzer_assert_ne {
    ($left:expr, $right:expr $(,)?) => {{
        let left_val = $left;
        let right_val = $right;
        if $crate::is_assertions() {
            assert_ne!(left_val, right_val);
        } else if left_val == right_val {
            println!(
                "Warning: fuzzer_assert_ne! failed: `{} == {}` (left: `{:?}`, right: `{:?}`)",
                stringify!($left),
                stringify!($right),
                &left_val,
                &right_val,
            );
        }
    }};
    ($left:expr, $right:expr, $($arg:tt)+) => {{
        let left_val = $left;
        let right_val = $right;
        if $crate::is_assertions() {
            assert_ne!(left_val, right_val, $($arg)+);
        } else if left_val == right_val {
            println!(
                "Warning: fuzzer_assert_ne! failed: `{} == {}` (left: `{:?}`, right: `{:?}`): {}",
                stringify!($left),
                stringify!($right),
                &left_val,
                &right_val,
                format_args!($($arg)+),
            );
        }
    }};
}


////////////////
// LOGGING
/////////

/// Report that a fault injection has fired.
///
/// Always increments `GlobalState::fault_count` so orchestrators can detect
/// whether injection actually occurred. Additionally emits a `<fault>...</fault>`
/// JSON line to stdout if `trace_logging` is enabled.
///
/// Called once per successful injection hook (after the mutation is applied).
/// Because `is_injection_at_step` can only match once per run (by design —
/// `step` is strictly monotonic), this function is called at most once per
/// run under normal operation.
pub fn print_injection_info(
    inject_kind: &str,
    info: &String,
) {
    let mut state = GLOBAL_STATE.lock().unwrap();

    // Always count the fault — the orchestrator uses this to verify
    // injection actually happened (expected: exactly 1 per run).
    state.fault_count += 1;

    // Optionally emit a <fault>...</fault> tag for stdout-based parsing
    // (used by the original ARGUZZ Python fuzzer; arguzz-plus uses
    // `get_fault_count()` instead).
    if state.trace_logging {
        println!(
            "<fault>{{\
                \"step\":{}, \
                \"pc\":{}, \
                \"instruction\":\"{}\", \
                \"assembly\":\"{}\", \
                \"kind\":\"{}\", \
                \"info\":\"{}\"\
            }}</fault>",
            state.step,
            state.hint_pc,
            state.hint_instruction,
            state.hint_assembly,
            inject_kind,
            info,
        );
    }
}

pub fn print_trace_info() {
    let state = GLOBAL_STATE.lock().unwrap();
    if state.trace_logging {
        println!(
            "<trace>{{\
                \"step\":{}, \
                \"pc\":{}, \
                \"instruction\":\"{}\", \
                \"assembly\":\"{}\"\
            }}</trace>",
            state.step,
            state.hint_pc,
            state.hint_instruction,
            state.hint_assembly,
        );
    }
}


////////////////
// RANDOMNESS
/////////

pub fn random_bool() -> bool {
    let mut state = GLOBAL_STATE.lock().unwrap();
    state.rng.random::<bool>()
}

pub fn random_from_choices<T>(choices: Vec<T>) -> T
    where T : Clone
{
    let mut state = GLOBAL_STATE.lock().unwrap();
    choices.choose(&mut state.rng).unwrap().clone()
}

pub fn random_opcode(rng: &mut StdRng) -> VmOpcode {
    // 39 valid RV32IM opcodes in latest OpenVM.
    //
    // Breakdown: BaseAlu(5) + Shift(3) + LessThan(2) + LoadStore(8)
    //          + BranchEqual(2) + BranchLessThan(4) + JalLui(2) + Jalr(1)
    //          + Auipc(1) + Mul(1) + MulH(3) + DivRem(4) + HintStore(2)
    //          + System::TERMINATE(1) = 39
    match rng.random_range(0..=38) {
         0 => BaseAluOpcode::ADD.global_opcode(),
         1 => BaseAluOpcode::SUB.global_opcode(),
         2 => BaseAluOpcode::XOR.global_opcode(),
         3 => BaseAluOpcode::OR.global_opcode(),
         4 => BaseAluOpcode::AND.global_opcode(),
         5 => ShiftOpcode::SLL.global_opcode(),
         6 => ShiftOpcode::SRL.global_opcode(),
         7 => ShiftOpcode::SRA.global_opcode(),
         8 => LessThanOpcode::SLT.global_opcode(),
         9 => LessThanOpcode::SLTU.global_opcode(),
        10 => Rv32LoadStoreOpcode::LOADW.global_opcode(),
        11 => Rv32LoadStoreOpcode::LOADBU.global_opcode(),
        12 => Rv32LoadStoreOpcode::LOADHU.global_opcode(),
        13 => Rv32LoadStoreOpcode::STOREW.global_opcode(),
        14 => Rv32LoadStoreOpcode::STOREH.global_opcode(),
        15 => Rv32LoadStoreOpcode::STOREB.global_opcode(),
        16 => Rv32LoadStoreOpcode::LOADB.global_opcode(),
        17 => Rv32LoadStoreOpcode::LOADH.global_opcode(),
        18 => BranchEqualOpcode::BEQ.global_opcode(),
        19 => BranchEqualOpcode::BNE.global_opcode(),
        20 => BranchLessThanOpcode::BLT.global_opcode(),
        21 => BranchLessThanOpcode::BLTU.global_opcode(),
        22 => BranchLessThanOpcode::BGE.global_opcode(),
        23 => BranchLessThanOpcode::BGEU.global_opcode(),
        24 => Rv32JalLuiOpcode::JAL.global_opcode(),
        25 => Rv32JalLuiOpcode::LUI.global_opcode(),
        26 => Rv32JalrOpcode::JALR.global_opcode(),
        27 => Rv32AuipcOpcode::AUIPC.global_opcode(),
        28 => MulOpcode::MUL.global_opcode(),
        29 => MulHOpcode::MULH.global_opcode(),
        30 => MulHOpcode::MULHSU.global_opcode(),
        31 => MulHOpcode::MULHU.global_opcode(),
        32 => DivRemOpcode::DIV.global_opcode(),
        33 => DivRemOpcode::DIVU.global_opcode(),
        34 => DivRemOpcode::REM.global_opcode(),
        35 => DivRemOpcode::REMU.global_opcode(),
        36 => Rv32HintStoreOpcode::HINT_STOREW.global_opcode(),
        37 => Rv32HintStoreOpcode::HINT_BUFFER.global_opcode(),
        38 => SystemOpcode::TERMINATE.global_opcode(),
        // Note: PHANTOM (SystemOpcode::PHANTOM) is intentionally excluded
        // as injecting it can cause host-side panics.
        _  => panic!("selector value was out of bounds!"),
    }
}

pub fn random_new_opcode(opcode: VmOpcode, rng: &mut StdRng) -> VmOpcode {
    loop {
        let new_opcode = random_opcode(rng);
        if new_opcode != opcode {
            return new_opcode;
        }
    }
}

/// Mutate a single `u32` into a different `u32` using one of 8 strategies.
///
/// Strategies are biased toward **boundary values** that commonly expose
/// constraint-system bugs rather than purely uniform random values:
///
/// | Selector | Strategy                     | Tests                          |
/// |----------|------------------------------|--------------------------------|
/// | 0        | Set to `0`                   | Zero-handling paths            |
/// | 1        | Set to `1`                   | Minimal non-zero paths         |
/// | 2        | Set to `0xffffffff` (MAX)    | Saturation / overflow          |
/// | 3        | Set to `0xfffffffe` (MAX-1)  | Off-by-one near max            |
/// | 4        | Flip 1-31 random bits        | Partial corruption             |
/// | 5        | `saturating_add(1)`          | Just-slightly-more edge        |
/// | 6        | `saturating_sub(1)`          | Just-slightly-less edge        |
/// | 7        | Fully random `u32`           | Unbiased chaos                 |
///
/// The `while new_element == element` loop guarantees the output always
/// differs from the input (e.g., `saturating_add(1)` on `u32::MAX` retries).
///
/// Matches ARGUZZ's `internal_random_mod_of_u32` for reproducibility.
fn internal_random_mod_of_u32(element: u32, rng: &mut StdRng) -> u32 {
    let mut new_element = element;
    while new_element == element {
        let selector: u32 = rng.random_range(0..=7);
        new_element = match selector {
            0 => { 0 },
            1 => { 1 },
            2 => { 0xffffffff },
            3 => { 0xfffffffe },
            4 => {
                let n = rng.random_range(1..=31);
                let bits_to_flip = rand::seq::index::sample(rng, 31, n).into_vec();
                let mut flipped_element = element;
                for bit_to_flip in bits_to_flip {
                    flipped_element ^= 1 << bit_to_flip;
                }
                flipped_element
            },
            5 => { element.saturating_add(1) },
            6 => { element.saturating_sub(1) },
            7 => { rng.random::<u32>() },
            _ => unreachable!(),
        };
    }
    new_element
}

/// Mutate a random subset of elements in a fixed-size `u32` array.
///
/// Selects `k` random positions (where `1 <= k <= LEN`, uniform over that range)
/// and applies [`internal_random_mod_of_u32`] to each selected element. The
/// remaining elements are left unchanged.
///
/// Steps:
///   1. Generate all indices `[0, 1, ..., LEN-1]`
///   2. Fisher-Yates shuffle to produce a random permutation
///   3. Pick `k` (number of positions to mutate) uniformly in `1..=LEN`
///   4. Mutate the first `k` indices of the shuffled permutation
///
/// Uses the shared global RNG (not a caller-supplied one), so the choice
/// of both `k` and the positions is deterministic given the global seed.
///
/// Used by chip-level injection hooks (e.g. `BASE_ALU_RANDOM_OUTPUT`,
/// `AUIPC_PC_LIMBS_MODIFICATION`) to corrupt limb arrays like `[u32; 4]`
/// (4-limb register representation) where corrupting a subset of limbs
/// tests how many limbs the constraint system correctly binds.
pub fn random_mod_of_u32_array<const LEN: usize>(elements: &[u32; LEN]) -> [u32; LEN] {
    let mut state = GLOBAL_STATE.lock().unwrap();

    let mut new_elements = *elements;
    let mut indices: Vec<usize> = (0..LEN).collect();
    indices.shuffle(&mut state.rng);
    let num_to_modify = state.rng.random_range(1..=LEN);

    for &i in indices.iter().take(num_to_modify) {
        new_elements[i] = internal_random_mod_of_u32(elements[i], &mut state.rng);
    }

    new_elements
}

/// Mutate a prime-field element (e.g. BabyBear) using the u32 mutation table.
///
/// Converts the element to its canonical `u32` representation via
/// [`PrimeField32::as_canonical_u32`], applies [`internal_random_mod_of_u32`]
/// (which produces one of 8 boundary/chaos values), then converts back to `F`
/// via [`PrimeCharacteristicRing::from_u32`].
///
/// # Note on u32 vs field range
///
/// BabyBear's modulus is `15 * 2^27 + 1 ≈ 0x78000001` (a 31-bit prime), so
/// canonical field elements fit in `[0, 2^31)`. However,
/// `internal_random_mod_of_u32` can produce values up to `0xffffffff` that
/// exceed the modulus. These are **not out of bounds** — `F::from_u32` reduces
/// them modulo the field prime, so e.g. `0xffffffff` becomes
/// `0xffffffff mod 0x78000001 = 0x0ffffffc` (still an "interesting" boundary
/// value near `2^28`, just not the raw u32 max).
///
/// The retry loop inside `internal_random_mod_of_u32` checks u32 equality,
/// not field equality. In rare cases two different pre-reduction u32 values
/// could reduce to the same field element, but the probability is negligible
/// and the next iteration of the fuzzer would produce a different mutation.
///
/// Used by [`random_mutate_instruction`] to corrupt individual operand fields
/// (a, b, c, d, e, f, g) of an `Instruction<F>`. The caller supplies the RNG
/// rather than using the global, because this function is invoked inside a
/// `GLOBAL_STATE.lock()` critical section and re-locking would deadlock.
pub fn random_mutate_field_element<F: Field + PrimeField32>(element: F, rng: &mut StdRng) -> F {
    F::from_u32(internal_random_mod_of_u32(element.as_canonical_u32(), rng))
}

/// Mutate an OpenVM instruction by randomly corrupting 1-8 of its fields.
///
/// An OpenVM `Instruction<F>` has 8 fields total:
///   `opcode, a, b, c, d, e, f, g`
///
/// Selects a random subset of size `k` (uniform over `1..=8`) and mutates
/// each selected field:
///   - Field 0 (`opcode`): reset the entire instruction to default values,
///     then pick a new random opcode via [`random_new_opcode`]. This is a
///     "full reset" because an old opcode with mismatched operand types is
///     usually invalid, so we start from a clean slate.
///   - Fields 1-7 (`a..g`): mutate via [`random_mutate_field_element`],
///     which maps each operand through the boundary-value u32 mutation table.
///
/// # Order matters: opcode first
///
/// The selected indices are sorted after shuffling so that if index 0 (opcode)
/// is selected, it is applied *before* any operand mutations. Otherwise the
/// opcode reset would overwrite any earlier operand mutations.
///
/// # How the 1-8 selection works
///
///   1. Shuffle `[0, 1, 2, 3, 4, 5, 6, 7]`
///   2. Truncate to `k` (random in `1..=8`)
///   3. Sort the surviving indices (opcode-first guarantee)
///   4. Apply each mutation in order
///
/// This is the core mutation used by the `INSTR_WORD_MOD` injection hook
/// in the patched preflight interpreter: at the target step, the fetched
/// instruction is replaced with the output of this function before being
/// dispatched to the chip executor.
pub fn random_mutate_instruction<F: Field + PrimeField32>(instruction: &Instruction<F>) -> Instruction<F> {
    let mut state = GLOBAL_STATE.lock().unwrap();

    // Start from a copy of the original instruction; mutations are applied in place.
    let mut new_instruction = instruction.clone();

    // Pick how many fields to mutate (1-8, uniform).
    let update_fields = state.rng.random_range(1..=8);

    // All field indices: 0 = opcode, 1..=7 = operands a..g
    let mut update_options: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];

    // Random subset of size `update_fields`: shuffle + truncate
    update_options.shuffle(&mut state.rng);
    update_options.truncate(update_fields);

    // Sort so opcode (0) fires first — its "full reset" would otherwise
    // overwrite earlier operand mutations.
    update_options.sort();

    // Apply each selected mutation
    for option in update_options {
        match option {
            0 => {
                // Opcode mutation: reset to default and pick a new opcode.
                // The reset clears operand fields that may be invalid for
                // the new opcode (e.g. immediate vs register address space).
                new_instruction = Instruction::default();
                new_instruction.opcode = random_new_opcode(instruction.opcode, &mut state.rng);
            },
            1 => { new_instruction.a = random_mutate_field_element(new_instruction.a, &mut state.rng); },
            2 => { new_instruction.b = random_mutate_field_element(new_instruction.b, &mut state.rng); },
            3 => { new_instruction.c = random_mutate_field_element(new_instruction.c, &mut state.rng); },
            4 => { new_instruction.d = random_mutate_field_element(new_instruction.d, &mut state.rng); },
            5 => { new_instruction.e = random_mutate_field_element(new_instruction.e, &mut state.rng); },
            6 => { new_instruction.f = random_mutate_field_element(new_instruction.f, &mut state.rng); },
            7 => { new_instruction.g = random_mutate_field_element(new_instruction.g, &mut state.rng); },
            _ => unreachable!(),
        };
    }

    new_instruction
}
