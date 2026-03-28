"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict, deque
from dataclasses import dataclass, field
import random
from typing import NamedTuple
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class VRegLane(NamedTuple):
    """A specific lane of a virtual register — used as dep-graph address labels."""
    vr_id: int
    lane: int


class VReg:
    """
    Virtual register: symbolic scratch region with `size` lanes.
    Physical scratch addresses are assigned by assign_vregs() after scheduling,
    so the code can use descriptive names without manually tracking aliasing.
    """
    _next_id = 0

    def __init__(self, size: int = VLEN):
        self.id = VReg._next_id
        VReg._next_id += 1
        self.size = size

    def __repr__(self):
        return f"VR{self.id}[{self.size}]"


def _lane(r, j: int):
    """Lane j address label for r (VReg → VRegLane, int → r+j)."""
    return VRegLane(r.id, j) if isinstance(r, VReg) else r + j


def _vaddr(r, n: int) -> list:
    """Address labels for n lanes of r (VReg or int)."""
    if isinstance(r, VReg):
        return [VRegLane(r.id, j) for j in range(n)]
    return list(range(r, r + n))


@dataclass
class InstrNode:
    id: int
    instr: dict
    reads: frozenset   # scratch addresses read
    writes: frozenset  # scratch addresses written
    deps: set = field(default_factory=set)  # IDs this instruction depends on


def _slot_rw(engine: str, slot: tuple) -> tuple[set, set]:
    """Return (reads, writes) address label sets for one slot. Supports VReg and int."""
    reads, writes = set(), set()
    op = slot[0]
    if engine == "alu":
        # Slots are per-lane: (op, dest_lane, src1_lane, src2_lane)
        # dest/src are int or VRegLane (from _lane() in vexec_alu/vexec_valu)
        _, dest, src1, src2 = slot
        writes.add(dest)
        reads.update({src1, src2})
    elif engine == "valu":
        if op == "vbroadcast":
            _, dest, src = slot
            writes.update(_vaddr(dest, VLEN))
            reads.update(_vaddr(src, 1))   # reads only element 0
        elif op == "multiply_add":
            _, dest, a, b, c = slot
            writes.update(_vaddr(dest, VLEN))
            reads.update(_vaddr(a, VLEN))
            reads.update(_vaddr(b, VLEN))
            reads.update(_vaddr(c, VLEN))
        else:
            _, dest, src1, src2 = slot
            writes.update(_vaddr(dest, VLEN))
            reads.update(_vaddr(src1, VLEN))
            reads.update(_vaddr(src2, VLEN))
    elif engine == "load":
        if op == "const":
            _, dest, _val = slot   # _val is a literal, not a scratch addr
            writes.add(dest)
        elif op == "load":
            _, dest, addr = slot
            writes.add(dest)
            reads.add(addr)
        elif op == "vload":
            _, dest, addr = slot
            writes.update(_vaddr(dest, VLEN))
            reads.add(addr)
    elif engine == "store":
        if op == "store":
            _, addr, src = slot
            reads.update({addr, src})
        elif op == "vstore":
            _, addr, src = slot
            reads.add(addr)
            reads.update(_vaddr(src, VLEN))
    elif engine == "flow":
        if op == "select":
            _, dest, cond, a, b = slot
            writes.add(dest)
            reads.update({cond, a, b})
        elif op == "vselect":
            _, dest, cond, a, b = slot
            writes.update(_vaddr(dest, VLEN))
            reads.update(_vaddr(cond, VLEN))
            reads.update(_vaddr(a, VLEN))
            reads.update(_vaddr(b, VLEN))
        # pause / jump: no scratch deps
    elif engine == "debug":
        if op == "vcompare":
            _, addr, _ref = slot
            reads.update(_vaddr(addr, VLEN))
        elif op == "compare":
            _, addr, _ref = slot
            reads.add(addr)
        # comment: no deps
    return reads, writes


def get_rw(instr: dict) -> tuple[frozenset, frozenset]:
    """Return (reads, writes) scratch address sets for an instruction dict."""
    reads, writes = set(), set()
    for engine, slots in instr.items():
        for slot in slots:
            r, w = _slot_rw(engine, slot)
            reads.update(r)
            writes.update(w)
    return frozenset(reads), frozenset(writes)


def build_dep_graph(instrs: list[dict]) -> list[InstrNode]:
    """
    Assign sequential IDs and build a dependency graph covering RAW, WAW, and WAR hazards.
    Returns InstrNodes in original program order; node.deps holds the IDs of direct predecessors.
    """
    nodes: list[InstrNode] = []
    last_writer: dict[int, int] = {}          # addr -> node id of last writer
    last_readers: dict[int, list[int]] = defaultdict(list)  # addr -> node ids of last readers

    for i, instr in enumerate(instrs):
        reads, writes = get_rw(instr)
        node = InstrNode(id=i, instr=instr, reads=reads, writes=writes)

        # RAW: we read an address someone else wrote
        for addr in reads:
            if addr in last_writer:
                node.deps.add(last_writer[addr])

        # WAW + WAR: we write an address others have read or written
        for addr in writes:
            if addr in last_writer:
                node.deps.add(last_writer[addr])       # WAW
            for rid in last_readers[addr]:
                node.deps.add(rid)                     # WAR
            last_readers[addr] = []
            last_writer[addr] = i

        # Record ourselves as a reader for future WAR tracking
        for addr in reads:
            last_readers[addr].append(i)

        nodes.append(node)

    return nodes


def schedule(nodes: list[InstrNode]) -> list[dict]:
    """
    Greedy FIFO topological scheduler: pack as many ready instructions as
    possible into each VLIW cycle, processing the ready queue in FIFO order
    so pipeline-slot chains naturally interleave.
    """
    if not nodes:
        return []

    n = len(nodes)
    successors: list[list[int]] = [[] for _ in range(n)]
    in_degree = [len(node.deps) for node in nodes]
    for node in nodes:
        for dep_id in node.deps:
            successors[dep_id].append(node.id)

    ready: deque[int] = deque(i for i in range(n) if in_degree[i] == 0)

    result: list[dict] = []
    remaining = n

    while remaining > 0:
        cycle: dict = {}
        slot_counts: dict[str, int] = defaultdict(int)
        scheduled_now: list[int] = []
        deferred: list[int] = []

        while ready:
            nid = ready.popleft()
            node = nodes[nid]
            fits = all(
                slot_counts[eng] + len(slots) <= SLOT_LIMITS[eng]
                for eng, slots in node.instr.items()
                if eng != "debug"
            )
            if fits:
                for eng, slots in node.instr.items():
                    cycle.setdefault(eng, []).extend(slots)
                    if eng != "debug":
                        slot_counts[eng] += len(slots)
                scheduled_now.append(nid)
            else:
                deferred.append(nid)

        if not scheduled_now:
            raise RuntimeError(f"Scheduler deadlock at cycle {len(result)}")

        ready.extend(deferred)
        for nid in scheduled_now:
            for succ_id in successors[nid]:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    ready.append(succ_id)

        remaining -= len(scheduled_now)
        result.append(cycle)

    return result


def assign_vregs(cycles: list[dict], scratch_start: int) -> tuple[list[dict], int]:
    """
    Linear-scan register allocator for VReg labels in a scheduled instruction list.

    VReg lanes (VRegLane) are symbolic addresses. This function computes their
    live ranges, assigns physical scratch slots via greedy linear scan (reusing
    slots whose live ranges don't overlap), then rewrites all VRegLane/VReg
    references to physical ints.

    Returns (rewritten_cycles, new_scratch_end).
    """
    # --- Pass 1: collect live ranges per vr_id ---
    # vreg_info[vr_id] = [first_write_cycle, last_read_cycle, size]
    vreg_info: dict[int, list] = {}

    for ci, cycle in enumerate(cycles):
        for engine, slots in cycle.items():
            for slot in slots:
                r, w = _slot_rw(engine, slot)
                for lbl in w:
                    if isinstance(lbl, VRegLane):
                        vid = lbl.vr_id
                        if vid not in vreg_info:
                            vreg_info[vid] = [ci, ci, lbl.lane + 1]
                        else:
                            vreg_info[vid][1] = ci
                            vreg_info[vid][2] = max(vreg_info[vid][2], lbl.lane + 1)
                for lbl in r:
                    if isinstance(lbl, VRegLane):
                        vid = lbl.vr_id
                        if vid in vreg_info:
                            vreg_info[vid][1] = max(vreg_info[vid][1], ci)

    if not vreg_info:
        return cycles, scratch_start

    # --- Pass 2: greedy linear scan ---
    alloc: dict[int, int] = {}              # vr_id -> physical base
    free: dict[int, list[int]] = defaultdict(list)   # size -> [freed bases]
    active: list[tuple[int, int, int]] = [] # (last_use, vr_id, phys_base)
    ptr = scratch_start

    for vr_id, (first, last, size) in sorted(vreg_info.items(), key=lambda x: x[1][0]):
        # Expire VRegs whose last use is strictly before this interval starts
        still_active, expired = [], []
        for entry in active:
            (expired if entry[0] < first else still_active).append(entry)
        for lu, vid, base in expired:
            free[vreg_info[vid][2]].append(base)
        active = still_active

        phys = free[size].pop() if free[size] else None
        if phys is None:
            phys = ptr
            ptr += size
            assert ptr <= SCRATCH_SIZE, f"VReg scratch overflow: {ptr} > {SCRATCH_SIZE}"

        alloc[vr_id] = phys
        active.append((last, vr_id, phys))

    # --- Pass 3: rewrite ---
    rewrite: dict[VRegLane, int] = {
        VRegLane(vid, lane): alloc[vid] + lane
        for vid, (_, _, size) in vreg_info.items()
        for lane in range(size)
    }

    def fix(x):
        if isinstance(x, VRegLane):
            return rewrite[x]
        if isinstance(x, VReg):
            return alloc[x.id]   # VALU base address
        return x

    def fix_slot(slot):
        return tuple(fix(x) if not isinstance(x, list) else x for x in slot)

    result = [
        {eng: [fix_slot(s) for s in slots] for eng, slots in cycle.items()}
        for cycle in cycles
    ]
    return result, ptr


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}
        self.vec_const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            if isinstance(slot, list):
                instrs.append({engine: slot})
            else:
                instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def schedule_instrs(self):
        """
        Schedule self.instrs in-place, packing instructions into VLIW cycles.
        Splits at pause boundaries so pauses stay in their original positions.
        """
        segments: list = []
        current_seg: list[dict] = []
        for instr in self.instrs:
            if "flow" in instr and any(s[0] == "pause" for s in instr["flow"]):
                segments.append(current_seg)
                segments.append(instr)   # pause marker (a plain dict, not a list)
                current_seg = []
            else:
                current_seg.append(instr)
        if current_seg:
            segments.append(current_seg)

        result: list[dict] = []
        for seg in segments:
            if isinstance(seg, dict):       # pause
                result.append(seg)
            else:
                result.extend(schedule(build_dep_graph(seg)))
        self.instrs = result

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, f"Out of scratch space {self.scratch_ptr}"
        return addr

    def scratch_const(self, val, val2=None, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            if val2 is not None:
                addr2 = self.alloc_scratch(name)
            # app = {"load": [("const", addr, val)] + ([] if val2 is None else ["const", addr2, val2])}
            # print(app)
            self.instrs.append(
                {"load": [("const", addr, val)] + ([] if val2 is None else [("const", addr2, val2)])}
            )
            self.const_map[val] = addr
            if val2 is not None:
                self.const_map[val2] = addr2
        if val2 is not None:
            return self.const_map[val], self.const_map[val2]
        return self.const_map[val]

    def vec_const(self, val, name=None, target=None):
        if val not in self.vec_const_map:
            addr = self.alloc_scratch(name, VLEN)
            t = target if target is not None else self.instrs
            if val in self.const_map:
                t.append({"valu": [("vbroadcast", addr, self.const_map[val])]})
            else:
                t.append({"load": [("const", addr, val)]})
                t.append({"valu": [("vbroadcast", addr, addr)]})
            self.vec_const_map[val] = addr
        return self.vec_const_map[val]

    def build_vhash_vreg(self, val_in, round, i):
        """Hash val_in through HASH_STAGES using VRegs (SSA style).
        Returns (instrs_list, val_out_vreg)."""
        instrs = []
        val_cur = val_in
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            val_next = VReg()
            if op1 == "+" and op2 == "+" and op3 == "<<":
                mul_const = self.vec_const(1 + (1 << val3))
                instrs.append({"valu": [("multiply_add", val_next, val_cur, mul_const, self.vec_const(val1))]})
            else:
                tmp_a = VReg()
                tmp_b = VReg()
                instrs.append({"valu": [(op1, tmp_a, val_cur, self.vec_const(val1))]})
                instrs.append({"valu": [(op3, tmp_b, val_cur, self.vec_const(val3))]})
                instrs.append({"valu": [(op2, val_next, tmp_a, tmp_b)]})
            instrs.append({"debug": [("vcompare", val_next, [(round, x, "hash_stage", hi) for x in range(i, i+VLEN)])]})
            val_cur = val_next
        return instrs, val_cur

    def build_vhash_slotted(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op1 == "+" and op2 == "+" and op3 == "<<":
                # a = (a + val1) + (a << val3) = a * (1 + 2**val3) + val1
                # Use fused multiply_add: dest = a * b + c
                mul_const = self.vec_const(1 + (1 << val3))
                slots.append({"valu": [("multiply_add", val_hash_addr, val_hash_addr, mul_const, self.vec_const(val1))]})
            else:
                # this is the case where we have this, a is the val_hash_addr
                # a = (a ^ const1) ^ (a >> const2)

                # a = (a ^ const1 ^ (a >> const2))
                slots.append({"valu": [(op1, tmp1, val_hash_addr, self.vec_const(val1))]})
                slots.append({"valu": [(op3, tmp2, val_hash_addr, self.vec_const(val3))]})
                slots.append({"valu": [(op2, val_hash_addr, tmp1, tmp2)]})
            slots.append({"debug": [("vcompare", val_hash_addr, [(round, x, "hash_stage", hi) for x in range(i, i+VLEN)])]})
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        PIPELINE = 16
        assert batch_size % (PIPELINE * VLEN) == 0

        # Shared broadcast vectors for round 0 and 1 tree node values.
        # All elements start at idx=0, so round 0 always loads forest_values[0].
        # After round 0, idx ∈ {1,2}; after round 1, idx ∈ {3,4,5,6}.
        nv0_vec  = self.alloc_scratch("nv0_vec",  VLEN)
        nv_addr1 = self.alloc_scratch("nv_addr1", 1)
        nv_addr2 = self.alloc_scratch("nv_addr2", 1)
        nv1_vec  = self.alloc_scratch("nv1_vec",  VLEN)
        nv2_vec  = self.alloc_scratch("nv2_vec",  VLEN)
        nv_addr3 = self.alloc_scratch("nv_addr3", 1)
        nv_addr4 = self.alloc_scratch("nv_addr4", 1)
        nv_addr5 = self.alloc_scratch("nv_addr5", 1)
        nv_addr6 = self.alloc_scratch("nv_addr6", 1)
        nv3_vec  = self.alloc_scratch("nv3_vec",  VLEN)
        nv4_vec  = self.alloc_scratch("nv4_vec",  VLEN)
        nv5_vec  = self.alloc_scratch("nv5_vec",  VLEN)
        nv6_vec  = self.alloc_scratch("nv6_vec",  VLEN)
        nv_diff_34_vec = self.alloc_scratch("nv_diff_34_vec", VLEN)
        nv_diff_56_vec = self.alloc_scratch("nv_diff_56_vec", VLEN)
        self.scratch_const(1, 2)
        self.scratch_const(3, 4)
        self.scratch_const(5, 6)

        # Shared init vars — broadcast scalars into VLEN vectors so valu can use them.
        # Only load what's actually used as a vector in instructions.
        # mem layout: 0=rounds,1=n_nodes,2=batch_size,3=forest_height,4=forest_values_p,5=inp_indices_p,6=inp_values_p
        init_vars = [
            ("n_nodes",         1),
            ("forest_values_p", 4),
        ]
        for name, _ in init_vars:
            self.alloc_scratch(name, VLEN)
        tmp_a = self.alloc_scratch("_init_tmp_a", 2)
        tmp_b = self.alloc_scratch("_init_tmp_b", 2)
        for i in range(0, len(init_vars), 2):
            paired = i + 1 < len(init_vars)
            self.instrs.append({"load": [("const", tmp_a + (i % 2), init_vars[i][1])] + ([("const", tmp_b+ (i % 2), init_vars[i+1][1])] if paired else [])})
            self.instrs.append({"load": [("load", tmp_a+ (i % 2), tmp_a+ (i % 2))] + ([("load", tmp_b+ (i % 2), tmp_b+ (i % 2))] if paired else [])})
            self.instrs.append({"valu": [("vbroadcast", self.scratch[init_vars[i][0]], tmp_a+ (i % 2))]
                                       + ([("vbroadcast", self.scratch[init_vars[i+1][0]], tmp_b+ (i % 2))] if paired else [])})

        one_vec  = self.vec_const(1)
        vecs = self.alloc_scratch(length=VLEN*3)
        zero_vec = vecs
        two_vec  = vecs + 8
        four_vec = vecs + 16
        self.instrs.append({"valu": [("-", zero_vec, one_vec, one_vec), ("+", two_vec, one_vec, one_vec)]})
        self.instrs.append({"valu": [("+", four_vec, two_vec, two_vec)]})

        def vexec_alu(op, dest, a1, a2, id_num):
            """Vector op preferring ALU (7/8 pipelines on ALU, 1/8 on VALU).
            Use when VALU is the throughput bottleneck and ALU lanes are idle.
            ALU case emits VLEN per-lane slots atomically — 1-cycle latency."""
            use_alu = id_num % 8 < 7
            if use_alu:
                return [{"alu": [(op, _lane(dest, j), _lane(a1, j), _lane(a2, j)) for j in range(VLEN)]}]
            return [{"valu": [(op, dest, a1, a2)]}]

        def vexec_valu(op, dest, a1, a2, id_num):
            """Vector op preferring VALU (7/8 pipelines on VALU, 1/8 on ALU).
            Use when both engines share load; spills 1/8 to ALU to reduce VALU contention."""
            use_alu = id_num % 8 == 0
            if use_alu:
                return [{"alu": [(op, _lane(dest, j), _lane(a1, j), _lane(a2, j)) for j in range(VLEN)]}]
            return [{"valu": [(op, dest, a1, a2)]}]

        # Scalar pointers for inp_indices_p (mem[5]) and inp_values_p (mem[6])
        inp_indices_p_s = self.alloc_scratch("inp_indices_p_s", 1)
        inp_values_p_s  = self.alloc_scratch("inp_values_p_s",  1)
        self.instrs.append({"load": [("const", inp_indices_p_s, 5), ("const", inp_values_p_s, 6)]})
        self.instrs.append({"load": [("load", inp_indices_p_s, inp_indices_p_s), ("load", inp_values_p_s, inp_values_p_s)]})

        # Stride tables: scratch[idx_ptr_base + k] = inp_indices_p_val + k*VLEN
        # so vload(dest, idx_ptr_base + k) loads the k-th VLEN chunk of the array.
        n_chunks = batch_size // VLEN
        idx_ptr_base = self.alloc_scratch("idx_ptrs", n_chunks)
        val_ptr_base = self.alloc_scratch("val_ptrs", n_chunks)
        for k in range(0, n_chunks, 2):
            off_c, off_c2 = self.scratch_const(k * VLEN, k * VLEN + VLEN)
            # off_c = self.scratch_const(k * VLEN)
            self.instrs.append({"alu": [
                ("+", idx_ptr_base + k, inp_indices_p_s, off_c),
                ("+", val_ptr_base + k, inp_values_p_s,  off_c),
                ("+", idx_ptr_base + k+1, inp_indices_p_s, off_c2),
                ("+", val_ptr_base + k+1, inp_values_p_s,  off_c2),
            ]})

        # Schedule the pre-pause segment so stride table ALU, init_vars, and const
        # loads are packed rather than emitted sequentially.
        self.instrs = schedule(build_dep_graph(self.instrs))

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow",  ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        instrs = []

        # Pre-emit all constants needed by the hash at the top of instrs so they
        # are visible to the dep-graph and can be scheduled alongside the nv preloads
        # (both are independent load+VALU work). Later vec_const() calls from
        # build_vhash_vreg will be cache hits and emit nothing.
        seen = set()
        for op1, val1, op2, op3, val3 in HASH_STAGES:
            for v in ([val1, val3, 1 + (1 << val3)] if op1 == "+" and op2 == "+" and op3 == "<<" else [val1, val3]):
                if v not in seen:
                    self.vec_const(v, target=instrs)
                    seen.add(v)

        # Pre-load tree node values for rounds 0 and 1 (once, shared across all i-iters).
        # Round 0: all idx=0 → broadcast forest_values[0]
        instrs.append({"load": [("load", nv0_vec, self.scratch["forest_values_p"])]})
        instrs.append({"valu": [("vbroadcast", nv0_vec, nv0_vec)]})
        # Round 1: idx ∈ {1,2} → load forest_values[1] and [2], broadcast each
        instrs.append({"alu": [
            ("+", nv_addr1, self.scratch["forest_values_p"], self.scratch_const(1)),
            ("+", nv_addr2, self.scratch["forest_values_p"], self.scratch_const(2)),
        ]})
        instrs.append({"load": [
            ("load", nv1_vec, nv_addr1),
            ("load", nv2_vec, nv_addr2),
        ]})
        instrs.append({"valu": [
            ("vbroadcast", nv1_vec, nv1_vec),
            ("vbroadcast", nv2_vec, nv2_vec),
        ]})
        # Round 2: idx ∈ {3,4,5,6} → preload forest_values[3..6] as broadcast vectors
        # and precompute nv_diff_34 = nv3-nv4, nv_diff_56 = nv5-nv6 for arithmetic mux
        instrs.append({"alu": [
            ("+", nv_addr3, self.scratch["forest_values_p"], self.scratch_const(3)),
            ("+", nv_addr4, self.scratch["forest_values_p"], self.scratch_const(4)),
        ]})
        instrs.append({"alu": [
            ("+", nv_addr5, self.scratch["forest_values_p"], self.scratch_const(5)),
            ("+", nv_addr6, self.scratch["forest_values_p"], self.scratch_const(6)),
        ]})
        instrs.append({"load": [
            ("load", nv3_vec, nv_addr3),
            ("load", nv4_vec, nv_addr4),
        ]})
        instrs.append({"load": [
            ("load", nv5_vec, nv_addr5),
            ("load", nv6_vec, nv_addr6),
        ]})
        instrs.append({"valu": [
            ("vbroadcast", nv3_vec, nv3_vec),
            ("vbroadcast", nv4_vec, nv4_vec),
        ]})
        instrs.append({"valu": [
            ("vbroadcast", nv5_vec, nv5_vec),
            ("vbroadcast", nv6_vec, nv6_vec),
        ]})
        instrs.append({"valu": [
            ("-", nv_diff_34_vec, nv3_vec, nv4_vec),
            ("-", nv_diff_56_vec, nv5_vec, nv6_vec),
        ]})

        for i in range(0, batch_size, PIPELINE * VLEN):
            i_chunk = i // VLEN
            # Per-pipeline VRegs for idx and val — fresh each i-iteration.
            # SSA: each write gets its own VReg; assign_vregs allocates physical slots.
            vr_idx = []
            vr_val = []
            for p in range(PIPELINE):
                k      = i_chunk + p
                offset = i + p * VLEN
                vi, vv = VReg(), VReg()
                vr_idx.append(vi)
                vr_val.append(vv)
                # instrs.append({"load": [
                #     ("vload", vi, idx_ptr_base + k),
                #     ("vload", vv, val_ptr_base + k),
                # ]})
                instrs.append({"load": [
                    ("vload", vi, idx_ptr_base + k),
                    ("vload", vv, val_ptr_base + k),
                ]})
                instrs.append({"debug": [("vcompare", vi, [(0, x, "idx") for x in range(offset, offset+VLEN)])]})
                instrs.append({"debug": [("vcompare", vv, [(0, x, "val") for x in range(offset, offset+VLEN)])]})

            cycle_len = forest_height + 1
            for round in range(0, rounds):
                effective = round % cycle_len
                is_last = (round == rounds - 1)

                if effective == 0:
                    for p in range(PIPELINE):
                        k      = i_chunk + p
                        offset = i + p * VLEN
                        instrs.append({"debug": [("vcompare", nv0_vec, [(round, x, "node_val") for x in range(offset, offset+VLEN)])]})
                        vr_xor = VReg()
                        instrs.extend(vexec_alu("^", vr_xor, vr_val[p], nv0_vec, p))
                        h_instrs, vr_hashed = self.build_vhash_vreg(vr_xor, round, offset)
                        instrs.extend(h_instrs)
                        instrs.append({"debug": [("vcompare", vr_hashed, [(round, x, "hashed_val") for x in range(offset, offset+VLEN)])]})
                        vr_bit, vr_idx_new = VReg(), VReg()
                        instrs.extend(vexec_alu("&", vr_bit,     vr_hashed,  one_vec, p))
                        instrs.extend(vexec_alu("+", vr_idx_new, vr_bit,     one_vec, p))
                        instrs.append({"debug": [("vcompare", vr_idx_new, [(round, x, "next_idx")    for x in range(offset, offset+VLEN)])]})
                        instrs.append({"debug": [("vcompare", vr_idx_new, [(round, x, "wrapped_idx") for x in range(offset, offset+VLEN)])]})
                        vr_val[p], vr_idx[p] = vr_hashed, vr_idx_new
                        store_slots = [] if is_last else [("vstore", idx_ptr_base + k, vr_idx_new)]
                        store_slots += [("vstore", val_ptr_base + k, vr_hashed)]
                        instrs.append({"store": store_slots})

                elif effective == 1:
                    for p in range(PIPELINE):
                        k      = i_chunk + p
                        offset = i + p * VLEN
                        vr_mask, vr_nv = VReg(), VReg()
                        instrs.extend(vexec_alu("==", vr_mask, vr_idx[p], one_vec, p))
                        instrs.append({"flow": [("vselect", vr_nv, vr_mask, nv1_vec, nv2_vec)]})
                        instrs.append({"debug": [("vcompare", vr_nv, [(round, x, "node_val") for x in range(offset, offset+VLEN)])]})
                        vr_xor = VReg()
                        instrs.extend(vexec_alu("^", vr_xor, vr_val[p], vr_nv, p))
                        h_instrs, vr_hashed = self.build_vhash_vreg(vr_xor, round, offset)
                        instrs.extend(h_instrs)
                        instrs.append({"debug": [("vcompare", vr_hashed, [(round, x, "hashed_val") for x in range(offset, offset+VLEN)])]})
                        vr_bit, vr_bit1, vr_idx_new = VReg(), VReg(), VReg()
                        instrs.extend(vexec_alu("&", vr_bit,  vr_hashed, one_vec, p))
                        instrs.extend(vexec_alu("+", vr_bit1, vr_bit,    one_vec, p))
                        instrs.append({"valu": [("multiply_add", vr_idx_new, vr_idx[p], two_vec, vr_bit1)]})
                        instrs.append({"debug": [("vcompare", vr_idx_new, [(round, x, "next_idx")    for x in range(offset, offset+VLEN)])]})
                        instrs.append({"debug": [("vcompare", vr_idx_new, [(round, x, "wrapped_idx") for x in range(offset, offset+VLEN)])]})
                        vr_val[p], vr_idx[p] = vr_hashed, vr_idx_new
                        store_slots = [] if is_last else [("vstore", idx_ptr_base + k, vr_idx_new)]
                        store_slots += [("vstore", val_ptr_base + k, vr_hashed)]
                        instrs.append({"store": store_slots})

                elif effective == 2:
                    # idx ∈ {3,4,5,6} — arithmetic 4-way mux, no scatter loads.
                    # ml = idx & 1, mh = 4 < idx
                    # nl = ml*(nv3−nv4)+nv4, nh = ml*(nv5−nv6)+nv6
                    # diff = nh − nl, node_val = mh*diff + nl
                    for p in range(PIPELINE):
                        k      = i_chunk + p
                        offset = i + p * VLEN
                        ml, mh, nl, nh = VReg(), VReg(), VReg(), VReg()
                        instrs.append({"valu": [("&", ml, vr_idx[p], one_vec)]})
                        instrs.append({"valu": [("<", mh, four_vec, vr_idx[p])]})
                        instrs.append({"valu": [
                            ("multiply_add", nl, ml, nv_diff_34_vec, nv4_vec),
                            ("multiply_add", nh, ml, nv_diff_56_vec, nv6_vec),
                        ]})
                        vr_diff, vr_nv = VReg(), VReg()
                        instrs.append({"valu": [("-", vr_diff, nh, nl)]})
                        instrs.append({"valu": [("multiply_add", vr_nv, mh, vr_diff, nl)]})
                        instrs.append({"debug": [("vcompare", vr_nv, [(round, x, "node_val") for x in range(offset, offset+VLEN)])]})
                        vr_xor = VReg()
                        instrs.append({"valu": [("^", vr_xor, vr_val[p], vr_nv)]})
                        h_instrs, vr_hashed = self.build_vhash_vreg(vr_xor, round, offset)
                        instrs.extend(h_instrs)
                        instrs.append({"debug": [("vcompare", vr_hashed, [(round, x, "hashed_val") for x in range(offset, offset+VLEN)])]})
                        vr_bit, vr_bit1, vr_idx_new = VReg(), VReg(), VReg()
                        instrs.extend(vexec_alu("&", vr_bit,  vr_hashed, one_vec, p))
                        instrs.extend(vexec_alu("+", vr_bit1, vr_bit,    one_vec, p))
                        instrs.append({"valu": [("multiply_add", vr_idx_new, vr_idx[p], two_vec, vr_bit1)]})
                        instrs.append({"debug": [("vcompare", vr_idx_new, [(round, x, "next_idx")    for x in range(offset, offset+VLEN)])]})
                        instrs.append({"debug": [("vcompare", vr_idx_new, [(round, x, "wrapped_idx") for x in range(offset, offset+VLEN)])]})
                        vr_val[p], vr_idx[p] = vr_hashed, vr_idx_new
                        store_slots = [] if is_last else [("vstore", idx_ptr_base + k, vr_idx_new)]
                        store_slots += [("vstore", val_ptr_base + k, vr_hashed)]
                        instrs.append({"store": store_slots})

                else:
                    # Full scatter load.
                    for p in range(PIPELINE):
                        k      = i_chunk + p
                        offset = i + p * VLEN
                        vr_addr, vr_nv = VReg(), VReg()
                        instrs.extend(vexec_valu("+", vr_addr, self.scratch["forest_values_p"], vr_idx[p], p))
                        for j in range(0, VLEN, 2):
                            instrs.append({"load": [
                                ("load", VRegLane(vr_nv.id, j),   VRegLane(vr_addr.id, j)),
                                ("load", VRegLane(vr_nv.id, j+1), VRegLane(vr_addr.id, j+1)),
                            ]})
                        instrs.append({"debug": [("vcompare", vr_nv, [(round, x, "node_val") for x in range(offset, offset+VLEN)])]})
                        vr_xor = VReg()
                        instrs.extend(vexec_valu("^", vr_xor, vr_val[p], vr_nv, p))
                        h_instrs, vr_hashed = self.build_vhash_vreg(vr_xor, round, offset)
                        instrs.extend(h_instrs)
                        instrs.append({"debug": [("vcompare", vr_hashed, [(round, x, "hashed_val") for x in range(offset, offset+VLEN)])]})
                        vr_bit, vr_bit1, vr_idx_pre = VReg(), VReg(), VReg()
                        instrs.extend(vexec_valu("&", vr_bit, vr_hashed, one_vec, p))
                        instrs.extend(vexec_valu("+", vr_bit1, vr_bit, one_vec, p))
                        instrs.append({"valu": [("multiply_add", vr_idx_pre, vr_idx[p], two_vec, vr_bit1)]})
                        instrs.append({"debug": [("vcompare", vr_idx_pre, [(round, x, "next_idx") for x in range(offset, offset+VLEN)])]})
                        vr_cmp, vr_idx_new = VReg(), VReg()
                        instrs.extend(vexec_valu("<", vr_cmp, vr_idx_pre, self.scratch["n_nodes"], p))
                        instrs.append({"flow": [("vselect", vr_idx_new, vr_cmp, vr_idx_pre, zero_vec)]})
                        instrs.append({"debug": [("vcompare", vr_idx_new, [(round, x, "wrapped_idx") for x in range(offset, offset+VLEN)])]})
                        vr_val[p], vr_idx[p] = vr_hashed, vr_idx_new
                        store_slots = [] if is_last else [("vstore", idx_ptr_base + k, vr_idx_new)]
                        store_slots += [("vstore", val_ptr_base + k, vr_hashed)]
                        instrs.append({"store": store_slots})

        nodes = build_dep_graph(instrs)
        scheduled = schedule(nodes)
        scheduled, _ = assign_vregs(scheduled, self.scratch_ptr)
        self.instrs.extend(scheduled)
        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])

        # if i == 0:
        # print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
        # print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])

        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(1234)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
