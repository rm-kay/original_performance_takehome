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


@dataclass
class InstrNode:
    id: int
    instr: dict
    reads: frozenset   # scratch addresses read
    writes: frozenset  # scratch addresses written
    deps: set = field(default_factory=set)  # IDs this instruction depends on


def _slot_rw(engine: str, slot: tuple) -> tuple[set, set]:
    """Return (reads, writes) scratch address sets for one slot."""
    reads, writes = set(), set()
    op = slot[0]
    if engine == "alu":
        _, dest, src1, src2 = slot
        writes.add(dest)
        reads.update({src1, src2})
    elif engine == "valu":
        if op == "vbroadcast":
            _, dest, src = slot
            writes.update(range(dest, dest + VLEN))
            reads.add(src)
        else:
            _, dest, src1, src2 = slot
            writes.update(range(dest, dest + VLEN))
            reads.update(range(src1, src1 + VLEN))
            reads.update(range(src2, src2 + VLEN))
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
            writes.update(range(dest, dest + VLEN))
            reads.add(addr)
    elif engine == "store":
        if op == "store":
            _, addr, src = slot
            reads.update({addr, src})
        elif op == "vstore":
            _, addr, src = slot
            reads.add(addr)
            reads.update(range(src, src + VLEN))
    elif engine == "flow":
        if op == "select":
            _, dest, cond, a, b = slot
            writes.add(dest)
            reads.update({cond, a, b})
        elif op == "vselect":
            _, dest, cond, a, b = slot
            writes.update(range(dest, dest + VLEN))
            reads.update(range(cond, cond + VLEN))
            reads.update(range(a, a + VLEN))
            reads.update(range(b, b + VLEN))
        # pause / jump: no scratch deps
    elif engine == "debug":
        if op == "vcompare":
            _, addr, _ref = slot
            reads.update(range(addr, addr + VLEN))
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
    Greedy VLIW scheduler: topological BFS that packs InstrNodes into cycles
    while respecting SLOT_LIMITS. Pauses must be stripped out before calling this.
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
        deferred: deque[int] = deque()

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

        ready = deferred
        for nid in scheduled_now:
            for succ_id in successors[nid]:
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    ready.append(succ_id)

        remaining -= len(scheduled_now)
        result.append(cycle)

    return result


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
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def vec_const(self, val, name=None):
        if val not in self.vec_const_map:
            addr = self.alloc_scratch(name, VLEN)
            self.add("load", ("const", addr, val))
            self.add("valu", ("vbroadcast", addr, addr))
            self.vec_const_map[val] = addr
        return self.vec_const_map[val]

    def build_vhash_slotted(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append({"valu": [(op1, tmp1, val_hash_addr, self.vec_const(val1)),
                                   (op3, tmp2, val_hash_addr, self.vec_const(val3))]})
            slots.append({"valu": [(op2, val_hash_addr, tmp1, tmp2)]})
            slots.append({"debug": [("vcompare", val_hash_addr, [(round, x, "hash_stage", hi) for x in range(i, i+VLEN)])]})
        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        PIPELINE = 8
        assert batch_size % (PIPELINE * VLEN) == 0

        # Allocate per-pipeline-slot scratch so each slot's instructions are
        # independent in the dep graph and the scheduler can interleave them.
        ps = []
        for p in range(PIPELINE):
            ps.append(dict(
                tmp1      = self.alloc_scratch(f"tmp1_{p}",      VLEN),
                tmp2      = self.alloc_scratch(f"tmp2_{p}",      VLEN),
                tmp3      = self.alloc_scratch(f"tmp3_{p}",      VLEN),
                tmp_idx   = self.alloc_scratch(f"tmp_idx_{p}",   VLEN),
                tmp_val   = self.alloc_scratch(f"tmp_val_{p}",   VLEN),
                tmp_nv    = self.alloc_scratch(f"tmp_nv_{p}",    VLEN),
                idx_addr  = self.alloc_scratch(f"idx_addr_{p}",  VLEN),
                val_addr  = self.alloc_scratch(f"val_addr_{p}",  VLEN),
            ))

        # Contiguous block of forest-address pointers, one VLEN entry per pipeline slot.
        # Kept separate so load patterns across slots are easy to inspect and analyze.
        addr_cache_base = self.alloc_scratch("addr_cache", PIPELINE * VLEN)
        for p in range(PIPELINE):
            ps[p]["tmp_addr"] = addr_cache_base + p * VLEN

        # Shared init vars — broadcast scalars into VLEN vectors so valu can use them
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, VLEN)
        # Use two temporaries so we can pack 2 consts + 2 loads + 2 vbroadcasts per
        # triplet cycle, roughly halving init overhead.
        tmp_a, tmp_b = ps[0]["tmp1"], ps[1]["tmp1"]
        for i in range(0, len(init_vars), 2):
            paired = i + 1 < len(init_vars)
            self.instrs.append({"load": [("const", tmp_a, i)] + ([("const", tmp_b, i+1)] if paired else [])})
            self.instrs.append({"load": [("load", tmp_a, tmp_a)] + ([("load", tmp_b, tmp_b)] if paired else [])})
            self.instrs.append({"valu": [("vbroadcast", self.scratch[init_vars[i]], tmp_a)]
                                       + ([("vbroadcast", self.scratch[init_vars[i+1]], tmp_b)] if paired else [])})

        zero_vec = self.vec_const(0)
        one_vec  = self.vec_const(1)
        two_vec  = self.vec_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow",  ("pause",))
        self.add("debug", ("comment", "Starting loop"))

        instrs = []

        for i in range(0, batch_size, PIPELINE * VLEN):
            # Load idx and val once per (i, p) — scratch persists across rounds
            for p in range(PIPELINE):
                offset = i + p * VLEN
                off_c  = self.scratch_const(offset)
                t      = ps[p]

                instrs.append({"alu": [
                    ("+", t["idx_addr"], self.scratch["inp_indices_p"], off_c),
                    ("+", t["val_addr"], self.scratch["inp_values_p"],  off_c),
                ]})
                instrs.append({"load": [
                    ("vload", t["tmp_idx"], t["idx_addr"]),
                ]})
                instrs.append({"load": [
                    ("vload", t["tmp_val"], t["val_addr"]),
                ]})
                instrs.append({"debug": [("vcompare", t["tmp_idx"], [(0, x, "idx") for x in range(offset, offset+VLEN)])]})
                instrs.append({"debug": [("vcompare", t["tmp_val"], [(0, x, "val") for x in range(offset, offset+VLEN)])]})

            for round in range(rounds):
                for p in range(PIPELINE):
                    offset = i + p * VLEN
                    off_c  = self.scratch_const(offset)
                    t      = ps[p]

                    # node_val = mem[forest_values_p + idx]  (scatter load — non-contiguous)
                    instrs.append({"valu": [("+", t["tmp_addr"], self.scratch["forest_values_p"], t["tmp_idx"])]})
                    for j in range(0, VLEN, 2):
                        loc1 = t["tmp_addr"] + j
                        loc2 = t["tmp_addr"] + j + 1
                        instrs.append({"load": [
                            ("load", t["tmp_nv"] + j,     loc1),
                            ("load", t["tmp_nv"] + j + 1, loc2),
                        ]})
                    instrs.append({"debug": [("vcompare", t["tmp_nv"], [(round, x, "node_val") for x in range(offset, offset+VLEN)])]})

                    # val = myhash(val ^ node_val)
                    instrs.append({"valu": [("^", t["tmp_val"], t["tmp_val"], t["tmp_nv"])]})
                    instrs.extend(self.build_vhash_slotted(t["tmp_val"], t["tmp1"], t["tmp2"], round, offset))
                    instrs.append({"debug": [("vcompare", t["tmp_val"], [(round, x, "hashed_val") for x in range(offset, offset+VLEN)])]})

                    # idx = 2*idx + (1 if val % 2 == 0 else 2)
                    instrs.append({"valu": [("&",  t["tmp1"],   t["tmp_val"],  one_vec)]})
                    instrs.append({"valu": [("*",  t["tmp_idx"],   t["tmp_idx"],  two_vec)]})
                    instrs.append({"valu": [("+",  t["tmp1"],   t["tmp1"],  one_vec)]})
                    instrs.append({"valu": [("+",  t["tmp_idx"],   t["tmp_idx"],  t["tmp1"])]})

                    instrs.append({"debug": [("vcompare", t["tmp_idx"], [(round, x, "next_idx") for x in range(offset, offset+VLEN)])]})

                    # idx = 0 if idx >= n_nodes else idx
                    instrs.append({"valu": [("<", t["tmp1"], t["tmp_idx"], self.scratch["n_nodes"])]})
                    instrs.append({"flow": [("vselect", t["tmp_idx"], t["tmp1"], t["tmp_idx"], zero_vec)]})
                    instrs.append({"debug": [("vcompare", t["tmp_idx"], [(round, x, "wrapped_idx") for x in range(offset, offset+VLEN)])]})

            # Commit results to memory once after all rounds
            for p in range(PIPELINE):
                t = ps[p]
                instrs.append({"store": [
                    ("vstore", t["idx_addr"], t["tmp_idx"]),
                    ("vstore", t["val_addr"], t["tmp_val"]),
                ]})

        nodes = build_dep_graph(instrs)
        self.instrs.extend(schedule(nodes))
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
