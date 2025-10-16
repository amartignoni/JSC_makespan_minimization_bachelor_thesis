#!/usr/bin/env python3
import csv
import sys
import numpy as np

def _to_int_safe(x):
    """Convert token to python int if possible, handling '1.0' and np types."""
    try:
        return int(float(x))
    except Exception:
        raise ValueError(f"Cannot convert to int: {x!r}")

def _to_float_safe(x):
    """Convert token to float (start/end may be floats)."""
    try:
        return float(x)
    except Exception:
        raise ValueError(f"Cannot convert to float: {x!r}")

def load_schedule(schedule_csv_path):
    """
    Load schedule file where each row is:
    machine, job, op, start, end, job, op, start, end, ...
    Returns: dict machine -> list of (job:int, op:int, start:float, end:float)
    """
    schedule_by_machine = {}

    with open(schedule_csv_path, newline='') as f:
        reader = csv.reader(f)
        for rownum, raw_row in enumerate(reader, start=1):
            if not raw_row:
                continue
            # strip tokens and remove empty tokens caused by trailing commas
            tokens = [tok.strip() for tok in raw_row if tok is not None and tok.strip() != ""]
            if not tokens:
                continue
            # Try parse first token as machine id; if it's a header (non-numeric), skip row.
            first = tokens[0]
            try:
                machine = _to_int_safe(first)
            except ValueError:
                # skip header / malformed first column
                # You may want to detect and handle a header differently; we skip it here.
                continue

            # ensure machine key present
            if machine not in schedule_by_machine:
                schedule_by_machine[machine] = []

            # remaining tokens should come in groups of 4: job,op,start,end
            rest = tokens[1:]
            if len(rest) == 0:
                # no operations on this machine: continue
                continue

            # process groups of 4; if trailing tokens remain that are incomplete, we ignore them but warn
            if len(rest) % 4 != 0:
                # we'll still parse the full groups and ignore trailing incomplete tokens
                pass

            for i in range(0, len(rest) - 3, 4):
                t_job, t_op, t_start, t_end = rest[i:i+4]
                try:
                    job = _to_int_safe(t_job)
                    op = _to_int_safe(t_op)
                    start = _to_float_safe(t_start)
                    end = _to_float_safe(t_end)
                except ValueError as e:
                    # skip invalid group but include a message
                    print(f"[WARN] Skipping invalid op group on line {rownum}: {e}", file=sys.stderr)
                    continue

                # sanity: start < end
                if not (start <= end):
                    print(f"[WARN] On machine {machine} job {job} op {op} has start>=end ({start} >= {end})", file=sys.stderr)

                schedule_by_machine[machine].append((job, op, start, end))

    # Sort operations on each machine by start time (stable)
    for m in schedule_by_machine:
        schedule_by_machine[m].sort(key=lambda tup: tup[2])

    return schedule_by_machine


def verify_schedule(schedule_by_machine, cg_matrix):
    """
    Verify:
      - job operation order (op k finishes before op k+1 starts)
      - no overlapping ops on same machine
      - conflicting jobs (cg_matrix[i,j]==1) do not overlap in time

    Returns (feasible:bool, messages:list[str])
    """
    feasible = True
    messages = []

    # Flatten all operations: list of (job, op, machine, start, end)
    all_ops = []
    for m, ops in schedule_by_machine.items():
        for job, op, start, end in ops:
            all_ops.append((int(job), int(op), int(m), float(start), float(end)))

    # 1) Job order: group by job
    jobs = {}
    for job, op, m, s, e in all_ops:
        if job not in jobs:
            jobs[job] = []
        jobs[job].append((int(op), float(s), float(e)))

    for job, op_list in jobs.items():
        # sort by operation index
        op_list_sorted = sorted(op_list, key=lambda x: x[0])
        # check sequentiality
        for i in range(len(op_list_sorted) - 1):
            cur_op, cur_s, cur_e = op_list_sorted[i]
            next_op, next_s, next_e = op_list_sorted[i+1]
            # require that op indices are consecutive? we only check order (cur index < next index)
            # ensure the end time of current <= start time of successor
            if cur_e > next_s + 1e-9:  # small epsilon tolerance
                feasible = False
                messages.append(
                    f"Job {job} order violation: op {cur_op} ends at {cur_e} but next op {next_op} starts at {next_s}"
                )

    # 2) Machine overlaps: per machine, sorted by start
    for m, ops in schedule_by_machine.items():
        ops_sorted = sorted(ops, key=lambda x: x[2])
        for i in range(len(ops_sorted) - 1):
            j1, o1, s1, e1 = ops_sorted[i]
            j2, o2, s2, e2 = ops_sorted[i+1]
            # overlap if previous end > next start (allow equality)
            if e1 > s2 + 1e-9:
                feasible = False
                messages.append(
                    f"Machine {m} overlap: job {j1} op {o1} ({s1}–{e1}) overlaps with job {j2} op {o2} ({s2}–{e2})"
                )

    # 3) Conflict graph: any pair of ops from conflicting jobs must not overlap
    num_jobs_in_cg = cg_matrix.shape[0] if isinstance(cg_matrix, np.ndarray) else 0
    n = len(all_ops)
    for i in range(n):
        job1, op1, m1, s1, e1 = all_ops[i]
        for j in range(i+1, n):
            job2, op2, m2, s2, e2 = all_ops[j]
            if job1 == job2:
                continue
            # skip if job index out of bounds for cg_matrix, but warn
            if job1 >= num_jobs_in_cg or job2 >= num_jobs_in_cg:
                messages.append(f"[WARN] Job index out of range for cg_matrix: {job1} or {job2}; skipping conflict check for this pair.")
                continue
            if int(cg_matrix[job1, job2]) == 1:
                # check if intervals overlap (not (e1 <= s2 or e2 <= s1))
                if not (e1 <= s2 + 1e-9 or e2 <= s1 + 1e-9):
                    feasible = False
                    messages.append(
                        f"Conflict violation between job {job1} (op {op1}, {s1}–{e1}) and job {job2} (op {op2}, {s2}–{e2})"
                    )

    return feasible, messages


def main():
    if len(sys.argv) != 3:
        print("Usage: python verify_schedule.py best_schedule.csv cg_matrix.csv", file=sys.stderr)
        sys.exit(2)

    schedule_csv = sys.argv[1]
    cg_csv = sys.argv[2]

    try:
        schedule_by_machine = load_schedule(schedule_csv)
    except Exception as e:
        print(f"[ERROR] Failed to parse schedule file: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        cg_matrix = np.genfromtxt(cg_csv, delimiter=",", dtype=int)
    except Exception as e:
        print(f"[ERROR] Failed to read cg_matrix: {e}", file=sys.stderr)
        sys.exit(2)

    feasible, messages = verify_schedule(schedule_by_machine, cg_matrix)

    if feasible:
        print("[OK] Schedule is feasible ✅")
        sys.exit(0)
    else:
        print("[ERROR] Schedule is NOT feasible ❌")
        for msg in messages:
            print(" -", msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
