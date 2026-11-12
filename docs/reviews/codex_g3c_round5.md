- **BLOCKER — sft/g3c_arm.sh:24:** The Nov 25 cutoff exists only in the waiter. A direct `run` invocation with A-005 and existing valid G4/backup paths passes these gates after Nov 25 and starts training, violating preregistration §5. **Fix:** enforce the cutoff inside `g3c_arm.sh` before starting GPU work.

Not ready until the run entry point enforces the cutoff.