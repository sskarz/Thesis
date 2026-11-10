# Codex review round 2 of the distillation data (mlx-t2), 2026-11-06

Reviewer: Codex CLI 0.167.0, gpt-6.1-sol, medium effort, read-only, env not inherited. Verdict: NO-GO (contamination, config and format verified OK; semantic and grounding defects remained). Addressed in mlx-t3.

**NO-GO.** The frozen dataset passes contamination and format checks, but still contains incorrect supervision. Do not launch training or the gate yet.

I reviewed all 156 targets and 26 source tasks, replayed their 183 gold calls offline, and checked retained actions against logged execution results. No files were modified.

Row numbers below refer to [`mlx-t2-null156/train.jsonl`](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/sft/data/distill/mlx-t2-null156/train.jsonl).

1. **Critical — Previous finding 1: PARTLY fixed. Wrong gold outcomes remain.**  
   The nine quarantines remove the original sibling-directory cases, including `0103`. However, [`semantic_issue()`](/Users/sanskarthapa/Documents/GitHub/Thesis-wt-distill/benchmark/distill_tasks.py:233) skips destinations containing dots:
   - Task **0067**, retained **row 115**, copies `ep_015.txt` to `destination='..'`. My replay created a **file named `..` inside `episodes`**, leaving no backup in `wavelength`.
   - Task **0098** gold and teacher both execute `mv(guest_list.txt, '..')`. Replay **removes the original name and creates `clients/..`**, rather than moving into `wedding_planner`. Only its earlier rows **155–156** enter this subset; the source episode remains semantically invalid.
   - Task **0005**, **row 36**, requests three decimal places, but `square_root(..., precision=3)` sets **three significant digits** and returns **3.84**. Gold has the same defect.

   **Fix:** quarantine these episodes; validate actual destination paths, source preservation/removal, and numerical precision against the request. Reject unsupported parent-directory transfers.

2. **Important — Previous finding 2: PARTLY fixed. Failed calls are removed; incorrect successful calls survive.**  
   I independently reproduced **16 failed-target drops**, **309 duplicate drops**, and **494 pre-cap rows** across 97 episodes. The frozen subset contains **zero retained action targets with logged execution errors**, and none lacks a subsequent observation. Nevertheless:
   - **Row 35 / task 0005** batches conversion with `square_root(number=4.5)`: the user requested the square root of the **converted feet**.
   - **Row 125 / task 0165** batches division with an unobserved, approximated `2.723`, then rounds to **3 shares**, exceeding the requested $600 budget. Row 126 subsequently buys 2.
   - **Task 0105 gold / row 92** uses statistics returning **received_count** to answer how many messages were **sent**; teacher row 93 retrieves sent messages afterward.

   **Fix:** reject incorrect intermediate targets despite later recovery; correct gold and regenerate affected episodes.

3. **Important — Previous finding 3: PARTLY fixed.**  
   Literal overlap is persisted and PLAN wording is narrower, but grounding remains heuristic. Retained filename guesses without prior lookup remain at **rows 57, 61, 63 and 66**: `itinerary.txt`, `packing_list.txt`, `snacks.txt`, `gas_receipt.txt`. Row 63 contains two guesses.  
   **Fix:** require filename discovery before these targets. Review numerical dependencies separately from literal matching.

4. **Important — Previous finding 4: FIXED for reproducibility.**  
   The frozen train file exactly matches `mlx-t2/train_156.jsonl`; validation matches `mlx-t2/valid.jsonl`, including recorded hashes. I counted **156 train / 33 valid**, **26 training tasks**, **33 stop targets**, and **184 calls**.

   Against **800 excluded IDs / 1,153 unique turns**:
   - ID, exact-turn and shared-8-gram hits: **0**, separately for train and valid.
   - ≥0.85 similarity hits: **0**, including **381 distinct history user messages**.
   - Maximum source-turn similarity: **0.7957 train; 0.5714 valid**.
   - Train/valid task overlap, duplicate message sequences within either split, and exact cross-split overlap: **all 0**.

5. **Important — Previous finding 5: FIXED in YAML; preserve it at launch.**  
   Parsed YAML equals the readable Run 73 config **except adapter_path**. Installed MLX-LM updates only when `iteration % accumulation == 0`: **156 microbatches / 4 = 39 optimizer updates**. All rows have a messages list ending in an assistant target. `ChatDataset` masks the entire preceding history. Cached template/tokenizer rendering verified **189 rows**, maximum **7,936 tokens**, zero truncation or masking-prefix failures.  
   **Launch directly with the frozen data/config:** the default `sft/train_mlx.py` entry point rebuilds data and supplies older config/CLI overrides.

6. **Minor — Previous finding 6: PARTLY fixed.**  
   Acceptance-time deduplication now runs under the lock. A **process-level** nonce does **not** isolate repeated identical `solve_one()` calls within that process; author validation IDs also remain reusable. **Fix:** namespace each invocation.

7. **Minor — Previous finding 7: PARTLY fixed.**  
   PLAN wording improved; the report’s “All fixed” claim is incorrect. Before stage 4, freeze base artifacts, paired pass IDs, regression accounting for both banks, and distinct candidate identifiers. Apply handoff §6 unchanged: **≥17/20**, fresh-bank ≥base, ≤1 regression, then fail-fast without gate-driven variants.