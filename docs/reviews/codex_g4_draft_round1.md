- **BLOCKER — [G4_draft.md:41](docs/DISTILLATION_G4_PREREGISTRATION_DRAFT.md:41):** Requiring screening results to “exist” permits a failed candidate onto the BFCL 200. This violates handoff §4’s requirement that the screen pass before protected-bank evaluation; G3c permits retraining even after failed initial screens.  
  **Fix:** Require the selected final model to pass both registered FK gates: run 1 and restarted repeat each ≥70/155.

- **BLOCKER — [G4_draft.md:11](docs/DISTILLATION_G4_PREREGISTRATION_DRAFT.md:11):** The adapter is identified only by arm and scale. Nothing binds evaluation to the screened checkpoint, so the retained pass-1 adapter or another conversion could silently be scored as the final model.  
  **Fix:** Before launch, freeze checkpoint and LoRA GGUF paths/hashes, bind them to the screening reports, and require runtime verification of the loaded adapter and scale, with no adapter on comparator servers.

- **BLOCKER — [G4_draft.md:22](docs/DISTILLATION_G4_PREREGISTRATION_DRAFT.md:22):** Time estimates have no hard per-run or whole-chain deadline, and no required shutdown on completion, failure, cancellation, or timeout. A stalled evaluation could hold the GPU/8080 indefinitely; handoff §4 explicitly requires registered caps and stop conditions.  
  **Fix:** Register enforced deadlines and require cleanup of evaluator/server processes on every exit, verifying that GPU resources and port 8080 are released.

Not ready: these launch requirements need to be fixed before approval.