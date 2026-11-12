- **BLOCKER — sft/g2b_chain.sh:38:** The whole-chain cap allows 763 minutes despite the registered roughly 6-hour limit. Audits and judging can continue hours after authoring ends. **Fix:** enforce a 6-hour deadline across the entire chain.

- **BLOCKER — sft/g2b_chain.sh:162–167:** The continuation gate checks deadline losses but never checks API rejects. Authoring can exceed the registered 2% API-error threshold and still proceed into judging, producing a bank from a run that should have stopped. **Fix:** reject aggregate API errors above 2% of planned slots before downstream work.

Not ready for the smoke until these launch controls match the preregistration.