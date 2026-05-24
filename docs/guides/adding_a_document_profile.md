# Adding a document profile

A document profile says how evidence is *shaped*. Statutes are prose; tariff
schedules are tables; treaty memberships are lookups. Each profile is one Python
class implementing `DocumentProfileStrategy` (see
`packages/rie-profiles/src/rie_profiles/base.py`).

## Step-by-step

1. **Create a new strategy file** under
   `packages/rie-profiles/src/rie_profiles/strategies/<your_profile>.py`. Implement:
   - `profile: DocumentProfile = field(default=DocumentProfile.YOUR_PROFILE)`
   - `child_chunk_size()`, `child_chunk_overlap()`
   - `applicable_element_types()`
   - `supports_llm_classification()` — return False for lookup-only profiles
   - `post_extract(meta, elements, edges)` — profile-specific normalisation
2. **Register it** in
   `packages/rie-profiles/src/rie_profiles/strategies/__init__.py`.
3. **Add the enum value** in `packages/rie-contracts/src/rie_contracts/models.py`
   `DocumentProfile`. NOTE: contracts are frozen — adding a new enum value is a
   permitted, additive change but requires the freeze-tag bump signed off by the
   lead.
4. **Update the JSON schema** in `pillars/_schema/pillar.schema.json` to list the
   new profile in the `document_profile` enum.
5. **Add a regression test** in `packages/rie-profiles/tests/` covering at least
   the chunk_size / chunk_overlap / supports_llm_classification expectations and
   the post_extract identity behaviour.

## Hard rules

- Strategies must be stateless beyond constructor args.
- Strategies must NOT import any other adapter package — they live above the
  adapter line and are wired in by `rie-extract` and `rie-retrieval`.
- A profile strategy with `supports_llm_classification() = False` means the
  orchestrator skips the classifier entirely for that indicator and instead
  routes to a profile-specific evaluator (e.g. tariff lookup table, treaty
  membership list).
