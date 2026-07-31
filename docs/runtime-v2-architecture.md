# Runtime V2 architecture

Runtime V2 replaces import-time source-pipeline patching with explicit core modules.

## Validated

- `TaskProfile` selects generic, crawler, or repository-task policy without site schemas.
- `SourceResponseCache` caches canonical raw responses and derives text, JSON, and header views locally.
- `ArtifactSemanticContract` and its validator cover declared shape, counts, fields, types, URLs, uniqueness, and ordering.
- `VerificationScheduler` invalidates producer and dependent validator evidence on each edit epoch.
- `RepairCoordinator` stops a third identical failure as non-convergent.
- `RepositoryIndex`, `RepositoryPlanner`, and `TaskGraph` provide bounded multi-language repository context.
- `FinalizationController` represents explicit final gates.
- Importing `docode` has no runtime patch side effects.

## Experimental

- The new controllers exist as independently tested components. `CodingAgentLoop` still contains legacy orchestration and has not yet been reduced to the intended thin shell.
- Artifact contracts are integrated into QualityGate for explicit required and absolute-URL fields; full scheduler/finalization wiring remains incomplete.

## Not supported as a release claim

- No frozen 8-case crawler plus 3-case large-repository Runtime V2 suite has completed.
- Runtime V2 release thresholds have not been met.
