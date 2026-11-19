# Models Catalog Reference

The app comparison story is seeded by `src/app/models.json`. Operators can
override the active model selection in the app; the file remains the default
catalog and the source of explicit contract and inference-option mappings.

## What It Controls

- which endpoints appear in the UI before an operator-saved selection exists
- whether the endpoint is treated as `fmapi` or `custom`
- which structured-output contract is used
- endpoint-specific settings such as token limits or timeout values

## Showcase Use

For customer demos, this file is the bridge between:

- the decision framework in `showcase/decision-framework.md`
- the comparison cockpit in `src/app`

In practice:

- FMAPI entries show the managed baseline
- custom entries show domain-model or fine-tuned flexibility
- contract settings show that one app can speak to multiple serving interfaces

Selecting an endpoint in the UI does not grant the app access to it. Add every
endpoint used by the demo to `resources.apps.clinical_extractor.resources` in
`resources/clinical_extractor.app.yml` with `CAN_QUERY` before deployment.
