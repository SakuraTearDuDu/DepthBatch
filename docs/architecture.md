# Architecture

DepthBatch is organized around four stable layers:

1. `providers/`
   Encodes upstream model semantics. For V1 this means Depth Anything V2 presets, preprocessing, postprocessing, and license metadata.

2. `backends/`
   Loads executable sessions and performs inference or export. Backends do not scan files or write artifacts.

3. `pipelines/`
   Orchestrates run-level workflows such as image inference, video inference, ONNX export, benchmarking, and inspection.

4. `io/` and `manifests/`
   Resolves inputs, writes outputs, and records reproducible run metadata.

## Canonical Data Flow

```text
input resolver
  -> job planner
  -> provider preprocess
  -> backend infer/export
  -> provider postprocess
  -> artifact writer
  -> manifest recorder
```

## Provider Boundary

Only the provider layer knows:

- model aliases such as `da-v2-small`
- encoder presets
- official preprocessing details
- render defaults and license notes

Everything else treats the provider as a contract.

## Backend Boundary

Backends expose:

- `capabilities()`
- `open()`
- `infer()`
- `inspect()`
- `close()`

Export-capable backends also expose `export_onnx()`.

## Run Layout

Each run writes a fixed root layout:

```text
<run-root>/
  manifest.json
  items.jsonl
  resolved-config.yaml
  environment.json
  artifacts/
  reports/
```
