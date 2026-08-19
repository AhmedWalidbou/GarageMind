# GarageMind

AI-powered vehicle diagnostic copilot: from raw CAN bus frames to an explainable, context-aware diagnosis.

GarageMind reads a vehicle's low-level data (CAN bus, OBD-II, UDS), decodes it into named physical signals, reads stored fault codes together with the conditions under which they occurred, and produces a prioritized diagnostic report. The end goal is an agentic assistant that reasons about faults the way an experienced mechanic does, cross-referencing live signals, fault codes and repair knowledge.

Status: Module 1 (Scan Engine) is complete and tested. Module 2 (Anomaly Benchmark Lab) has a full two-model benchmark on the four HCRL attack types, with stealthier datasets and sequence models planned as extensions. Module 3 (EBR-RAG) benchmarks three retrieval systems over a curated bilingual repair-case base. Modules 4 and 5 are in progress; see the roadmap.

## Motivation

Modern vehicles expose a continuous stream of low-level data, but turning it into an actionable diagnosis requires two rarely combined skill sets: automotive protocol engineering (CAN, DBC, UDS/ISO 14229, ISO-TP) and applied AI (anomaly detection, retrieval-augmented generation, agents). GarageMind is built at that intersection.

The diagnostic workflow mirrors the logic of real industry tools such as Bosch ESI[tronic] and KTS testers: identify the vehicle, scan the electronic control units, read fault codes with the freeze-frame conditions under which they were set, and produce prioritized repair guidance. GarageMind reproduces that workflow on open data and adds an AI reasoning layer on top of it.

A trouble code on its own is not a diagnosis. The same code can have different root causes depending on the operating conditions when it appeared. Capturing and reasoning about that context is the central idea of the project.

## Project structure (five modules)

| Module | Name | Status | Scope |
|--------|------|--------|-------|
| M1 | Scan Engine | Complete | CAN/DBC decoding, UDS diagnostics, unified diagnostic report |
| M2 | Anomaly Benchmark Lab | Benchmarked (2 models) | LSTM autoencoder vs classical baseline on the 4 HCRL attacks; sequence models on stealthier data planned |
| M3 | EBR-RAG | Benchmarked (3 systems) | Retrieval over curated repair cases; dense, BM25 and hybrid RRF compared |
| M4 | Diagnostic Agent | Planned | LangGraph agent that asks clarifying questions and reasons over evidence |
| M5 | Interface and Edge | Planned | Dashboard and a quantized edge SLM for on-device inference |

## Module 1: Scan Engine

The foundation of the project. It turns unreadable raw frames into a structured, correlated diagnostic report.

### Components

VIN decoder. Decodes manufacturer, model year and country of origin from the 17-character VIN, using local World Manufacturer Identifier tables with the NHTSA public API as an enrichment fallback.

CAN parser. Reads raw CAN logs in the HCRL Car-Hacking format, computes bus-level statistics (frame rate, unique identifiers, per-ID frequency), and surfaces anomalies directly. For example, the injected high-priority 0x0000 flood used in denial-of-service attacks stands out immediately in the identifier distribution.

Multi-brand DBC decoder. Translates raw payload bytes into named physical signals using OpenDBC databases. The decoder is brand-agnostic by construction: the same engine decodes any manufacturer once the corresponding DBC is supplied. It has been exercised on Hyundai, Toyota, Volkswagen, Ford and Tesla databases, and loads malformed DBC files in a permissive mode instead of failing the whole scan.

Signal extraction pipeline. Automatically selects the best-matching DBC by measuring identifier coverage against the observed traffic, extracts full time-series for every decoded signal, validates values against physical plausibility ranges, and exports the result to Parquet for downstream modules.

DTC reader. Interprets OBD-II diagnostic trouble codes with structural decoding of the code itself (system, standardized versus manufacturer-specific, subsystem) and enriches known codes with bilingual French and English descriptions, severity ranking and likely causes.

UDS stack. A faithful implementation of the real diagnostic protocol chain:

- ISO-TP (ISO 15765-2) transport layer: segmentation of long messages into CAN frames, reassembly, sequence-number validation and flow control.
- UDS ECU simulator (ISO 14229): DiagnosticSessionControl, ReadDataByIdentifier, ReadDTCInformation with multiple sub-functions, ClearDiagnosticInformation, and freeze-frame snapshots stored per fault.
- UDS client: runs a complete diagnostic scan through the ISO-TP layer and interprets the results, including per-fault status bits.

Report generator. Unifies every component into a single report containing vehicle identity, an overall health score, a severity-ordered repair plan with indicative effort estimates, and a quantitative correlation between each fault's freeze-frame conditions and the observed live-signal distribution. Reports are exported in both JSON (machine-readable, consumed by later modules) and Markdown (human-readable).

### Context-aware diagnosis

The report links each fault to the conditions under which it was recorded and checks those conditions against the live data actually observed in the log:

[P0301] Cylinder 1 misfire
Context: at idle, engine warm, vehicle stationary
RPM = 617 vs observed live range [596, 662], z = 0.01 -> consistent with the log

[P2002] Diesel particulate filter efficiency below threshold
Context: high load, engine warm, moving at 70 km/h
RPM = 2400 vs observed live range [596, 662] -> conditions absent from this log


The correlation layer is deliberately conservative: it only computes a z-score when the signal actually varies, and reports a constant-signal note otherwise, rather than emitting a meaningless statistic.

## Module 2: Anomaly Benchmark Lab

Benchmarks anomaly-detection models on real CAN intrusion data (HCRL Car-Hacking dataset: DoS, Fuzzy, gear spoofing, RPM spoofing), under a protocol designed so that results are comparable and reproducible.

### Protocol

Every model is evaluated on exactly the same data under the same rules:

- Frames are parsed with a variable-DLC-aware loader (the flag is the last field of each row, not a fixed column), then enriched with per-frame features: log1p inter-arrival time, rolling CAN-ID frequency, DLC and the 8 zero-padded payload bytes.
- The train/test split is temporal and performed on frames before windowing. Overlapping sliding windows (64 frames, stride 32) would leak identical frames into both sets under a random split.
- Models are calibrated label-free: thresholds are percentiles of the anomaly scores on normal training windows only, with the same ~2 percent false-alarm budget for every model.
- One fixed seed, identical windows for every model, shared metric code. Re-running the benchmark reproduces the published numbers to the fourth decimal.

### Models

LSTM autoencoder with bilateral thresholds. A seq2seq autoencoder trained on normal traffic only, scoring windows by reconstruction error. The naive rule "attack = high reconstruction error" failed spectacularly on DoS (ROC-AUC 0.056, a near-perfectly inverted ranking): a flood hammers one ID with constant payloads, producing traffic more regular than normal and therefore easier to reconstruct. The corrected rule is bilateral: a window is anomalous if its error leaves the normal P1-P99 band in either direction. That failure analysis is documented in the module docstring and kept in the code history.

Isolation Forest baseline. A classical model on aggregated window statistics (mean, std, min, max per feature). Temporal order inside the window is deliberately discarded: the baseline measures how much of the signal is captured by distributional statistics alone.

### Results (500k frames per dataset, seed 42)

| Model | Attack | Precision | Recall | F1 | ROC-AUC |
|-------|--------|-----------|--------|-----|---------|
| lstm_ae | DoS | 0.956 | 0.599 | 0.7365 | 0.9073 |
| iforest | DoS | 0.9933 | 0.9949 | 0.9941 | 0.9995 |
| lstm_ae | Fuzzy | 0.9894 | 0.9918 | 0.9906 | 0.9991 |
| iforest | Fuzzy | 0.9847 | 0.9995 | 0.9921 | 0.9999 |
| lstm_ae | Gear | 0.9644 | 0.3615 | 0.5259 | 0.869 |
| iforest | Gear | 0.9886 | 0.9993 | 0.9939 | 0.9998 |
| lstm_ae | RPM | 0.9909 | 1.0 | 0.9954 | 1.0 |
| iforest | RPM | 0.9834 | 0.9996 | 0.9914 | 0.9997 |

### Findings

The headline result is that the 40-line classical baseline outperforms the LSTM autoencoder on every HCRL attack, including the autoencoder's weak spot (gear spoofing: F1 0.99 versus 0.53). The mechanism is clear: HCRL attacks are massive injections, so the feature distribution inside an attacked window is violently shifted and aggregated statistics suffice; the autoencoder averages its reconstruction error over the window and is diluted by mixed windows. This matches the published observation that HCRL attacks are easy enough that they do not discriminate between detection methods.

The honest conclusion, and the design principle for the rest of the module: sequence models must justify their complexity, and they can only do so on stealthy attacks. The planned extension evaluates the same protocol on the ROAD dataset (ORNL), whose flam-delivery fabrication and masquerade attacks inject single frames with legitimate IDs, precisely the regime where distributional statistics are expected to fail.

Full metrics, configurations and confusion matrices are versioned in `results/benchmark_attacks.json`.

### Reproducing

```bash
# Train and evaluate the LSTM AE on DoS with saved model and metrics
python -m src.anomaly.train_lstm_ae

# Full multi-model benchmark on all four attacks
python -m src.anomaly.evaluate_attacks
```

## Module 3: Experience-Based Reasoning (EBR-RAG)

Retrieves relevant repair cases from a curated bilingual knowledge base, so the diagnostic agent (Module 4) reasons over documented cases rather than over model priors alone. Three retrieval systems are benchmarked against each other under one shared harness.

### Protocol

- The knowledge base holds 20 curated repair cases (14 vehicle systems: DPF, EGR, PureTech timing, turbo, AL4 gearbox, MAF, glow plugs, injectors, CAN network faults). Each case is validated at load time: malformed DTC codes, missing fields or duplicate ids are rejected rather than silently indexed.
- Each case is flattened into two monolingual documents (fr, en), 40 in total. Mixed-language documents pull embeddings toward the middle of both languages; monolingual ones do not.
- The evaluation set holds 25 queries phrased as a mechanic or a customer would ask them, deliberately not reusing corpus wording, so metrics measure retrieval rather than string overlap. Five queries accept several cases, two are generic DTC questions.
- Hit@k counts a query as solved at k if at least one acceptable case id appears in the top-k unique cases; MRR uses the rank of the first relevant case. Results are deduplicated by case id, keeping the best-scoring language variant.
- One harness scores all three systems: they expose the same `retrieve(query, top_k)` interface, so protocol divergence between them is impossible by construction.
- Raw scores are never compared across systems. Cosine similarities, BM25 scores and RRF sums live on different scales; only ranks and rank-based metrics are compared.

### Systems

Dense retrieval. `intfloat/multilingual-e5-small` (384 dimensions) over an embedded Qdrant index, cosine distance on L2-normalized vectors. The e5 asymmetric prefixes (`query:` / `passage:`) are applied in a single place in the code, since forgetting them silently degrades retrieval. Search runs across all 40 documents with no language filter: the multilingual space maps FR and EN to the same region, verified empirically before the design was adopted.

BM25 baseline. `BM25Okapi` over the same 40 documents, tokenized with lowercasing, NFKD accent stripping and alphanumeric runs, so DTC codes survive as whole tokens. Following the Module 2 principle, the neural system is only worth its cost if it beats a classical one. BM25 has no cross-lingual ability by construction: a French query only matches French tokens plus language-independent anchors.

Hybrid RRF. Reciprocal Rank Fusion of both rankings, `1/(60 + rank)` summed per case. The constant is the standard value from the literature and is deliberately left untuned: tuning it on the 25-query set would turn the evaluation into training data. Fusion was implemented only after complementarity was measured, not assumed.

### Results (25 queries, top_k=5, index of 40 documents)

| System | Hit@1 | Hit@3 | Hit@5 | MRR | Latency mean/p95 (ms) |
|--------|-------|-------|-------|-----|------------------------|
| dense (e5-small) | 0.88 | 0.96 | 1.00 | 0.9280 | 24 / 28 |
| bm25 | 0.92 | 1.00 | 1.00 | 0.9600 | < 1 / < 1 |
| hybrid (RRF) | 1.00 | 1.00 | 1.00 | 1.0000 | 23 / 30 |

Per-language slices are symmetric for the hybrid system (1.00 on both the 18 French and the 7 English queries). Full protocol, per-query rankings and latencies are versioned in `results/retrieval_benchmark.json`.

### Findings

BM25 beats the dense retriever on this corpus, which contradicts the prediction registered before the run. The mechanism: workshop vocabulary is narrow (mechanics and cases share a few hundred terms), case texts are long enough to offer many matching opportunities, and DTC codes act as near-unique lexical anchors. Paraphrasing alone does not defeat lexical matching on this terrain. The dense retriever's two closest failures were a pure DTC question and a photo-finish separated by less than 0.001 in cosine similarity, both symptoms of the same cause: e5 similarities are heavily compressed on this corpus, so only ranks are meaningful, never absolute scores.

The fusion result is stronger than a simple union of successes. Three queries were solved at rank 1 by exactly one system, and the hybrid captured all three; but query q-023 was solved at rank 1 by neither system, and the hybrid solves it. Consensus across two independent rankings outranks a single confident vote: a case placed second by both systems scores 2/62, above a case placed first by one system only at 1/61. Fusion produces a result neither component produced, at the cost of fifteen lines of code and no measurable latency, since the dense embedding dominates and BM25 is free.

The perfect score is a limitation, not an achievement. Hit@1 of 1.00 on 25 queries means the evaluation set has reached its resolution limit: no further improvement to the retriever could be measured with it. Any next step on this module therefore starts by widening the corpus and the query set, not by tuning the retrievers. Three follow-ups are deliberately deferred until then: metadata filtering by brand and vehicle system, calibration of a "no relevant case" threshold (which the compressed cosine range currently makes unreliable), and an ablation on a larger embedding model.

### Reproducing

```bash
# Build the persistent Qdrant index from the knowledge base
python scripts/build_index.py

# Benchmark dense, BM25 and hybrid retrieval on the 25 evaluation queries
python scripts/evaluate_retrieval.py
```

## Installation

```bash
git clone https://github.com/AhmedWalidbou/GarageMind.git
cd GarageMind
conda create -n garagemind python=3.11 -y
conda activate garagemind
pip install -e .
```

### Datasets

The datasets are not bundled with the repository and must be obtained separately.

CAN traffic: the HCRL Car-Hacking dataset (CAN logs captured from a Hyundai vehicle over the OBD-II port). Place the four CSV files (DoS_dataset.csv, Fuzzy_dataset.csv, gear_dataset.csv, RPM_dataset.csv) in data/raw/.

DBC files: the OpenDBC project by comma.ai, cloned into data/dbc/.

## Usage

```bash
# Full diagnostic scan, producing JSON and Markdown reports
garagemind scan

# Decode a VIN using local tables only
garagemind decode-vin WAUZZZ8V1JA123456 --no-api

# Analyze a raw CAN log, forcing a given brand's DBC databases
garagemind analyze-can data/raw/DoS_dataset.csv --brand hyundai

# Show the version
garagemind version
```

The scan command prints a health score from 0 to 100, a severity-ranked repair plan with indicative effort estimates, and the per-fault freeze-frame correlation, then writes the full report to disk.

## Results on real data

Running the extraction pipeline on the HCRL dataset produces the following, using only open data and the OpenDBC databases:

- Automatic DBC selection identifies hyundai_i30_2014.dbc as the best match at 80 percent CAN-identifier coverage, compared with 28 percent for a wrong-variant database. The automatic selection is what makes the difference.
- Roughly 863,000 datapoints are extracted across 252 named signals.
- Decoded values are physically consistent: engine speed around 617 rpm at idle, coherent torque values, realistic temperatures.
- The physical-validation layer distinguishes genuine measurements from non-populated sensors. An unfilled fuel-temperature signal reading its scale floor is flagged rather than reported as a real minus-forty-eight-degree reading.

These figures come from a generic DBC applied to a different vehicle variant than the one that produced the log, which is precisely why coverage is partial and why the tooling reports coverage explicitly.

## Testing

```bash
pytest -v
```

The suite contains 184 unit tests covering DTC encoding and interpretation, the ISO-TP transport layer including sequence-error detection, the UDS ECU simulator and client, the report generator including the zero-variance correlation edge case, the command-line interface including exit codes and input validation, the anomaly preprocessing pipeline (temporal-split leakage guards, window labeling at the threshold boundary), the LSTM autoencoder (architecture, training, bilateral calibration, end-to-end detection of both irregular and flood-like synthetic attacks), the Isolation Forest baseline (aggregation verified value by value, determinism under a fixed seed), and the retrieval stack (knowledge-base validation at the boundary, e5 prefix handling with a fake model, idempotent Qdrant re-indexing, case-level deduplication, BM25 tokenization including accent stripping and DTC-code preservation, and RRF fusion arithmetic verified by hand).

## Technology and standards

Python 3.11, cantools, pandas, numpy, pyarrow, matplotlib, torch (CPU), scikit-learn, sentence-transformers, qdrant-client, rank-bm25, pytest.

Standards implemented: ISO 11898 (CAN), ISO 15765-2 (ISO-TP), ISO 14229 (UDS), and OBD-II diagnostic trouble codes.

## Roadmap

Module 2 extensions: evaluate the same benchmark protocol on the ROAD dataset (ORNL), whose flam-delivery and masquerade attacks are the stealthiest publicly available, then introduce sequence models (Transformer-based detectors) that have to beat the classical baseline there to earn their place.

Module 3 extensions: widen the knowledge base and the evaluation set, since a perfect Hit@1 on 25 queries means the current set can no longer measure progress. Metadata filtering by brand and system, a calibrated "no relevant case" threshold and a larger-embedding ablation are deferred until then.

Module 4 will introduce a LangGraph diagnostic agent that actively requests missing information and reasons jointly over live signals, fault codes and retrieved documentation, using the retrieval stack of Module 3 as a tool.

Module 5 will provide an interactive dashboard and a quantized small language model for on-device, offline inference.

## Note on data and simulation

The CAN decoding and signal extraction operate on real recorded vehicle traffic. The UDS diagnostic exchange runs against a simulated ECU that responds according to ISO 14229, because the public CAN dataset does not contain diagnostic-session traffic. This separation is intentional and stated explicitly so that nothing in the pipeline is presented as more than it is.

## License

MIT