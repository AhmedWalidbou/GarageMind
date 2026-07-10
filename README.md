# GarageMind

AI-powered vehicle diagnostic copilot: from raw CAN bus frames to an explainable, context-aware diagnosis.

GarageMind reads a vehicle's low-level data (CAN bus, OBD-II, UDS), decodes it into named physical signals, reads stored fault codes together with the conditions under which they occurred, and produces a prioritized diagnostic report. The end goal is an agentic assistant that reasons about faults the way an experienced mechanic does, cross-referencing live signals, fault codes and repair knowledge.

Status: Module 1 (Scan Engine) is complete and tested. Modules 2 to 5 are in progress; see the roadmap.

## Motivation

Modern vehicles expose a continuous stream of low-level data, but turning it into an actionable diagnosis requires two rarely combined skill sets: automotive protocol engineering (CAN, DBC, UDS/ISO 14229, ISO-TP) and applied AI (anomaly detection, retrieval-augmented generation, agents). GarageMind is built at that intersection.

The diagnostic workflow mirrors the logic of real industry tools such as Bosch ESI[tronic] and KTS testers: identify the vehicle, scan the electronic control units, read fault codes with the freeze-frame conditions under which they were set, and produce prioritized repair guidance. GarageMind reproduces that workflow on open data and adds an AI reasoning layer on top of it.

A trouble code on its own is not a diagnosis. The same code can have different root causes depending on the operating conditions when it appeared. Capturing and reasoning about that context is the central idea of the project.

## Project structure (five modules)

| Module | Name | Status | Scope |
|--------|------|--------|-------|
| M1 | Scan Engine | Complete | CAN/DBC decoding, UDS diagnostics, unified diagnostic report |
| M2 | Anomaly Benchmark Lab | In progress | LSTM autoencoder vs Transformers vs foundation models on CAN traffic |
| M3 | EBR-RAG | Planned | Retrieval over DTC definitions, repair manuals and solved cases |
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

```
[P0301] Cylinder 1 misfire
  Context: at idle, engine warm, vehicle stationary
  RPM = 617 vs observed live range [596, 662], z = 0.01  -> consistent with the log

[P2002] Diesel particulate filter efficiency below threshold
  Context: high load, engine warm, moving at 70 km/h
  RPM = 2400 vs observed live range [596, 662]  -> conditions absent from this log
```

The correlation layer is deliberately conservative: it only computes a z-score when the signal actually varies, and reports a constant-signal note otherwise, rather than emitting a meaningless statistic.

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

CAN traffic: the HCRL Car-Hacking dataset (CAN logs captured from a Hyundai vehicle over the OBD-II port), available on Kaggle. Place the CSV files in data/raw/.

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

The suite contains 52 unit tests covering DTC encoding and interpretation, the ISO-TP transport layer including sequence-error detection, the UDS ECU simulator and client, the report generator including the zero-variance correlation edge case, and the command-line interface including exit codes and input validation.

## Technology and standards

Python 3.11, cantools, pandas, numpy, pyarrow, matplotlib, pytest.

Standards implemented: ISO 11898 (CAN), ISO 15765-2 (ISO-TP), ISO 14229 (UDS), and OBD-II diagnostic trouble codes.

## Roadmap

Module 2 will benchmark anomaly-detection approaches on the CAN traffic, comparing an LSTM autoencoder baseline against Transformer-based detectors and zero-shot time-series foundation models, and will make use of the dataset's real attack scenarios (denial of service, fuzzing, spoofing).

Module 3 will add retrieval-augmented generation over repair manuals, DTC definitions and solved repair cases, in the spirit of experience-based repair systems used in professional workshops.

Module 4 will introduce a LangGraph diagnostic agent that actively requests missing information and reasons jointly over live signals, fault codes and retrieved documentation.

Module 5 will provide an interactive dashboard and a quantized small language model for on-device, offline inference.

## Note on data and simulation

The CAN decoding and signal extraction operate on real recorded vehicle traffic. The UDS diagnostic exchange runs against a simulated ECU that responds according to ISO 14229, because the public CAN dataset does not contain diagnostic-session traffic. This separation is intentional and stated explicitly so that nothing in the pipeline is presented as more than it is.

## License

MIT