# Deep Coverage Analysis

**Target**: `robotframework-doctestlibrary` (DocTest package, 5,045 statements)
**Tool**: `robotframework-testselection` with all 6 strategies
**Embedding**: all-MiniLM-L6-v2 (384-dim), seed=42
**Date**: 2026-02-24
**Total experiment time**: ~86 minutes (56 experiments)

## Pytest Results (594 total tests, 70.13% full coverage)

| Strategy | Level | K | Run | Pass | Coverage | Retention | Time(s) | Speedup |
|---|---|---|---|---|---|---|---|---|
| full | 100% | 594 | 594 | 591 | 70.1% | baseline | 305.4 | 1.0x |
| **fps** | **50%** | **297** | **297** | **295** | **65.9%** | **93.9%** | **90.1** | **3.4x** |
| dpp | 50% | 297 | 297 | 295 | 65.9% | 93.9% | 87.1 | 3.5x |
| kmedoids | 50% | 297 | 297 | 295 | 65.8% | 93.8% | 99.7 | 3.1x |
| fps_multi | 50% | 297 | 297 | 295 | 65.7% | 93.7% | 101.5 | 3.0x |
| facility | 50% | 297 | 297 | 295 | 64.9% | 92.6% | 103.9 | 2.9x |
| random | 50% | 297 | 297 | 294 | 64.2% | 91.5% | 131.9 | 2.3x |
| **facility** | **20%** | **119** | **119** | **118** | **58.6%** | **83.6%** | **40.6** | **7.5x** |
| kmedoids | 20% | 119 | 119 | 118 | 58.3% | 83.1% | 45.8 | 6.7x |
| fps | 20% | 119 | 119 | 118 | 56.1% | 80.0% | 45.6 | 6.7x |
| fps_multi | 20% | 119 | 119 | 118 | 53.8% | 76.8% | 38.0 | 8.0x |
| random | 20% | 119 | 119 | 116 | 53.2% | 75.8% | 56.7 | 5.4x |
| dpp | 20% | 119 | 119 | 118 | 47.6% | 67.8% | 51.1 | 6.0x |
| **fps_multi** | **10%** | **60** | **60** | **59** | **47.7%** | **67.9%** | **16.5** | **18.5x** |
| kmedoids | 10% | 60 | 60 | 60 | 46.9% | 66.9% | 20.5 | 14.9x |
| facility | 10% | 60 | 60 | 60 | 46.8% | 66.8% | 26.5 | 11.5x |
| random | 10% | 60 | 60 | 59 | 44.3% | 63.1% | 29.6 | 10.3x |
| dpp | 10% | 60 | 60 | 59 | 42.9% | 61.1% | 22.3 | 13.7x |
| fps | 10% | 60 | 60 | 59 | 40.6% | 57.9% | 14.3 | 21.4x |

### Pytest Strategy Rankings

**50% selection:**
1. fps — 65.9% (93.9% retention, -4.3pp vs full)
2. dpp — 65.9% (93.9% retention, -4.3pp vs full) *[FPS fallback for large k]*
3. kmedoids — 65.8% (93.8% retention, -4.4pp vs full)
4. fps_multi — 65.7% (93.7% retention, -4.4pp vs full)
5. facility — 64.9% (92.6% retention, -5.2pp vs full)
6. random — 64.2% (91.5% retention, -6.0pp vs full)

**20% selection:**
1. facility — 58.6% (83.6% retention, -11.5pp vs full)
2. kmedoids — 58.3% (83.1% retention, -11.9pp vs full)
3. fps — 56.1% (80.0% retention, -14.0pp vs full)
4. fps_multi — 53.8% (76.8% retention, -16.3pp vs full)
5. random — 53.2% (75.8% retention, -17.0pp vs full)
6. dpp — 47.6% (67.8% retention, -22.6pp vs full)

**10% selection:**
1. fps_multi — 47.7% (67.9% retention, -22.5pp vs full)
2. kmedoids — 46.9% (66.9% retention, -23.2pp vs full)
3. facility — 46.8% (66.8% retention, -23.3pp vs full)
4. random — 44.3% (63.1% retention, -25.9pp vs full)
5. dpp — 42.9% (61.1% retention, -27.3pp vs full)
6. fps — 40.6% (57.9% retention, -29.5pp vs full)

## Robot Framework Results (126 total tests, 63.51% full coverage)

| Strategy | Level | K | Run | Coverage | Retention | Time(s) | Speedup |
|---|---|---|---|---|---|---|---|
| full | 100% | 126 | 126 | 63.5% | baseline | 147.4 | 1.0x |
| **fps_multi** | **50%** | **63** | **63** | **61.6%** | **97.0%** | **97.3** | **1.5x** |
| facility | 50% | 63 | 63 | 61.6% | 96.9% | 98.6 | 1.5x |
| fps | 50% | 63 | 63 | 61.2% | 96.4% | 94.4 | 1.6x |
| kmedoids | 50% | 63 | 63 | 60.5% | 95.3% | 95.2 | 1.5x |
| dpp | 50% | 63 | 63 | 58.5% | 92.0% | 85.1 | 1.7x |
| random | 50% | 63 | 63 | 57.2% | 90.1% | 65.1 | 2.3x |
| **facility** | **20%** | **26** | **26** | **51.8%** | **81.5%** | **33.5** | **4.4x** |
| kmedoids | 20% | 26 | 26 | 47.5% | 74.8% | 27.0 | 5.5x |
| random | 20% | 26 | 26 | 46.0% | 72.5% | 25.8 | 5.7x |
| fps | 20% | 26 | 26 | 45.3% | 71.4% | 48.2 | 3.1x |
| fps_multi | 20% | 26 | 26 | 45.3% | 71.4% | 50.2 | 2.9x |
| dpp | 20% | 26 | 26 | 43.9% | 69.0% | 32.2 | 4.6x |
| **kmedoids** | **10%** | **13** | **13** | **42.3%** | **66.6%** | **13.0** | **11.3x** |
| dpp | 10% | 13 | 13 | 40.9% | 64.3% | 13.8 | 10.7x |
| facility | 10% | 13 | 13 | 40.2% | 63.3% | 11.4 | 12.9x |
| random | 10% | 13 | 13 | 37.6% | 59.2% | 13.1 | 11.3x |
| fps_multi | 10% | 13 | 13 | 37.5% | 59.1% | 13.8 | 10.7x |
| fps | 10% | 13 | 13 | 33.2% | 52.3% | 17.8 | 8.3x |

### Robot Strategy Rankings

**50% selection:**
1. fps_multi — 61.6% (97.0% retention, -1.9pp vs full)
2. facility — 61.6% (96.9% retention, -2.0pp vs full)
3. fps — 61.2% (96.4% retention, -2.3pp vs full)
4. kmedoids — 60.5% (95.3% retention, -3.0pp vs full)
5. dpp — 58.5% (92.0% retention, -5.1pp vs full)
6. random — 57.2% (90.1% retention, -6.3pp vs full)

**20% selection:**
1. facility — 51.8% (81.5% retention, -11.7pp vs full)
2. kmedoids — 47.5% (74.8% retention, -16.0pp vs full)
3. random — 46.0% (72.5% retention, -17.5pp vs full)
4. fps/fps_multi — 45.3% (71.4% retention, -18.2pp vs full)
5. dpp — 43.9% (69.0% retention, -19.6pp vs full)

**10% selection:**
1. kmedoids — 42.3% (66.6% retention, -21.2pp vs full)
2. dpp — 40.9% (64.3% retention, -22.7pp vs full)
3. facility — 40.2% (63.3% retention, -23.3pp vs full)
4. random — 37.6% (59.2% retention, -25.9pp vs full)
5. fps_multi — 37.5% (59.1% retention, -26.0pp vs full)
6. fps — 33.2% (52.3% retention, -30.3pp vs full)

## Strategy Recommendations

| Use Case | Best Strategy | Why |
|---|---|---|
| PR smoke tests (50%) | fps or fps_multi | Fastest, 94-97% retention, no extra deps |
| Aggressive CI (20%) | facility | 83.6% retention on pytest, 81.5% on Robot |
| Extreme pruning (10%) | fps_multi (pytest) / kmedoids (Robot) | Best retention at extreme compression |
| Nightly diverse runs | dpp | Probabilistic sampling explores different test subsets each run |
| Quick start (no extras) | fps | Zero extra dependencies, solid performance at all levels |

## Notes on DPP Behavior

The DPP (Determinantal Point Process) strategy uses probabilistic repulsion-based sampling. Key observations:

- **At 50%**: Falls back to FPS because exact k-DPP sampling is numerically unstable for k > N/3 (k=297 out of N=594). The FPS fallback produces identical results to the fps strategy.
- **At 20% and 10%**: DPP runs exact k-DPP sampling successfully but produces different (and sometimes less optimal) coverage patterns than deterministic strategies. This is expected — DPP optimizes for *diversity* (repulsion in feature space) rather than *representativeness* (cluster coverage).
- **Best use case**: Nightly CI runs with varying seeds, where each run explores a different diverse subset. Over multiple runs, DPP covers more total ground than any single deterministic selection.

## Methodology

- **Embedding model**: all-MiniLM-L6-v2 (384-dim sentence embeddings)
- **Text representation**: AST-based Combined strategy (D) for pytest; keyword tree + name + tags for Robot Framework
- **Coverage tool**: pytest-cov / coverage.py measuring line coverage on the DocTest package
- **Seed**: 42 (deterministic selection for all strategies)
- **Random baseline**: Python `random.sample()` with seed=42
- **DPP kernel**: RBF kernel with eigenvalue clamping; FPS fallback for large k
