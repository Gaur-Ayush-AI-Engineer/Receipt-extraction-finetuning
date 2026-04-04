## Fine-tuning Results

**Baseline**: Baseline (No Fine-tuning)  
**Fine-tuned**: Fine-tuned Model

### Per-field Exact Match

| Field | Before | After | Delta | % Improvement |
|:------|-------:|------:|------:|--------------:|
| Company | 75.1% | 90.4% | +15.2% | +20.3% |
| Date | 53.3% | 98.0% | +44.7% | +83.8% |
| Address (exact) | 47.7% | 77.2% | +29.4% | +61.7% |
| Address (fuzzy) | 94.6% | 97.7% | +3.1% | +3.2% |
| Total | 89.3% | 98.0% | +8.6% | +9.7% |
| **All fields correct** | **32.5%** | **71.1%** | **+38.6%** | — |
| JSON parse failure rate | 0.5% | 0.0% | -0.5% | — |

### Remaining Failure Analysis (fine-tuned model)

| Failure Type | Count | % of Failures |
|:-------------|------:|--------------:|
| Partial Match | 57 | 100.0% |
| Parse Failure | 0 | 0.0% |
| Wrong Format | 0 | 0.0% |
| Wrong Value | 0 | 0.0% |
