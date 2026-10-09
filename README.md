# ML-Mini-Project: Pancreatic Cancer Prognosis Using Clinical and Radiomic Data

**Team:** PES2UG24CS324 | PES2UG24CS325 

To be added:
Project Status
Data
Repository Layout

## Installation

```bash
py -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Linux/macOS: To Be Added

```bash
pip install -r requirements.txt
```

For image feature extraction, also install:

```bash
pip install -r requirements-radiomics.txt
```

## Reproduce the included clinical analysis

Run from the repository root. Script 1 onwards work with the processed dataset. 0 needs the original. 

```bash
py src/1_datamain.py
py src/2_kaplanmeier.py
py src/3_naivebayesthreshold.py
py src/4_smoteclassification.py
py src/5_coxanalysis.py
py src/6_logrankanalysis.py
py src/7_coxriskkm.py
py src/8_finalresults.py

```

To Be Added:

Key outputs
Radiomics
Survival Endpoints
Limitations