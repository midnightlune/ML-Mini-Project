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

For image feature extraction, also install:

```bash
pip install -r requirements-radiomics.txt
```

## Reproduce the included clinical analysis

Run from the repository root. Script 1 onwards work with the processed dataset. 0 needs the original. 

```bash
py src/01_datamain.py
py src/02_kaplanmeier.py
py src/03_coxmodel.py
py src/04_preprocessclinical.py
py src/05_naivebayesthreshold.py
py src/06_smoteclassification.py
py src/07_coxanalysis.py
py src/08_logrankanalysis.py
py src/09_coxriskkm.py
py src/10_finalresults.py

```

To Be Added:

Key outputs
Radiomics
Survival Endpoints
Limitations