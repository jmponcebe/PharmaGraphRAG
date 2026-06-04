# Data Sources — Reference

## FDA FAERS (FDA Adverse Event Reporting System)

- **URL**: <https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html>
- **Format**: Quarterly ZIP files containing CSV-like text files ($-delimited)
- **Quarters used**: 2024Q3, 2024Q4
- **Key tables**: DEMO (demographics), DRUG (drug entries), REAC (reactions), OUTC (outcomes)
- **Scale**: ~816K reports, ~3.9M drug entries, ~2.8M reactions per quarter
- **Download script**: `src/pharmagraphrag/data/download_faers.py`
- **Cleaning script**: `src/pharmagraphrag/data/clean_faers.py`
- **Processed output**: Parquet files in `data/processed/faers/{quarter}/`

## DailyMed (Drug Labels)

- **URL**: <https://dailymed.nlm.nih.gov/dailymed/>
- **Access**: REST API via openFDA (not DailyMed directly)
- **Format**: JSON drug label documents
- **Drugs fetched**: 88 drugs (focused on high-interaction drugs)
- **Key sections extracted**: drug_interactions, adverse_reactions, warnings_and_cautions, contraindications, boxed_warning, indications_and_usage, dosage_and_administration, clinical_pharmacology, mechanism_of_action, pharmacodynamics, overdosage, warnings
- **Ingestion script**: `src/pharmagraphrag/data/ingest_dailymed.py`
- **Output**: JSON files in `data/raw/dailymed/`
