# Rice genomic selection: frozen-prediction reproduction

This package reproduces empirical prediction-to-result calculations for the rice study *Phenotyping part of the target family improves within-family genomic prediction in rice nested association mapping families*. It includes the author's frozen predictions and exact scoring identities, rather than synthetic example data.

It covers the original NAM, MAGIC and IRRI comparisons; the completed NAM four-strategy comparison; the four-penalty NAM grid in all three scoring views; and both IRRI policies across the complete 20-partition grid. The entry scores 4,940 frozen arm-by-view cells and produces 9,240 metric rows. Original primary comparisons retain their original role.

## Run

Use Python 3.12 and NumPy 2.3.5. The reference execution used Python 3.12.14 on macOS. Install the declared dependency in your chosen environment, then run:

```text
python reproduce.py --wei-xlsx /path/to/DataS4.xlsx --irri-plotzip /path/to/RYT_plotdata_by_GHID_corrected_PUBLIC_ACCESS.zip --out results
```

Choose an output directory that does not exist. The program performs no downloads, model fitting or parameter tuning. It checks every bundled resource and the two local source-data versions before scoring. Missing or mismatched inputs fail explicitly.

Obtain the Wei et al. supplementary Data S4 workbook from https://doi.org/10.1126/science.adm8762 and the corrected IRRI plot ZIP from Dryad version 1, https://doi.org/10.5061/dryad.7369p. Source SHA-256 values and fields are specified in resources/inputs.json. The manuscript's unresolved field-year and original aggregation/adjustment interpretation is retained: the Wei adapter uses the released K/Z values directly and does not reconstruct field BLUEs. The IRRI adapter averages the replicate observations of the scored material-seasons; it relies on the fixed file hash and does not repeat the full historical qualification of every source record.

The package includes no raw genotypes or source phenotype workbook. Genotype and family-identity provenance is https://doi.org/10.6084/m9.figshare.19166475.v1 and https://doi.org/10.6084/m9.figshare.31881931.v1 (CC BY 4.0). IRRI source data are CC0 1.0. Cite Wei et al. (2024) and Spindel et al. (2015) alongside the source archives when reusing these materials.

## Outputs and interpretation

metrics.csv reports U, Pearson r, requested whole-set Spearman correlations and selected-ID order digests. The remaining CSVs report paired family means, strategy contrasts, arm means and observed family dispersion. All scientifically favorable, adverse and zero outcomes use the same rules.

NAM averages five paired draws within family and then weights fifteen families equally. Conditional intervals reuse the original 10,000 by 15 resampling matrix and linear-interpolated quantiles. Pointwise and selected-six Bonferroni sensitivities are distinct. The new fixed-penalty r summaries have no new interval or test. IRRI summaries describe overlapping partitions and repeated seasons of one cohort. U is a phenotypic selection differential, not realized offspring gain.

The reference check contains 8,940 metric records plus all 22 original MAGIC contrast summaries. It requires exact N and q, absolute U error at most 1e-8 in the stated yield unit, and correlation error at most 1e-10. An independent local execution passed 41,676 field checks, including the original MAGIC selections and all 22 original MAGIC contrasts; maximum differences were 1.82e-12 for U and 2.23e-16 for Pearson r. This validation checks frozen predictions through results. It does not validate newly fitted models.

## Scope

The numerical implementations in models/ are provided for inspection and reuse. The reproduction entry does not retrain or retune these models, reconstruct the historical development controllers, reproduce all earlier exploratory branches, reproduce the MAGIC matched-penalty/domain extensions, or reconstruct native Word/PDF layouts. Existing frozen predictions are the empirical input to this release. Raw-data-to-new-model reproduction is a separate scope.

study.json specifies inputs, versions, task counts and tolerances. resources/study.json.gz contains ordered candidate identities, calibration identities, task roles, parameters where recorded, and prediction keys. Source-only arms have no fitted calibration labels; their scoring domains still reflect the calibration partition. resources/reference.csv.gz contains the stored metric references. SHA256SUMS lists the checksums of the other v1.0.0 release files; README.md and CITATION.cff have since been updated on the main branch, so their checksums no longer match.

## License and citation

Author software and documentation: MIT, copyright 2026 Jixiao Li. Author-generated predictions and numerical results: CC BY 4.0. Source-derived genotype identities retain their CC BY 4.0 attribution; IRRI-derived identities retain their CC0 provenance. These licenses do not relicense the Wei supplementary phenotype workbook, which users must obtain and use under its source terms.

Use CITATION.cff for this software version. Version 1.0.0 is archived at https://doi.org/10.5281/zenodo.23438608 (all versions: https://doi.org/10.5281/zenodo.23438607).
