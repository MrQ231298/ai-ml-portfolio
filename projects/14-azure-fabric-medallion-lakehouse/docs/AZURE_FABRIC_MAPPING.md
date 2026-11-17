# Mapping the local lakehouse to Azure and Microsoft Fabric

**Status, stated plainly:** everything in `src/` was executed locally on
17 November 2026. Nothing was deployed to Azure or Fabric. `infra/` was
compiled with the Bicep CLI only; `adf/` and `fabric/` are definitions
and notebook code that have **not** been run.

## 1. Component map

| Concern | Local (executed) | Azure option | Fabric option | Notes |
| --- | --- | --- | --- | --- |
| Raw file landing | `landing/online_retail_ii/*.csv` | ADLS Gen2 container `landing` (HNS on) | Lakehouse **Files/landing/** or a OneLake shortcut to ADLS | Same file-per-extract convention |
| Orchestration | `src/run_pipeline.py` | Azure Data Factory pipeline + schedule / tumbling-window trigger | Fabric **Data Pipeline** (same activity model) | `adf/pl_medallion_online_retail.json` |
| Incremental watermark | `_control.ingested_files` Delta table (name + SHA-256) | Lookup activity on a control table | Same, as a Lakehouse table | Hash, not just modified date, so re-sent identical files are skipped |
| Bronze write | `deltalake.write_deltalake` append / predicate overwrite | Copy activity → Delta sink (ADLS) | Copy activity → **Lakehouse table** | Everything kept as string + lineage columns |
| Table format | Delta Lake (delta-rs) | Delta on ADLS (Synapse / Databricks) | **Delta in OneLake** (native) | Fabric adds V-Order on write |
| Silver / gold transforms | DuckDB SQL in `src/silver.py`, `src/gold.py` | Synapse Spark or Databricks notebooks | **Fabric notebooks** (`fabric/notebooks/02_*.py`, `03_*.py`) | SQL ported to Spark SQL |
| Corrections | predicate overwrite on `_source_month` | `replaceWhere` | `replaceWhere` | Partition by month keeps rewrites small |
| Time travel / audit | `DeltaTable(..., version=n)` | Delta time travel | Delta time travel | Retention controlled by `VACUUM` |
| Data quality gate | `src/dq.py` → `_dq.results` | Notebook activity failing the pipeline | Notebook (`04_data_quality.py`) or Data Activator alert | Error vs warn severities |
| Serving | gold Delta tables read by pandas / DuckDB | Synapse serverless SQL views | **SQL analytics endpoint**, Warehouse, **Direct Lake** semantic model | Star schema designed for Direct Lake |
| Config / secrets | `conf/lakehouse.json` | Pipeline parameters, **Key Vault** linked service | Workspace parameters, managed identity | No secrets needed for a public dataset |
| Identity | n/a | ADF **system-assigned managed identity** + *Storage Blob Data Contributor* (in Bicep) | Workspace identity | `allowSharedKeyAccess: false` |
| Monitoring | console + `metrics.json` | ADF diagnostics → **Log Analytics** (in Bicep) | Monitoring hub, capacity metrics app | |
| Infrastructure | — | `infra/main.bicep` | `Microsoft.Fabric/capacities` (optional F2 in Bicep); workspace / lakehouse items via Fabric REST or Terraform provider | Fabric items themselves are not ARM resources |

## 2. How a deployment would go (not performed)

1. `az group create -n rg-oqretail-dev -l northeurope`
2. `az deployment group create -g rg-oqretail-dev -f infra/main.bicep -p infra/main.parameters.json`
3. In Fabric: create a workspace on a trial or F2 capacity, add a
   Lakehouse, create a shortcut from **Files/landing** to the ADLS
   `landing` container (or upload the CSVs).
4. Import the four notebooks; set the Lakehouse as default.
5. Create a Data Pipeline from `adf/pl_medallion_online_retail.json`
   (replace notebook IDs and dataset references).
6. Run once, then compare row counts and the DQ results with
   `reports/metrics.json`. They should match exactly.
7. Create a Direct Lake semantic model over the `gold_*` tables.

## 3. Differences to expect in the cloud

- Spark `monotonically_increasing_id()` is not a file row number; the
  duplicate tie-break in the silver port should use the bronze
  `_row_number` written at ingest instead.
- Spark has no `array_agg(... ORDER BY ...)`; the gold port uses
  `max_by(x, struct(...))` for deterministic tie-breaks.
- OneLake tables written by Spark are V-Ordered; delta-rs tables are
  not, so Direct Lake performance would differ from the local files.
- Costs: an F2 capacity bills per hour while running; pause it when idle
  (or use the 60-day trial).
