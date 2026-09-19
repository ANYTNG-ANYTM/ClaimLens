# Final release verification

**Date:** 2026-09-19
**Status:** Ready for GitHub file review with explicit blockers: Docker runtime, CSV redistribution rights, and an independent final model holdout. No commit or push was made.

## Environment and source state

- Windows, Python 3.13.1. A new ignored `.verify-venv` began with only `pip==24.3.1`; declared dependencies were installed from `requirements.txt`. A second new `.venv` was created inside an ignored clean release copy. Both passed `pip check`.
- Tested key versions: NumPy 2.4.3, pandas 3.0.5, scikit-learn 1.9.0, FastAPI 0.141.1, Streamlit 1.63.0, nbclient 0.10.4, ipykernel 7.1.0. The complete direct dependency set is pinned in `requirements.txt`.
- During the original release-copy verification, `git rev-parse --show-toplevel` and `git status --short` returned “not a git repository.” Git was initialized afterward. The current follow-up review found `main` tracking `origin/main` at `b37d19c` (`Initial Commit`), with the intended 39 public files tracked and a clean baseline before the README updates. A read-only `git ls-remote origin HEAD` reached GitHub and returned the same commit. Ownership checks required a command-scoped `safe.directory` override in this sandbox; no global Git setting, remote, commit, or push was changed here. Push credentials were not tested.
- Original CSV SHA-256 before and after all work: `0a7d71bf1b07b6ace30e531a6ae460e796ec2e4dc86f976c6a7f7d08e1b68bb1`. The temporary release copy received a local copy with the same checksum solely for verification.

## Executed commands and outcomes

Commands below ran from the `FraudDetection` root unless marked **copy**. `P` denotes `.\.verify-venv\Scripts\python.exe` and `C` denotes `.\.venv\Scripts\python.exe` inside `.release-verify`. These are local verification directories excluded from the intended release.

| Command or check | Status | Observed result |
| --- | --- | --- |
| `..\.venv\Scripts\python.exe -m venv .verify-venv` | PASS | Fresh Python 3.13.1 environment initially had only pip. |
| `P -m pip install -r requirements.txt` | PASS | All declared direct dependencies installed after network access was allowed. |
| `P -m pip check` | PASS | No broken requirements. |
| Python `ast.parse` over `src/*.py`, `scripts/*.py`, `tests/*.py` | PASS | 16 Python files parsed. |
| `P -m src.data_pipeline` | PASS | 1,000 rows, 40 columns, unique policy identifiers; fixed 600/200/200 partitions; CSV checksum matched. |
| `P -m src.eda` | PASS | Development EDA and five aggregate figures regenerated. |
| `P -m src.modelling` | PASS | All six configured candidates ran through 3-fold training CV; same model, threshold, and metrics reproduced. |
| `P -m scripts.make_example` | PASS | Synthetic request regenerated from training medians and modes; it matches no individual CSV row. |
| `P -m src.explain`; `P -m scripts.export_public_reports` | PASS | Private coefficients and aggregate-only public summaries regenerated. |
| `P -m scripts.run_notebooks` | PASS | All five current notebooks ran in fresh kernels. Model bundle SHA-256 was unchanged before and after notebook execution. |
| `P -m pytest -q` | PASS | Four focused tests passed: split/checksum, bundle metadata, direct/API/UI agreement, caching, missing/unseen/invalid values, and threshold behavior. |
| `P -m scripts.verify_runtime` | PASS | Started its own Uvicorn and Streamlit services; health/readiness, valid/invalid/incomplete/missing/unseen requests, direct score equality, and a real Streamlit form submission passed. It stopped only its own processes. |
| `..\.venv\Scripts\python.exe -m pytest -q`; `..\.venv\Scripts\python.exe -m scripts.verify_runtime` after temporary cleanup | PASS | Four tests and the live API/Streamlit interaction still passed after removing both temporary verification environments and the release copy. |
| `docker compose config --quiet` | PASS | Compose parsed after changing API model supply to a read-only volume. |
| `docker info --format '{{.ServerVersion}}'` | BLOCKED | Docker daemon pipe `npipe:////./pipe/docker_engine` was absent. |
| `docker compose build`, container startup, container API/UI tests | BLOCKED | Build could not connect to the daemon; no image or container result is claimed. |

The runtime verifier initially returned a **nonzero exit** when Streamlit could not resolve a relative import. The dashboard now adds its project root to the import path before importing `src.ui_client` and `src.util`; both the main and clean-copy runtime checks passed afterward. The repository has `src/util.py` (singular), not `src/utils.py`.

## Clean release-copy reproduction

An ignored `.release-verify` directory was assembled from **39 intended public files**: source, config, scripts, tests, five current notebooks, Docker files, README, verification documentation, five aggregate figures, aggregate JSON reports, and the synthetic example. It initially contained **no CSV, model bundle, processed data, private metrics, archive, legacy notebook, or old presentation image**. This was a local clean release copy, not a Git clone.

1. From the copy, `P -m src.data_pipeline` exited **1** before CSV placement with a clear “Missing private dataset” message. `GET /health` returned 200; `GET /ready` returned **503** and instructed `python -m src.modelling` while the model was absent.
2. The original CSV was copied locally into the ignored copy at `data/dataset/insurance_claims.csv`; its checksum matched the original.
3. From the copy, `python -m venv .venv`, `C -m pip install -r requirements.txt`, and `C -m pip check` all passed. The copy used no package from the parent project environment.
4. From the copy, `C -m src.data_pipeline`, `C -m src.eda`, `C -m src.modelling`, `C -m scripts.make_example`, `C -m src.explain`, and `C -m scripts.export_public_reports` all passed on the full 1,000-row CSV. Model and threshold matched the main fresh run; the two generated bundle files had identical SHA-256 values.
5. From the copy, `C -m scripts.run_notebooks` passed all five notebooks without changing the bundle hash. `C -m pytest -q` passed four tests. `C -m scripts.verify_runtime` passed live API requests and a Streamlit form submission. The updated runtime verifier was copied into the clean copy before this final service check.

This establishes a reproducible **Windows local release-copy workflow** after a user legally supplies the CSV. Linux/macOS commands in the README are equivalents and were **not executed**.

The temporary release copy, its dataset and virtual environment, and the separate fresh verification environment were removed after the checks. The active project dataset and model bundle remain in place.

## Reproduced model result and limitation

The selected class-weighted logistic model had training CV average precision **0.6077477444** (fold SD 0.069069), with validation threshold **0.34** chosen by fraud F2. Validation fraud precision/recall were **0.608108/0.918367**. Exploratory test precision/recall were **0.527778/0.775510**, with average precision **0.539681**. The test confusion matrix, actual and predicted `[N,Y]`, was `[[117,34],[11,38]]`.

**The test estimate is exploratory.** An earlier feature set was evaluated on this same partition before `incident_severity` was added. The current release pass did not alter the split, tune on the test result, or claim that repeated runs create an untouched holdout. A new independently collected, preferably later sample is required for an unbiased final estimate. Only 49 positive examples are in the test partition. Severity and hobby associations may reflect dataset artifacts; scores are uncalibrated and must not be treated as proof of fraud.

## Release fixes and hygiene

- Removed the Dockerfile's `COPY models/fraud_pipeline.joblib`. Compose now mounts locally generated `./models` read-only, and API readiness gates Streamlit startup. This makes a clean image build independent of ignored model files. Docker behavior still needs a running daemon to verify.
- Fixed Streamlit imports in the actual script execution context. The runtime verifier now detects occupied ports and checks invalid and supported missing requests through the live API.
- Added explicit missing-CSV guidance and a test that bundle metadata, threshold, features, and results agree.
- Added aggregate-only public results and coefficients. The private audit contains policy identifiers and remains ignored. The public example is synthetic and does not copy a raw row.
- Updated `.gitignore` and `.dockerignore` for local virtual environments, release copies, caches, logs, secrets, raw and processed data, models, private metrics, archive, legacy notebooks, and historical images. Current aggregate figures and public JSON reports remain in the intended release set.
- Reviewed active source, notebooks, scripts, and Docker files for legacy imports, absolute machine paths, and hardcoded credentials; no active dependency on ignored historical material was found. The largest intended public file before local setup was about 40 KB. README relative links resolved in the clean copy.

## Outstanding actions

1. Confirm the CSV's provenance and redistribution rights. Keep it out of a public repository until then; a new user must supply a compatible file.
2. Obtain a truly independent labelled holdout before making final performance or operational claims.
3. With a working Docker daemon, run `docker compose build`, `docker compose up -d`, a container API prediction, and a UI submission; then `docker compose down`. These checks remain **BLOCKED**, not passed.
4. Review and commit the current documentation changes before pushing `main` to the already configured `origin`. Confirm the destination and credentials with your own Git session. No commit or push was performed during this follow-up review.
