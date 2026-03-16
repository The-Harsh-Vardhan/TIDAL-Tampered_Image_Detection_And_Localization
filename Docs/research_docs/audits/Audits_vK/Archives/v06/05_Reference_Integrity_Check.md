# Reference Integrity Check

This report checks whether `Docs6/13_References.md` and the research-facing claims in `Docs6/11_Research_Alignment.md` are grounded in the actual repository contents.

## What Checks Out

- The cited dataset links in `13_References.md` match files and links already present in the repo.
- The Tier A, Tier B, and Tier C paper buckets correspond to PDFs that actually exist under `Research Papers/`.
- The cited reference notebooks exist locally under `More Resources/`:
  - `image-detection-with-mask.ipynb`
  - `document-forensics-using-ela-and-rpa.ipynb`
- The exclusion of off-domain papers such as tempered-glass defect detection and watermarking papers is reasonable.

## Problems Found

| Area | Issue | Why it matters | Recommended fix |
|---|---|---|---|
| Paper ID mapping | `11_Research_Alignment.md` uses `P1`, `P2`, `P4`, `P6`, `P7`, `P10`, `P13`, `P14`, `P15`, `P16`, `P17`, `P18`, `P19`, `P20`, `P21`, but `13_References.md` does not define a matching numbered index | Research claims are harder to verify and easier to overstate | Add a stable ID map in `13_References.md` or remove the `P#` shorthand |
| Unclear references | Mentions such as `P4`, `P20`, and `P21` are not traceable from the current numbered list in `13_References.md` | Readers cannot tell which exact paper supports which claim | Make every research claim point to an explicit file or stable ID |
| Robustness overreach | `11_Research_Alignment.md` claims an 8-condition robustness suite including brightness, contrast, saturation, and combined degradation, but the actual v6 notebooks do not implement that suite | Research-backed implementation claims become inflated | Restrict the research-alignment narrative to implemented robustness conditions |
| Runtime-specific tools | `13_References.md` lists Kaggle Secrets API but does not cover Colab-specific helper usage now relevant to the v6 Colab notebook | Reproduction guidance is incomplete for the current repo state | Add a small runtime-tools section covering both Kaggle and Colab variants |

## Verdict

`13_References.md` is a useful inventory, but it is not yet a reliable citation backbone for `11_Research_Alignment.md`. The file existence is good; the citation traceability is not.
