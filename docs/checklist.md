# Final Submission Checklist

This checklist is maintained outside the research paper. Complete it against the
frozen submission revision, not against an intermediate working tree. Tasks are
ordered so that content is frozen before the final PDF, archive and hashes are
created.

## 1. Required authorship and approval

- [x] Retain the final three-author contribution table and collective
  responsibility statement. Do not reintroduce a fourth-member or
  contribution-related `TBD`.
- [ ] Obtain group approval of the final AI-use declaration.

## 2. Evidence and documentation consistency

- [ ] Reconcile the final WebText seed count across the paper, `README.md`,
  proposal, dataset card and reproducibility note. The paper and current results
  describe three WebText seeds and 420 test conditions, while several supporting
  documents still describe one WebText seed and 390 conditions.
- [ ] Review the ROCStories and WebText licence entries. If no clearer upstream
  terms can be verified, replace each `TBD` with an explicit statement that the
  terms remain unclarified and that raw text will not be redistributed.
- [ ] Complete the consolidated execution-environment record with every
  recoverable Python and PyTorch version, accelerator/backend and CPU/RAM detail.
  Do not invent missing values or map environments to individual corpora unless
  retained evidence supports that mapping.
- [ ] Resolve Reference 17: cite it in the manuscript where relevant or remove it
  from the reference list. References 1-16 are currently cited.
- [x] Replace brittle source-line references in Appendix F with stable class,
  method and source-file references.
- [ ] Run a final placeholder scan and resolve every remaining `TBD` in the paper.

## 3. Manuscript QA

- [ ] Confirm that the final main-text count is within 4,000-5,500 words,
  excluding references and appendices, using the same counting convention for
  the submitted version. The current draft is near the upper limit, so repeat
  this check after all author and licensing edits.
- [ ] Recheck all numerical claims, table values, captions and corpus/model names
  against the retained metrics and plot receipts.
- [ ] Verify that every citation is present, every reference is cited and every
  local link and figure reference resolves.
- [ ] Confirm the paper still contains the assignment's required sections,
  complete-test metrics, rare/unseen analysis, at least ten qualitative cases,
  smoothing ablation, runtime, OOV and reproducibility disclosures.
- [ ] Confirm that no raw ROCStories, Reuters or WebText text is included in the
  submission archive contrary to the recorded terms or non-redistribution
  decision.

## 4. Code and supporting-material QA

- [ ] Reconcile the final repository changes and commit only the intended source,
  paper, supporting documents, figures and compact evidence. Exclude temporary
  render files and unrelated data/results.
- [ ] From a clean environment, verify the README's installation and inexpensive
  read-only/check commands. Do not rerun expensive training solely for final QA.
- [ ] Run the retained completeness checks for every corpus and preserve their
  output or receipts.
- [ ] Confirm that the final archive contains `README.md`, requirements or an
  environment file, source code/notebooks, data-access instructions, compact
  results, figures, dataset card, proposal, and the contribution/AI-use
  declaration.

## 5. Final PDF and submission archive

- [ ] Render the frozen manuscript as `research_paper.pdf`.
- [ ] Inspect every PDF page for clipped text, broken equations, missing Mermaid
  diagrams, unreadable plots, table overflow, incorrect page breaks and heading
  or figure-numbering problems.
- [ ] Confirm that the final PDF uses the frozen manuscript revision and that all
  embedded figures are the intended publication versions.
- [ ] Create the single submission ZIP and verify its contents, paths, size and
  ability to extract cleanly.
- [ ] Record the frozen Git commit SHA.
- [ ] Generate `SHA256SUMS.txt` only after the PDF and archive are final. At
  minimum, hash `research_paper.pdf` and the submitted ZIP; also hash any file
  uploaded separately.
- [ ] Verify the generated hashes from a second command/read and retain the
  manifest with the submission records.
- [ ] Upload exactly one final group submission and retain the portal receipt.

## Verified in the current audit

- [x] All ten local images referenced by `Paper_Final_Draft.md` exist.
- [x] Dataset-access instructions, dataset card, proposal, contribution/AI-use
  document, README and requirements file exist in the repository.
- [x] The paper documents the baseline, comparison models, controlled smoothing
  ablation, complete-test evaluation, rare/unseen analysis and qualitative error
  cases.
- [x] The revised WebText scoring-event archive and contribution values are
  documented as independently verified without neural retraining.
- [x] No final-named `research_paper.pdf`, submission ZIP or `SHA256SUMS.txt`
  exists yet; their absence is intentional until the preceding tasks are closed.
