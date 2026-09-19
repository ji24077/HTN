# Independently authored extraction benchmark

`data/experiments/independent-review/heldout.jsonl` contains 50 fictional appointment records written individually by a separate reviewing agent. This is an evaluation-only file: do not add it to training data or use its outputs to select a model and then report it as untouched evaluation.

The author had inspected the previous synthetic-data code and evaluation reports, including their chronology and name-copying failures, before writing these examples. This is **not blinded human evaluation**, independently collected real-world data, or evidence about an unknown deployment population. The author did not inspect the concurrent new training-data work or share benchmark contents with its author. Sentences were composed individually, without importing generator templates, prior rows, or entity pools.

There are ten examples in each of five primary categories:

- `chronology`: appointment years among opening, publication, recruitment, and qualification dates.
- `subject_selection`: an explicit appointment subject alongside reporters, colleagues, officials, or other named people.
- `diacritics`: names containing multiple non-ASCII characters, with varied scripts within Latin text.
- `punctuation`: quotes, apostrophes, brackets, braces, parentheses, ampersands, and punctuation inside labels.
- `record_formats`: prose, employment confirmations, interview questions, and multiline administrative notes.

Every row explicitly states one person's age at appointment, appointment year, employer, and role. Labels preserve verbatim name, organisation, and title text. The age refers to joining or appointment, not current age. Some facts span sentences and require pronoun or document-subject resolution. All five facts are present; missing-fact or intrinsically ambiguous prompts are deliberately excluded because the existing schema has no abstention or null policy. None of these fictional records should be interpreted as biographical claims about real people.

The author reviewed all labels against their sentences and checked the existing strict `Record` schema, exact presence of string labels and numeric values, unique IDs/sentences/records, category counts, and a SHA-256 of the exact UTF-8 file. The manifest records that freeze. These checks establish formatting and direct label support, not human agreement on every sentence. Keep the file unchanged after inference. If an error is discovered later, record it separately and version a replacement benchmark rather than silently editing measured examples.

Run both baseline and candidate with the same generation batch, dtype, token budget, prompt, and scoring version. Request all 50 examples. Use a sufficiently long sequence cap for the secondary teacher-forced loss; loss truncation must not drop accuracy cases. Record the manifest hash with each report and retain every prediction. Inspect per-category errors alongside exact-match and field scores. Ten examples per category produce coarse estimates; the benchmark supports a targeted diagnostic comparison, not precise general accuracy or production certification.

After observing results, further development should use a new separate validation set and preserve this file as an already-observed test. A strong next step is an independently human-labeled benchmark from representative permitted documents, with explicit annotation rules and ambiguity handling.
