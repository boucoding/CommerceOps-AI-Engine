# Recorded LLM traffic

`make record` runs the agent live on the sample emails and saves every model exchange to
`llm_recordings.json` here. `LLM_MODE=replay` (and `make eval`) then replays it byte for byte:
no API key, no network, deterministic results. Any change to prompts, tools, catalog data or
emails changes the request fingerprints, so stale recordings fail loudly instead of answering
a different question. Commit the file to let CI and reviewers re-run the evaluation.
