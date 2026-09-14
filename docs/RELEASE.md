# Release Status

This page records the public artifact boundary for INTACT. The status table in
the root README is authoritative.

## Available

- paper, project website, method notes, and audited result tables;
- task-specific and four-task shared-encoder training code;
- one standard evaluation entrypoint with Direct, CEM, and Guarded-A modes;
- Official LeWM and CLEAR-LeWM v0.8 scoring adapters;
- exact training/evaluation configurations and dependency locks;
- paper checkpoint manifests, SHA-256 records, and the bundled compatibility
  runtime;
- current E1 single-task releases with and without causal previous-action
  context, organized under `INTACT/` and `INTACT-no-previous-action/` in the
  public Hugging Face repository;
- all 72 paper checkpoint shards under the immutable public Hugging Face
  revision [`paper-e5-goal-v1`](https://huggingface.co/INTACT-JEPA/INTACT/tree/paper-e5-goal-v1),
  also organized under the current [`INTACT-unified/`](https://huggingface.co/INTACT-JEPA/INTACT/tree/main/INTACT-unified)
  directory.

## Distributed Separately

Model weights are too large for the source repository and are distributed
through a dedicated model host. A checkpoint is considered public only when
its download location is available without repository credentials and its hash
matches the checked-in manifest.

## Release Checklist

- [x] paper title, citation, method terminology, and headline values aligned;
- [x] training and evaluation entrypoints included;
- [x] dependency lock and installation verification included;
- [x] source tree checked for credentials and machine-specific paths;
- [x] large generated outputs excluded from Git history;
- [x] public model hosting linked and anonymously verified end to end;
- [x] one anonymous checkpoint download and extracted-shard verification completed;
- [ ] one clean-clone reference evaluation completed.
