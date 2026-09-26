# Security Policy

## Supported versions

| version | supported |
| --- | --- |
| 0.3.x | yes |
| < 0.3.0 | no |

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Use GitHub's private vulnerability reporting:
[Security > Advisories > Report a vulnerability](https://github.com/divyanshudhruv/oev/security/advisories/new).

Include what you can of the following:

- affected version or commit
- the component (inference, server, export, training, packaging)
- a minimal reproduction or proof of concept
- expected vs actual behavior

## Scope notes

OEV ships model weights, an inference API and an optional HTTP server. Points
worth reporting include code execution through crafted checkpoints, unsafe
deserialization, injection through question or state payloads in the server,
and path handling in export tooling. Loading a checkpoint executes stored
configuration, so only load checkpoints from sources you trust; the published
SHA-256 hashes in BENCHMARKS.md cover the artifacts on the Hugging Face repo.
