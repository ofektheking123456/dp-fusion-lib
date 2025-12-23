# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-01

### Added

- Initial release of DP-Fusion-Lib
- `DPFusion` class for differentially private text generation
  - Message-based context building with `add_message()`
  - Direct context generation with `generate()`
  - Token-level generation with `generate_from_tokens()`
- `Tagger` class for automatic private phrase extraction
  - Integration with Document Privacy API
  - Support for multiple document types (HEALTH, FINANCE, LEGAL)
- Privacy accounting functions
  - `compute_epsilon_single_group()` for single-group privacy guarantees
  - `compute_dp_epsilon()` for multi-group scenarios
- Utility functions for advanced usage
  - `compute_renyi_divergence_clipped_symmetric()` for divergence computation
  - `find_lambda()` for binary search of mixing parameter
  - `replace_sequences_with_placeholder_fast()` for token-level redaction
- Support for HuggingFace transformers models
- Incremental decoding with KV-cache optimization
- Comprehensive documentation and examples

### Dependencies

- PyTorch >= 2.0.0
- Transformers >= 4.25.0
- Requests >= 2.25.0
