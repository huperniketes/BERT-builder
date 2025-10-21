# C-Bert Constitution

## Core Principles

### I. Research-Grade
   * The objective of this project is to recreate the system described in the paper *Exploring Software Naturalness through Neural Language Models* (2006.12641v2). The code must be research-grade.
   * Prioritize algorithmic correctness and reproducibility over all else.
   * Ensure the code is well-documented, especially the parts related to the core algorithm and data transformations.
   * The implementation should be modular and easy to modify for future experiments.
   * Avoid hardcoded values. All hyperparameters should be configurable through command-line arguments or a configuration file.
   * Include assertions and checks to validate intermediate results and data shapes.
   * The code should be self-contained, with all dependencies clearly listed in `requirements.txt`.

### II. Library-First
Every feature starts as a standalone library; Libraries must be self-contained, independently testable, documented; Clear purpose required - no organizational-only libraries

### III. CLI Interface
Every library exposes functionality via CLI; Text in/out protocol: stdin/args → stdout, errors → stderr; Support JSON + human-readable formats

### IV. Test-First (NON-NEGOTIABLE)
TDD mandatory: Tests written → User approved → Tests fail → Then implement; Red-Green-Refactor cycle strictly enforced

### V. Code Quality
Code must be clear, maintainable, and efficient. All contributions will be measured against these standards through code reviews and static analysis. Rationale: High-quality code reduces bugs, lowers maintenance costs, and accelerates future development.

### VI. Testing Standards
Comprehensive and automated testing is mandatory. This includes unit, integration, and end-to-end tests to ensure functionality and reliability. Rationale: Rigorous testing is the foundation of a stable and trustworthy system.

Focus areas requiring integration tests: New library contract tests, Contract changes, Inter-service communication, Shared schemas

### VII. User Experience Consistency
A cohesive and predictable user interface and interaction model must be maintained across the entire project. Rationale: Consistency improves usability and reduces the learning curve for new users.

### VIII. Performance Requirements
The system must meet defined and measurable performance benchmarks. Performance testing is required for any change that may impact speed or resource consumption. Rationale: A high-performance system is critical for user satisfaction and scalability.

### IX. Observability
### X. Versioning & Breaking Changes

### XI. Simplicity
Text I/O ensures debuggability; Structured logging required; Or: MAJOR.MINOR.BUILD format; Or: Start simple, YAGNI principles

## Additional Constraints, Security Requirements, Performance Standards, etc.
CMake is used to create the build system for contributors and users to install and use locally.
compliance standards, deployment policies, etc.

## Development Workflow, Review Process, Quality Gates, etc.
Code review requirements, testing gates, deployment approval process, etc.

## Governance
This Constitution supersedes all other practices. Amendments require documentation, approval, and a migration plan.

### Governance Rules
All PRs/reviews must verify compliance; Complexity must be justified; Use [GUIDANCE_FILE](TBD) for runtime development guidance

**Version**: 1.0 | **Ratified**: 2025-10-12
