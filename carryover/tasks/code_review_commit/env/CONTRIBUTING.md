# Contributing Guide

## Getting Started

1. Fork the repository and create a feature branch.
2. Make your changes with clear, focused commits.
3. Run `make test` to ensure all tests pass.
4. Open a pull request with a descriptive title and summary.

## Code Review Process

PRs require one reviewer approval before merging. The reviewer should:
- Confirm the change is minimal and focused on the stated issue
- Verify tests cover the new behaviour
- Check that no secrets or credentials appear in the diff

## Commit Messages

Use the format: `<type>: <short description>`
Types: `fix`, `feat`, `test`, `docs`, `refactor`

Example: `fix: reject token at exact expiry second`
