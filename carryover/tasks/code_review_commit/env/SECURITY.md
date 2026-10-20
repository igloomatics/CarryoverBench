# Security Policy

## Reporting Vulnerabilities

Report security issues privately to security@company.com.
Do not open public issues for security vulnerabilities.

## Secrets Management

- Never commit secrets, API keys, or passwords to the repository.
- All environment variables must live in `.env` (gitignored).
- The CI pipeline runs secret-scanning on every PR.

## Code Review Security Checklist

- [ ] No hardcoded credentials in the diff
- [ ] Input validation present for user-supplied data
- [ ] No new external dependencies with known CVEs
- [ ] Authentication and authorisation logic reviewed by a second engineer
