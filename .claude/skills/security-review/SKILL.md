---
name: security-review
description: Security review checklist based on OWASP Top 10. Apply during code review and pre-deployment checks.
allowed-tools: Read, Grep, Glob
---

# Security Review Checklist

## OWASP Top 10 Checks

### A01: Broken Access Control
- [ ] Endpoints requiring authentication have authentication checks
- [ ] Users cannot access other users' resources
- [ ] Keycloak token verification is implemented correctly

### A02: Cryptographic Failures
- [ ] SSL/TLS is enabled (`verify=True` or specified CA certificate)
- [ ] Passwords are not stored in plain text
- [ ] Secrets are not written to logs

### A03: Injection
- [ ] SQL: parameterized queries are used
- [ ] XSS: user input is escaped
- [ ] Command injection: subprocess is not called with `shell=True`

### A04: Insecure Design
- [ ] Input validation is appropriate (Pydantic Field)
- [ ] Rate limiting is configured
- [ ] Error messages do not leak internal information

### A05: Security Misconfiguration
- [ ] CORS is restricted to the minimum necessary origins
- [ ] Debug mode is disabled in production
- [ ] No default credentials are used

## Search Patterns

```bash
# Hardcoded secrets
grep -rn "api_key\|token\|secret\|password" --include="*.py" --include="*.ts"

# Insecure external communication
grep -rn "verify=False" --include="*.py"

# Use of eval/exec
grep -rn "eval(\|exec(" --include="*.py"
```
