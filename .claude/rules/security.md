# Security Rules

## Absolutely Forbidden

- Do not hardcode API keys, tokens, or passwords in source code
- Do not commit `.env` files
- Do not use `eval()` or `exec()` (templates inside autogen FunctionTool are an exception)
- Do not embed user input into SQL, HTML, or shell commands without sanitization

## Required Practices

- Retrieve secrets via environment variables or configuration files
- Enable SSL verification for outbound requests (`verify=True` or specify a CA certificate)
- Limit CORS configuration to the minimum necessary origins
- Do not log secrets (tokens, passwords). Mask them instead.

## Review Checks

When writing or reviewing code, confirm:
1. There are no string literals containing `api_key`, `token`, `secret`, `password`
2. `requests.get/post` is not called with `verify=False`
3. User input is not embedded into SQL via f-strings
