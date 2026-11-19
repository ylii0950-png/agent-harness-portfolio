# Contributing

Use Python 3.11 or 3.12 and the locked uv environments in the [reproduction guide](docs/reproduce.md). Run the Core and Demo tests, lint, formatting, type checks and package builds before submitting changes.

- Keep default tests and examples offline and credential-free. Use fictional policy documents, tickets and identities for the demo.
- Core is independent of the business package. The Demo uses public Core interfaces. Permission checks, exact approvals, workspace containment and execution budgets apply to every tool invocation.
- Model output, file content and Skills are untrusted inputs. Keep local credentials, model weights, private evaluation inputs and raw session traces out of Git.
- Preserve the project and upstream licenses, attribution and reference commit. Keep the frozen public evidence inputs byte-identical when they are bound by a source lock.
- Describe results using the scope in [release status](docs/release-status.md). Run the offline check suite for engineering changes.
