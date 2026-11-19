# Architecture decisions

| Decision | Reason |
| --- | --- |
| Core and business application ship as separate Python packages. | The Harness stays independent of ticket schemas and web dependencies. |
| The default business demo uses fictional data, A1 retrieval and a deterministic Mock provider. | Approval and evidence behavior can be reproduced without model credentials. |
| Policy citations are checked against complete canonical source reads. | Search results guide source selection. Verified source content and trusted rules determine the business outcome. |
| The runtime binds an ASK approval to the exact write. | The permission engine, business rules, ticket revision and readback remain authoritative for side effects. |
| The dashboard serves a local, in-memory demonstration. | Loopback access, session isolation and exact approval binding fit the Portfolio scope. |
| R1 and typed source assertions are opt-in experiments. | Public fixture results and historical independent quality gates cover different scopes. See [release status](release-status.md). |
