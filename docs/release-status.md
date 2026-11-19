# Portfolio release status

| Scope | Status |
| --- | --- |
| Portfolio MVP Engineering | **PASS** on the installed offline Core and Demo packages. |
| FastAPI dashboard and OpenAPI | **Implemented** for the local A1/Mock workflow. |
| Independent M2 Formal | **FAIL**. Historical run consumed. Retry disabled. |
| M5 Formal | **NOT_RUN**. |
| Original Final Release | **BLOCKED** under its original acceptance criteria. |

Independent M2 executed 28 cases. Recall@3 was 0.750000 against a 0.90 gate and MRR@3 was 0.597222 against a 0.80 gate. Documentary and negative checks each passed 8 of 8, with zero unauthorized effects. The historical source commit is `9550d8611ec4fd4622dea8a8ae83a9ca11cc902f`. Its source-set SHA-256 is `d0659d3e25e60ef41f53a174cf245b352213d01045aa3b0977f6f014c747bebb`. The original evaluation and source-lock records are retained in independent custody.

An earlier formal M2F run also failed. Its [historical report](phase3-m2f-verification.md) and [public summary](evidence/m2f/formal-summary.json) record a separate evaluation.

The engineering evidence covers simulated ticket operations, installed packages and offline regression. A1 is the default demo retriever. Optional R1 and source-assertion results are limited to the documented fixture domain. Real enterprise deployment and generalization outside that domain are unverified.

See [the Portfolio demo](portfolio-demo.md) and its [historical receipt](portfolio-demo-receipt.json) for reproducible public examples.
