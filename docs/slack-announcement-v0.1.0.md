*robotframework-testselection v0.1.0* — now available on PyPI

Run fewer tests. Keep your coverage.

`robotframework-testselection` uses semantic embeddings to select maximally diverse subsets of your Robot Framework test suites. In our benchmark against robotframework-doctestlibrary, selecting just *40% of tests retained 96% of code coverage*.

*What it does:*
- Embeds test cases as 384-dim vectors (name + tags + keyword tree) using sentence-transformers
- Selects diverse subsets via Farthest Point Sampling (or k-Medoids, DPP, Facility Location)
- Integrates with Robot Framework via PreRunModifier and Listener v3 — works with DataDriver tests too
- Caches embeddings with content hashing — re-vectorization only when sources change

*Get started:*
```
pip install robotframework-testselection[vectorize]
testcase-select run --suite tests/ --k 20 --strategy fps
```

*Links:*
- GitHub: https://github.com/manykarim/robotframework-testselection
- PyPI: https://pypi.org/project/robotframework-testselection/
- Docs: https://github.com/manykarim/robotframework-testselection#readme
