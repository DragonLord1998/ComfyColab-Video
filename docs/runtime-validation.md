# Video validation boundary

The pack-local suite proves the public node schema, graph expansion, catalog
selection, selected-download behavior, missing-upstream-node diagnosis, and
example workflow wiring without loading CUDA models.

It does not prove:

- that the full LTX-2.3 graph fits a specific Colab accelerator;
- output runtime or peak VRAM;
- audiovisual synchronization;
- visual or audio quality;
- the behavior of a future upstream revision.

Those claims require a bounded live Colab run against the exact manifest lock.
Record live evidence separately from local test results.
