# FlowEvidence

`flowevidence` is a Python package that provides evidence estimations from a set of MCMC samples and the associated unnormalized (log-)posterior values. 

## Getting started:
`flowevidence` estimates the posterior density by training a flow architecture directly on the samples. Then, for each of them, an evidence estimate can be computed as the ratio of the associated unnormalized posterior value and the flow pdf prediction.

The source code is hosted [here](https://github.com/asantini29/flowevidence].

## Documentation Contents
- [Installation](installation.md)
- [Usage (TODO)](usage.md)
- [API Reference](api/core.md)