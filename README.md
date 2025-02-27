# flowevidence: Estimate bayesian evidences using Normalizing Flows

`flowevidence` is a Python package that provides evidence estimations from a set of MCMC samples and the associated unnormalized (log-)posterior values. 

## Getting started:
`flowevidence` estimate the posterior density by training a flow architecture directly on the samples. Then, for each of them, an evidence estimation can be computed as the ratio of the associated unnormalized posterior value and the flow pdf prediction.

The package documentation is available at [this link](https://asantini29.github.io/flowevidence/). Check out the examples directory for more info (TODO).

### Prerequisites:

flowevidence heavily depends on `pytorch` and `normflows`.

## Installing:
1. Clone the repository:
 ```
 git clone https://github.com/asantini29/flowevidence.git
 cd flowevidence
 ```
2. Run install:
 ```
 python setup.py install
 ```

## Versioning

We use [SemVer](http://semver.org/) for versioning. 

Current Version: 0.0.1

## Authors

* **Alesandro Santini**

### Contributors

Get in touch if you would like to contribute!

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Citing

If you use `flowevidence` in your research, you can cite it in the following way:
(TODO)

A previous work in this context can be found in [ArXiv:2404.12294](https://arxiv.org/abs/2404.12294). Please consider citing it as well.

## References

The idea of translating the evidence-estimation problem in a density-estimation one can also be found in ["Statistics, Data Mining, and Machine Learning in Astronomy"](https://press.princeton.edu/books/hardcover/9780691198309/statistics-data-mining-and-machine-learning-in-astronomy-pdf), Željko, Andrew, Jacob, and Gray. Princeton University Press, 2012.