This is a repo forked from:
# Fashion-MNIST

[![GitHub stars](https://img.shields.io/github/stars/zalandoresearch/fashion-mnist.svg?style=flat&label=Star)](https://github.com/zalandoresearch/fashion-mnist/)

It serves as a benchmarking system for information retrieval using topic modeling techniques for text data. 

<details><summary>Table of Contents</summary><p>

* [Introduction](#introduction)
* [Basic Principles](#basic-principles)
* [Benchmark](#benchmark)
* [Contributing](#contributing)
* [Codebase Structure](#codebase-structure)
* [Contact](#contact)
* [License](#license)
</p></details><p></p>

## Benchmark
We built an automatic benchmarking system based on `scikit-learn` that covers 129 classifiers (but no deep learning) with different parameters. [Find the results here](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/).

<img src="doc/img/benchmark.gif" width="100%">

You can reproduce the results by running `benchmark/runner.py`. We recommend building and deploying [this Dockerfile](Dockerfile). 

## Contributing

Thanks for your interest in contributing! There are many ways to get involved; start with our [contributor guidelines](/CONTRIBUTING.md) and then check these [open issues](https://github.com/zalandoresearch/fashion-mnist/issues) for specific tasks.

## Codebase Structure

```console
.
|-- Dockerfile
|-- LICENSE
|-- MAINTAINERS
|-- README.md
|-- benchmark
|   |-- __init__.py
|   |-- baselines.json
|   `-- runner.py
|-- configs.py
|-- requirements.txt
|-- static
|   |-- css
|   |   `-- main.css
|   |-- img
|   |   `-- research_logo.png
|   |-- index.html
|   `-- js
|       `-- vue-binding.js
|-- utils
|   |-- __init__.py
|   |-- argparser.py
|   |-- helper.py
|   `-- mnist_reader.py
```

`configs.py` is the main configuration file. 
Variables including `LOGGER, BASELINE_PATH, JSON_LOGGER` were used in `runner.py`.




## Contact
[To discuss the repository](https://www.linkedin.com/in/sara-zeng/) 

## License

The MIT License (MIT) Copyright © [2017] Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.