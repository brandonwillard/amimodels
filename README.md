AMI Meter Models
================


Implementations of AMI data models to be used within the
[`eemeter`](https://github.com/impactlab/eemeter) framework.  

Documentation
-------------

TODO: Docs on [RTD](http://amimodels.readthedocs.org/en/latest/).

Dev Installation
----------------

    $ git clone https://github.com/openeemeter/amimodels
    $ cd amimodels
    $ mkvirtualenv amimodels
    (amimodels)$ pip install -e .
    (amimodels)$ pip install -r dev_requirements.txt
    (amimodels)$ workon # gives you access to virtualenv py.test executable

Testing
-------

This library uses the py.test framework. To develop locally, clone the repo,
and in a virtual environment execute the following commands:

    $ py.test

If you run into problems with the py.test executable, please ensure that you
are using the virtualenv py.test:

    $ py.test --version

Licence
-------

MIT
