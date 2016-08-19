from setuptools import setup, find_packages

version = __import__('amimodels').get_version()

setup(name='amimodels',
      version=version,
      description='AMI Models',
      long_description="Models for AMI data",
      url='https://github.com/impactlab/amimodels/',
      author='Brandon T. Willard',
      author_email='brandon@theimpactlab.co',
      classifiers=[
          'Development Status :: 1 - Alpha',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
      ],
      keywords='open energy efficiency ami modeling',
      packages=find_packages(),
      install_requires=['eemeter >= 0.3.11',
                        'numpy',
                        'scipy',
                        'pandas >= 0.18',
                        'patsy==0.4.1',
                        'scikit-learn==0.17.1',
                        'holidays >= 0.4.1',
                        'pymc==2.3.6',
                        'Theano==0.8.1',
                        ],
      )
