import setuptools

# Package information
name = 'multilearn'
version = '0.0.5'  # Need to increment every time to push to PyPI
description = 'Multi-task learning with Pytorch.'
url = 'https://github.com/leschultz/multilearn'
author = 'Lane E. Schultz'
author_email = 'laneenriqueschultz@gmail.com'
python_requires = '>=3.10'
classifiers = ['Programming Language :: Python :: 3',
               'License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               ]
packages = setuptools.find_packages(where='src')
install_requires = [
                    'torch',
                    'scikit-learn',
                    'lightning',
                    'pandas',
                    'matplotlib',
                    'dill',
                    'pytest',
                    ]

long_description = open('README.md').read()

# Passing variables to setup
setuptools.setup(
                 name=name,
                 version=version,
                 description=description,
                 url=url,
                 author=author,
                 author_email=author_email,
                 packages=packages,
                 package_dir={'': 'src'},
                 package_data={'multilearn': ['data/*']},
                 python_requires=python_requires,
                 classifiers=classifiers,
                 install_requires=install_requires,
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 )
