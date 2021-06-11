from distutils.util import convert_path
from setuptools import setup, find_packages


def readme() -> str:
    with open('README.md') as f:
        return f.read()


version_dict = {}

with open(convert_path('cpsrl/version.py')) as file:
    exec(file.read(), version_dict)

setup(
    name='cpsrl',
    version=version_dict['__version__'],
    description='Continuous state-action RL with GPs',
    long_description=readme(),
    classifiers=['Programming Language :: Python :: 3.6'],
    author='Stratis Markou',
    author_email='em626@cam.ac.uk',
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'matplotlib',
        'tensorflow',
        'tensorflow_probability',
        'gym',
        'tqdm'
    ],
    zip_safe=False,
)
