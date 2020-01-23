import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='hdmpy-maxhuppertz',
    version='0.0.1',
    author='Maximilian Huppertz',
    author_email='mhupp@umich.edu',
    description='The hdmpy package is a Python port of the R package hdm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/maxhuppertz/hdmpy',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
