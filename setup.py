from setuptools import setup, find_packages

setup(
    name='flowevidence',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 'matplotlib', 'torch', 'nflows', 'corner'
    ],
    entry_points={
        'console_scripts': [
            # Add command line scripts here
            # e.g., 'flowevidence=flowevidence.cli:main'
        ],
    },
    author='Alessandro Santini',
    author_email='alessandro.santini@aei.mog.de',
    description='A package to compute the evidence of a model using normalizing flows',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/asantini/flowevidence',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.9',
)