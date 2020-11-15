from setuptools import setup, find_packages


setup(
    name='DLToolbox',
    version='a0.0.1',
    description='This is a deep learning tools.',
    author='MoozIiSP',
    author_email='yuaolni@gmail.com',
    url='https://www.github.com/MoozIiSP/DLToolbox',
    packages=['toolbox.' + m for m in find_packages('toolbox')],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.8',
)