from setuptools import setup, find_packages


setup(
    name="project_ml",
    version="1.0",
    packages= find_packages(),
    install_requires=[

        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "transformers",
        "datasets",
        "tensorflow",
        "tf-keras"
],

author="Saul Mora",
description = " A machine learning model used for sentiment analysis using Naives & BERT",
)