from setuptools import setup

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["pandas", "numpy", "scikit-learn", "matplotlib", "jupyter", "adjustText", "pdfservices-sdk"]

setup(
    name='generateACP',
    packages_dir=["generateACP"],
    install_requires=requirements,
    version='0.1.0',
    description='Génréation des données et des graphiques d\'une Analyse en Composantes Principales ACP',
    author='Frédéric Gainza',
    author_email='kopatiktak@gmail.com',
    url='https://github.com/FredGainza/generateACP',
    license='MIT',
    classifiers = [
            "Framework :: Jupyter",
            "Topic :: Scientific/Engineering :: Visualization"]
)
