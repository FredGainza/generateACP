from setuptools import setup
setup(
    name='generate-acp-test11',
    packages_dir={"": "src"},
    install_requires=[
        "pandas >= 1.5.1",
        "numpy >= 1.23.5",
        "scikit-learn >= 1.1.3",
        "matplotlib >= 3.6.2",
        "jupyter >= 1.0.0",
        "adjustText >= 0.7.3"
    ],
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
