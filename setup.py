from setuptools import setup, find_packages

setup(
    name="purogene",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pandas",
        "mysql-connector-python",
        "ollama",
        "sqlalchemy",
    ],
) 