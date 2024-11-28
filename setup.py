import os
from setuptools import setup, find_packages

# Define the project metadata
NAME = "universal-consensus"
VERSION = "1.0.0"
DESCRIPTION = "A high-tech consensus algorithm for the universe"
AUTHOR = "Your Name"
AUTHOR_EMAIL = "your@email.com"
URL = "https://github.com/KOSASIH/universal-consensus"

# Define the dependencies
INSTALL_REQUIRES = [
    "flask==2.2.5",
    "flask_restful==0.3.8",
    "flask_sqlalchemy==2.5.1",
    "sqlalchemy==1.4.25",
    "psycopg2-binary==2.9.1",
    "requests==2.32.2",
    "cryptography==43.0.1",
]

# Define the testing dependencies
TESTS_REQUIRE = [
    "pytest==6.2.4",
    "pytest-cov==2.12.1",
    "coverage==6.3.2",
]

# Define the development dependencies
EXTRA_REQUIRE = {
    "dev": [
        "ipython==8.10.0",
        "jupyter==1.0.0",
    ],
}

# Define the package metadata
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    extras_require=EXTRA_REQUIRE,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
