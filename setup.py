from setuptools import setup
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(name = "helpers-kendeegz",
      version = "0.0.1",
      desrciption = "Helpers for statistics and data management.",
      url = "https://github.com/kendeegz/helpers",
      author = "Kenneth P. Callahan",
      author_email = "kennydeegz@gmail.com",
      license = "MIT",
      packages = setuptools.find_packages(where = "src"),
      python_requires=">=3.8",
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
      ],
      zipsafe = False)
