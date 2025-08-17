from setuptools import setup, find_packages

setup(
    name="droidflow",
    version="0.0.1",
    description="A multi agent orchestrator library",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Kaan Simsek",
    author_email="kaan.simsek01@gmail.com",
    url="https://github.com/KaanSimsek/droidflow",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        # Add your dependencies here, e.g.:
        # "requests>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    include_package_data=True,
)
