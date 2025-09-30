from setuptools import setup, find_packages

setup(
    name="droidflow",
    version="0.0.1.4",
    description="A multi agent orchestrator library",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Kaan Simsek",
    author_email="kaan.simsek01@gmail.com",
    url="https://github.com/KaanSimsek/droidflow",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["google-generativeai>=0.5.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
