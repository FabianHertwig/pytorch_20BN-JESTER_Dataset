import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jesterdataset",
    version="0.0.1",
    author="Fabian Hertwig",
    author_email="fabian.hertwig@gmail.com",
    description="A Pytorch Dataset to load the 20BN-JESTER hand gesture dataset or datasets that have the same format.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/FabianHertwig/pytorch_20BN-JESTER_Dataset",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["torch", "torchvision"],
)