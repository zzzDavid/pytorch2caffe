import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# dependencies
INSTALL_REQUIRES = [
    "torch>=1.0.0",
    "torchvision>=0.4.0",
    "numpy",
    "lmdb",
    "protobuf>=3.13.0"
]

setuptools.setup(
    name="pytorch2caffe",
    version="0.0.1",
    license="MIT",
    author="Niansong Zhang",
    author_email="nz264@cornell.edu",
    description="Pytorch2Caffe is a tool to convert pytorch models to (BVLC) Caffe network files (prototxt) and parameter files (caffemodel).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zzzDavid/pytorch2caffe",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=INSTALL_REQUIRES
)