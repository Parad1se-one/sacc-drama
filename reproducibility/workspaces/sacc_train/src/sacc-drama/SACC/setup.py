from setuptools import find_packages, setup

setup(
    name="sacc",
    version="0.1.0",
    description="Per-agent reward-machine HAPPO for the DRAMA benchmark",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "torch",
        "gymnasium",
        "tensorboard",
        "tqdm",
        "matplotlib",
    ],
)
