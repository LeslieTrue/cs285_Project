import setuptools



setuptools.setup(
    name="RewardTransformer",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "matplotlib",
    ],
)
