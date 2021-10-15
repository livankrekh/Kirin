import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Kirin-livankrekh",
    version="0.0.1",
    author="Livii Iabanzhi",
    author_email="iabanzhi.livii@gmail.com",
    description="Deep learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/livankrekh/Kirin",
    project_urls={
        "Bug Tracker": "https://github.com/livankrekh/Kirin/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
)
