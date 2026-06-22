from setuptools import setup, find_packages
setup(
    name="macha-ai",
    version="2.0.0",
    description="Macha AI - Unified open-source AI model",
    packages=find_packages(),
    install_requires=["torch>=2.0.0", "transformers>=4.30.0", "flask>=2.3.0", "flask-cors>=4.0.0", "numpy>=1.24.0", "requests>=2.31.0", "tqdm>=4.65.0"],
    python_requires=">=3.8",
)
