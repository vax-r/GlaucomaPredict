from setuptools import setup, find_packages

setup(
    name="glaucoma_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'pyyaml',
        'tqdm'
    ]
)