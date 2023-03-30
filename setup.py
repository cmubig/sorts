from setuptools import setup, find_packages
setup(
    name='sorts',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pandas==1.4.1',
        'matplotlib',
        'numpy',
        'scikit-learn==1.0.1',
        'scipy',
        'torch==1.13.0',
        'torchvision',
        'tensorboard==2.6.0',
        'tqdm',
        'imageio',
        'natsort'
    ],
    license='BSD 4-Clause License',
    long_description=open('README.md').read(),
)