from setuptools import setup, find_packages
setup(
    name='sorts',
    version='1.0',
    packages=find_packages(['gym.*', 'game.*'], exclude=['social-patternn']),
    install_requires=[
        'pandas==1.4.1',
        'matplotlib',
        'numpy',
        'scikit-learn==1.0.1',
        'scipy',
        'torch==1.13.0',
        'tensorboard==2.6.0',
        'torchvision==4.61.2',
        'tqdm',
    ],
    license='MIT License',
    long_description=open('README.md').read(),
)