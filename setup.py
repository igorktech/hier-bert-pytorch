from setuptools import setup, find_packages

setup(
    name='hier-bert-pytorch',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'tokenizers',
        'sentencepiece',
        'seaborn',
        'matplotlib',
    ],
    author='Igor',
    author_email='igorkuz.tech@gmail.com',
    description='Implementation og hierarchica attention bert',
    url='https://github.com/igorktech/hier-bert-pytorch',
)
