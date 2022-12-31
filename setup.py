from setuptools import find_packages, setup


with open('README.md', encoding='utf8') as file:
    long_description = file.read()


setup(
    name='emphases',
    description='Representations of prosody for detecting emphases',
    version='0.0.0',
    author='Interactive Audio Lab',
    author_email='interactiveaudiolab@gmail.com',
    url='https://github.com/interactiveaudiolab/emphases',
    install_requires=[
        'matplotlib',
        'pycwt',
        'pyfoal',
        'pypar',
        'pyyaml',
        'scipy',
        'tensorboard',
        'torch',
        'torchaudio',
        'tqdm',
        'yapecs'],
    packages=find_packages(),
    package_data={'emphases': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['audio', 'loudness', 'pitch', 'prosody', 'speech'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
