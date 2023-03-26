from setuptools import find_packages, setup


with open('README.md', encoding='utf8') as file:
    long_description = file.read()


setup(
    name='emphases',
    description='Datasets and Scaling Laws for Neural Emphasis Prediction',
    version='0.0.0',
    author='Interactive Audio Lab',
    author_email='interactiveaudiolab@gmail.com',
    url='https://github.com/interactiveaudiolab/emphases',
    install_requires=[
        'librosa',
        'matplotlib',
        'penn',
        'pycwt',
        'pyfoal',
        'pypar',
        'pyyaml',
        'reseval',
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
