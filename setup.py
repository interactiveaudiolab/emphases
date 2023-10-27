from setuptools import find_packages, setup


with open('README.md', encoding='utf8') as file:
    long_description = file.read()


setup(
    name='emphases',
    description='Crowdsourced and Automatic Speech Prominence Estimation',
    version='0.0.1',
    author='Interactive Audio Lab',
    author_email='interactiveaudiolab@gmail.com',
    url='https://github.com/interactiveaudiolab/emphases',
    install_requires=[
        'librosa',
        'matplotlib',
        'numpy',
        'pandas',
        'penn',
        'pycwt',
        'pyfoal',
        'pypar',
        'pyyaml',
        'reseval',
        'scipy',
        'tensorboard',
        'torch<2.0.0',
        'torchutil',
        'torchaudio<2.0.0',
        'tqdm',
        'yapecs'],
    packages=find_packages(),
    package_data={'emphases': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['annotatation', 'audio', 'emphasis', 'prominence', 'speech'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
