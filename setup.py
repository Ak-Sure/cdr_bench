from setuptools import setup, find_packages

setup(
    name='cdr_bench',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'h5py',
        'numpy',
        # Add other dependencies here
    ],
    author='Alexey Orlov',
    author_email='2alexeyorlov@gmail.com',
    description='Scripts for perform dimensionality reduciton on chemical datasets',
    #url='https://github.com/yourusername/your-repo',  # Update with your repo URL
    #classifiers=[
    #    'Programming Language :: Python :: 3',
    #    'License :: OSI Approved :: MIT License',
    #    'Operating System :: OS Independent',
    #],
    #python_requires='>=3.6',
)
