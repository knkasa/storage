from setuptools import setup, find_packages

setup(
    name='your-package-name',              # Package name
    version='0.1.0',                       # Version
    author='Your Name',                    # Author name
    author_email='your@email.com',         # Contact
    description='A short description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/your-repo',  # Optional
    packages=find_packages(),              # Probably better to have this.  It will automatically find all folders in your module.
    install_requires=[                     # Dependencies
        'numpy>=1.20',
        'pandas',
    ],
    classifiers=[                          # Optional Metadata
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
