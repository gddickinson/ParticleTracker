"""
Setup script for ParticleTracker package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
if readme_file.exists():
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = 'Advanced Particle Detection and Tracking for Fluorescence Microscopy'

# Read requirements
requirements_file = Path(__file__).parent / 'requirements.txt'
if requirements_file.exists():
    with open(requirements_file, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = []

setup(
    name='particletracker',
    version='1.0.0',
    author='ParticleTracker Development Team',
    author_email='your.email@example.com',
    description='Advanced particle detection and tracking for fluorescence microscopy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ParticleTracker',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.0',
            'flake8>=3.9.0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.2',
        ],
        'gui': [
            'PyQt5>=5.15.0',
            'pyqtgraph>=0.12.0',
        ],
        'deep': [
            'torch>=1.9.0',
            'torchvision>=0.10.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'particletracker=particletracker.gui.main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords='microscopy particle-tracking fluorescence STORM PALM TIRF super-resolution',
)
