#!/usr/bin/env python

"""
Package installation setup
"""

import os
import re
import subprocess
from pathlib import Path

from setuptools import find_packages, setup

version = "0.0.1"
sha = 'Unknown'
src_folder = 'deepocr'
package_index = 'deepOCR'

cwd = Path(__file__).parent.absolute()

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
elif sha != 'Unknown':
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    except Exception:
        pass
    version += '+' + sha[:7]

print(f"Building wheel {package_index}_{version}")

with open(cwd.joinpath(src_folder, 'version.py'), 'w') as f:
    f.write(f"__version__ = '{version}'\n")

with open('README.md', 'r') as f:
    readme = f.read()


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strip
    specific version information.
    Args:
        fname (str): Path to requirements file.
        with_version (bool, default=False): If True, include version specs.
    Returns:
        info (list[str]): List of requirements items.
    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


setup(
    # Metadata
    name=package_index,
    version=version,
    author='Arijit Das',
    author_email='arijitd@gmail.com',
    maintainer='Arijit Das',
    description='Optical Character Recognition using Deep Learning',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/das-projects/deepOCR',
    license='Apache',
    classifiers=[
        'Development Status :: 1 - Beta',
        'Intended Audience :: Developers',
        "Intended Audience :: Education",
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=['OCR', 'deep learning', 'computer vision', 'pytorch', 'pytorch-lightning', 'text detection',
              'text recognition'],

    # Package info
    packages=find_packages(),
    zip_safe=True,
    python_requires='>=3.6.0',
    include_package_data=True,
    install_requires=parse_requirements('requirements.txt'),
    package_data={'': ['LICENSE']}
)
