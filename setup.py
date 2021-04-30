# SPDX-Copyright: Copyright (c) Capital One Services, LLC
# SPDX-License-Identifier: Apache-2.0
# Copyright 2021 Capital One Services, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from io import open

from setuptools import setup

CURR_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(CURR_DIR, "README.md"), encoding="utf-8") as file_open:
    LONG_DESCRIPTION = file_open.read()

DESCRIPTION = "Global Explanations for Deep Neural Networks"

extras_require={
    "complete": [
        "dask-ml >= 1.8.0",
        "matplotlib >= 3.3.4",
        "pandas >= 1.1.3",
        "scikit-learn >= 0.23.2",
        "numpy >= 1.19.2"
    ]
}

setup(
    name="gam",
    version="1.1.1",
    packages=["gam", "tests"],
    maintainer="Brian Barr",
    maintainer_email="brian.barr@capitalone.com",
    url="https://github.com/capitalone/global-attribution-mapping",
    classifiers=["Programming Language :: Python :: 3"],
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    install_requires=[
        "py == 1.8.2",
        "dask[complete] >= 2021.2.0",
        "matplotlib >= 3.3.4",
        "pandas >= 1.1.3",
        "scikit-learn >= 0.23.2",
        "numpy >= 1.19.2"
    ],
    extras_require=extras_require,
    python_requires=">=3.6",
)
