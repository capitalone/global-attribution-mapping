# SPDX-Copyright: Copyright (c) Capital One Services, LLC
# SPDX-License-Identifier: Apache-2.0
# Copyright 2018 Capital One Services, LLC
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

from setuptools import setup

import os 
from io import open 

CURR_DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(CURR_DIR, "README.md"), encoding="utf-8") as file_open:
    LONG_DESCRIPTION = file_open.read()

DESCRIPTION = 'Global Explanations for Deep Neural Networks'

setup(
    name='GAM',
    version='0.0.1',
    packages=['gam', 'tests'],
    maintainer='Brian Barr',
    maintainer_email='brian.barr@capitalone.com',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license='Apache License 2.0',
    install_requires=[
        "matplotlib > 2.2.0",
        "pandas > 0.25.1",
        "scikit-learn > 0.21.3",
        "numpy > 1.17.2",
    ],
)
