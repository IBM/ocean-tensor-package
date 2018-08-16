# -------------------------------------------------------------------------
# Copyright 2018, IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

import sys as _sys_
import imp as _imp_

#print("Importing pyOceanDummy_gpu_v%d_%d" % (_sys_.version_info[0],_sys_.version_info[1]))
_module_name_ = "pyOceanDummy_gpu_v%d_%d" % (_sys_.version_info[0],_sys_.version_info[1])

# Check if the module exists
try :
   _imp_.find_module(_module_name_)
except ImportError:
   # Raise a ValueError to help differentiate between errors caused
   # by a missing library and import errors due to missing symbols
   # or other problems.
   raise ValueError("Module pyOceanDummy_gpu was not found")

# Import the module
_module_ = __import__(_module_name_, fromlist="*")

for _field_ in _module_.__dict__ :
   globals()[_field_] = _module_.__dict__[_field_]

# Clean up
del _field_
del _module_
del _module_name_
del _imp_
del _sys_
