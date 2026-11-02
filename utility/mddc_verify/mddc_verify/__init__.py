#
# Copyright 2026 SALMON developers
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
"""Verification of SALMON Ehrenfest-MD, dft_md and divide-and-conquer (DC) runs.

Three layers:
  generic   - exit status, normal termination, convergence, finite values, files, timer table
  physics   - per mode (ehrenfest, dft_md, dc, plus plain gs/rt)
  compare   - cross-run comparisons (CPU/GPU, 2.3.0/develop, original/modified)

Every numerical predicate is NaN-safe: a comparison involving a non-finite number never passes.
Core checks use only the standard library; figures need matplotlib (optional).
"""

__version__ = "0.1.0"
