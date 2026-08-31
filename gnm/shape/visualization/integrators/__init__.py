# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module that collects and registers all custom GNM integrators."""

from __future__ import annotations

import functools
import importlib

import mitsuba as mi  # pyrefly: ignore[missing-import]


def _integrators_variant_callback(old: str | None, new: str) -> None:
  """Imports and registers all custom integrators.

  This follows Mitsuba's structure to allow changing the variant for custom
  plugins.

  Args:
    old: The old variant.
    new: The new variant.
  """
  del old  # unused

  if new is None or new.startswith('scalar'):
    return

  # pylint: disable=g-import-not-at-top,import-outside-toplevel
  from gnm.shape.visualization.integrators import gnm_half_lambert_integrator
  # pylint: enable=g-import-not-at-top,import-outside-toplevel

  importlib.reload(gnm_half_lambert_integrator)
  gnm_half_lambert_integrator.register()


@functools.cache
def register() -> None:
  """Registers all custom GNM integrators.

  If integrators were already registered, this is a no-op.
  """
  mi.detail.add_variant_callback(_integrators_variant_callback)
  if variant := mi.variant():
    _integrators_variant_callback(None, variant)
