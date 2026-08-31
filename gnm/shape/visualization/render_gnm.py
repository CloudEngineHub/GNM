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

"""Visualization utilities for GNM."""

from gnm.shape import gnm_numpy
from gnm.shape.visualization import camera_conversions
from gnm.shape.visualization import gnm_pyrender
from gnm.shape.visualization import render_common
from gnm.shape.visualization import vertex_colors as vertex_colors_module
import numpy as np
import numpy.typing as npt


def render_gnm(
    gnm_np: gnm_numpy.GNM,
    vertices: render_common.FloatArray | None = None,
    world_to_camera: render_common.FloatArray | None = None,
    camera_to_image: render_common.FloatArray | None = None,
    image_size: tuple[int, int] = render_common.DEFAULT_IMAGE_SIZE,
    triangles: str | npt.NDArray[np.integer] = '~eye_exteriors',
    texture: render_common.Texture = render_common.DEFAULT_TEXTURE,
    multisample_antialiasing: int = 2,
    background_color: render_common.ColorOrImage = (
        render_common.DEFAULT_BACKGROUND_COLOR
    ),
    alpha: float = 1.0,
    vertex_colors: npt.NDArray[np.floating] | None = None,
    multiple_gnms: bool = False,
    include_shading: bool = True,
    verbose: bool = False,
) -> render_common.FloatArray:
  """Render GNM meshes.

  Uses a pyrender backend to render GNM meshes.

  All arguments specified with (...) in their shape are batchable. They can have
  an arbitrary number of leading dimensions, but they must all be broadcastable
  to the batch size of the biggest of them. e.g. if vertices has shape (3, 100,
  3) and world_to_camera has shape (4, 4), then world_to_camera will be
  broadcast over 3 frames.

  The user can pass the world-to-camera and camera-to-image transformations. If
  they are not given, then cameras are placed in front of each face, looking at
  the face. Note that this function assumes that the world-to-camera and
  camera-to-image transformations follow OpenCV's convention. This means
  that the camera coordinate system has X pointing to the right, Y downwards,
  and Z towards the scene. The camera-to-image
  should project points in the camera coordinate system to normalized
  coordinates in [-1, 1].

  Recall the following GNM dimension notation (from gnm_numpy.py):
  * N: Size of batch.
  * V: Number of vertices.
  * J: Number of joints.
  * I: Identity basis dimensionality.
  * E: Expression basis dimensionality.

  Additionally, we use:
  * M: Number of GNMs per image.

  Args:
    gnm_np: The NumPy GNM model, e.g., if you have a custom version of GNM you
      wish to pose with.
    vertices: Optional posed GNM vertices shaped (..., [M], V, 3). If not
      provided, use the template vertices.
    world_to_camera: Optional world-to-camera transformations, (..., 4, 4). If
      not given, use a default look-at transformation.
    camera_to_image: Optional camera-to-image transformations, (..., 4, 4). If
      not given, use a default fill factor.
    image_size: The width and height of the rendered image in pixels: (W, H).
    triangles: Determines which triangles to render in GNM. Either a vertex
      group name: triangles in that GNM vertex group will be rendered. Or an
      array of triangle indices.
    texture: Optional texture map(s) for GNM. If None, no texture will be used.
      Defaults to a skin edge-flow texture. If given as a single array, this
      will be used for the skin only. If given as a dictionary, then the keys
      should be GNM part names, and the values should be arrays. All arrays
      should be float32 [0-1], (..., H, W, 3).
    multisample_antialiasing: Internally render with e.g., double resolution,
      and then downsample for anti-aliasing.
    background_color: Color for the background. Can either be float (gray), an
      RGB tuple, or a background image (..., H, W, 3).
    alpha: An optional float value in [0, 1] range. If provided, it will be used
      to blend the rasterized GNM meshes with the background images.
    vertex_colors: Per-vertex colors, (..., [M], V, 3). In the case of multiple
      GNMs, 'M' will be inferred - if it matches the M dimension in vertices, it
      will be used, otherwise vertex_colors will be broadcast over the M
      dimension in vertices.
    multiple_gnms: If True, vertices is expected to be shape (..., M, V, 3), and
      we render M GNMs per image. Default cameras will be set relative to the
      first GNM in the sequence.
    include_shading: Whether to include shading. If False, the mesh will be
      rendered without light or shading.
    verbose: Whether to print progress bars.

  Returns:
    A rendered image of GNM, (..., H, W, 3).

  Raises:
    ValueError: If multiple_gnms=True, but vertices is only 2D.
  """

  width, height = image_size

  triangle_dict = {}
  all_triangle_indices = triangles
  if isinstance(triangles, str):
    all_triangle_indices = gnm_np.triangle_indices_for_group(triangles)

  for part_name in gnm_np.mesh_component_names:
    group_triangle_indices = gnm_np.triangle_indices_for_group(part_name)
    intersection = np.intersect1d(group_triangle_indices, all_triangle_indices)
    triangle_dict[part_name] = gnm_np.triangles[intersection]

  if vertices is None:
    vertices = gnm_np.template_vertex_positions

  if not multiple_gnms:
    vertices = vertices[..., None, :, :]  # Inject 'M' dimension.
  elif (vertices_dim := vertices.ndim) < 3:
    raise ValueError(
        f'Called with {multiple_gnms=}, but vertices is only {vertices_dim}D.'
    )

  if vertex_colors is None:
    vertex_colors = vertex_colors_module.get_vertex_colors(
        gnm_np, vertex_colors_module.DEFAULT_COLOR
    )

  colors_has_m_dim = (0, 0, *vertex_colors.shape)[-3] == vertices.shape[-3]
  if not colors_has_m_dim:
    # 'M' dimensions is not present but is required. Inject the dimension, and
    # expand to match the size of vertices' M dimension.
    vertex_colors = vertex_colors[..., None, :, :]
    vertex_colors_shape = list(vertex_colors.shape)
    vertex_colors_shape[-3] = vertices.shape[-3]
    vertex_colors = np.broadcast_to(vertex_colors, vertex_colors_shape)

  # Define default camera params based on the first GNM in the 'M' dimension.
  vertices_for_cameras = vertices[..., 0, :, :]

  if world_to_camera is None:
    world_to_camera = render_common.get_look_at_world_to_camera(
        gnm_np,
        vertices_for_cameras,
    )

  if camera_to_image is None:
    camera_to_image = render_common.get_fill_factor_camera_to_image(
        gnm_np, vertices_for_cameras, image_size=image_size
    )

  # Convert from OpenCV to OpenGL convention.
  world_to_camera = camera_conversions.opencv_extrinsics_to_opengl(
      world_to_camera
  )
  camera_to_image = (
      camera_conversions.opencv_intrinsics_matrix_to_opengl_view_matrix(
          camera_to_image,
          width=image_size[0],
          height=image_size[1],
          near=render_common.DEFAULT_NEAR,
          far=render_common.DEFAULT_FAR,
      )
  )

  vertex_normals = gnm_np.compute_vertex_normals(vertices)

  # Broadcast a background color to an image.
  if not isinstance(background_color, np.ndarray) or background_color.ndim == 1:
    background_color = np.broadcast_to(background_color, (height, width, 3))

  # Convert background color to float [0-1].
  if background_color.dtype == np.uint8:
    background_color = background_color.astype(np.float32) / 255.0

  texture_dict = render_common.load_texture(gnm_np, texture)
  textures = list(texture_dict.values())
  texture_keys = texture_dict.keys()
  if not set(texture_keys).issubset(gnm_np.mesh_component_names):
    missing_parts = set(texture_keys) - set(gnm_np.mesh_component_names)
    raise ValueError(
        f'Texture keys {missing_parts} are not GNM part names'
        f' {gnm_np.mesh_component_names}.'
    )

  # Find the maximum batch dimension that satisfies all batch-able arguments.
  try:
    batch_dims = render_common.get_batch_dim(
        (vertices, 3),
        (vertex_colors, 3),
        (world_to_camera, 2),
        (camera_to_image, 2),
        (background_color, 3),
        *[(t, 3) for t in textures],
    )
  except ValueError as e:
    raise ValueError(
        f' Batch dimensions incompatible: vertices {vertices.shape},'
        f' vertex_colors {vertex_colors.shape}, world_to_camera'
        f' {world_to_camera.shape}, camera_to_image {camera_to_image.shape},'
        f' background_color {background_color.shape}, texture'
        f' {[t.shape for t in textures]}.'
    ) from e

  def batchify(arr, non_batch_dims):
    """Broadcast to batch dimensions, and flatten the batch dimensions."""
    arr = np.broadcast_to(arr, (*batch_dims, *arr.shape[-non_batch_dims:]))
    return arr.reshape(int(np.prod(batch_dims)), *arr.shape[-non_batch_dims:])

  vertices = batchify(vertices, 3)
  vertex_normals = batchify(vertex_normals, 3)
  vertex_colors = batchify(vertex_colors, 3)
  world_to_camera = batchify(world_to_camera, 2)
  camera_to_image = batchify(camera_to_image, 2)
  background_color = batchify(background_color, 3)
  batched_textures = {part: batchify(x, 3) for part, x in texture_dict.items()}

  renders = gnm_pyrender.render(
      vertices=vertices,
      triangles=triangle_dict,
      world_to_camera=world_to_camera,
      camera_to_image=camera_to_image,
      image_size=image_size,
      texture=batched_textures,
      vertex_colors=vertex_colors,
      multisample_antialiasing=multisample_antialiasing,
      vertex_uvs=gnm_np.vertex_uvs,
      vertex_normals=vertex_normals,
      background_color=background_color,
      alpha=alpha,
      include_shading=include_shading,
      verbose=verbose,
  )

  width, height = image_size
  color = renders.reshape(*batch_dims, height, width, 3)

  return color
