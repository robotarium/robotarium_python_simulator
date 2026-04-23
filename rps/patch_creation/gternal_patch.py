import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path


@dataclass
class GTERNALPatchData:
    """
    Patch data container for a GTernal robot used in the Robotarium.

    Attributes:
        vertices : (V x 2) float array
            All polygon vertices in the robot body frame.
        faces : list of 1-D int arrays
            Each entry is an array of vertex indices (0-based) forming one
            closed polygon.
        colors : (P x 3) float array
            One RGB row per face/part.
    """
    vertices: NDArray[np.floating]
    faces: list
    colors: NDArray[np.floating]


def gternal_patch() -> GTERNALPatchData:
    """
    Generate patch data for the GTernal robot used in the Robotarium.
    YOU SHOULD NEVER HAVE TO USE THIS FUNCTION DIRECTLY — it is called
    internally by :class:`Robotarium` to build the robot visualisation.

    Returns a :class:`GTERNALPatchData` containing vertices, faces, and
    per-face colours for the GTernal tombstone-shaped body.

    The GTernal body is a "tombstone" shape: a rectangle with a
    semicircular arch on the top (front) and a flat base on the bottom
    (rear).  Two drive wheels protrude from the sides.  A single RGB LED
    is located in the front-right of the robot.

    Coordinate convention (body frame, facing +x / "right" on screen):
        +x = forward (front/arch of robot),  +y = left,
        origin = centre of the wheel axle.

    Reference:
        Kim, Soobum, et al. "GTernal: A Robot Design for the Autonomous
        Operation of a Multi-robot Research Testbed." DARS, 2024.
    """
    # ------------------------------------------------------------------
    # Physical dimensions (metres, consistent with ARobotarium constants)
    #   robot_diameter  = 0.11 m  -> body_width  = 0.11 m
    #   wheel_radius    = 0.016 m -> rendered wheel height = 0.032 m
    #
    # Tombstone geometry:
    #   - Width       : body_width  (= robot_diameter)
    #   - Rect height : body_rect_h  (rear flat section)
    #   - Arch radius : body_width / 2  (semicircle caps the top)
    #   - Total height: body_rect_h + arch_radius
    #
    # Origin is at the centre of the wheel axle.
    #   y_axle = 0  (axle centre)
    #   rear of body at y = -wheel_radius
    #   front of body at y = -wheel_radius + body_rect_h + arch_radius
    # ------------------------------------------------------------------
    wheel_width  = 0.008
    wheel_height = 0.032           # = 2 * wheel_radius
    wheel_radius = wheel_height / 2

    # Derived from physical constraints:
    #   outside wheel to outside wheel = 0.11 m  -> body_width = 0.11 - wheel_width
    #   back edge to front of arch     = 0.095 m -> body_rect_h = 0.095 - arch_radius
    body_width  = 0.110 - wheel_width   # = 0.102 m
    arch_radius = body_width / 2        # = 0.051 m
    body_rect_h = 0.095 - arch_radius   # = 0.044 m

    led_half = 0.008  # half side-length of square LED

    # y offset: shift all geometry so axle centre is at y = 0.
    yo = -wheel_radius

    # ------------------------------------------------------------------
    # 1. Body — tombstone outline
    #    Flat base at y=yo, straight sides, semicircular arch at top.
    # ------------------------------------------------------------------
    n_arch = 24
    arch_angles = np.linspace(0, np.pi, n_arch)   # right (0) to left (pi)
    arch_x = arch_radius * np.cos(arch_angles)
    arch_y = yo + body_rect_h + arch_radius * np.sin(arch_angles)
    arch_pts = np.column_stack([arch_x, arch_y])  # (n_arch, 2)

    body_verts = np.vstack([
        [-body_width / 2, yo],                      # bottom-left
        [ body_width / 2, yo],                      # bottom-right
        [ body_width / 2, yo + body_rect_h],        # top-right rect corner
        arch_pts[1:-1],                             # arch interior (no dup corners)
        [-body_width / 2, yo + body_rect_h],        # top-left rect corner
    ])
    n_body = len(body_verts)

    # ------------------------------------------------------------------
    # 2. Left wheel — centred on axle (y=0), rear edge at y=-wheel_radius
    # ------------------------------------------------------------------
    wx_l = -(body_width / 2 + wheel_width / 2)
    left_wheel = np.array([
        [wx_l - wheel_width / 2,  wheel_radius],
        [wx_l + wheel_width / 2,  wheel_radius],
        [wx_l + wheel_width / 2, -wheel_radius],
        [wx_l - wheel_width / 2, -wheel_radius],
    ])

    # ------------------------------------------------------------------
    # 3. Right wheel
    # ------------------------------------------------------------------
    wx_r = body_width / 2 + wheel_width / 2
    right_wheel = np.array([
        [wx_r - wheel_width / 2,  wheel_radius],
        [wx_r + wheel_width / 2,  wheel_radius],
        [wx_r + wheel_width / 2, -wheel_radius],
        [wx_r - wheel_width / 2, -wheel_radius],
    ])

    # ------------------------------------------------------------------
    # 4. Single LED — front-right of robot body.
    #    Placed at colour-slot index 3 (0-based), which is the slot
    #    Robotarium.draw_robots() overwrites with the current LED RGB.
    # ------------------------------------------------------------------
    led_cx = body_width * 0.28
    led_cy = yo + body_rect_h + arch_radius * 0.50
    led_verts = np.array([
        [led_cx - led_half, led_cy + led_half],
        [led_cx + led_half, led_cy + led_half],
        [led_cx + led_half, led_cy - led_half],
        [led_cx - led_half, led_cy - led_half],
    ])

    # ------------------------------------------------------------------
    # Assembly
    #   Part 0 : body
    #   Part 1 : left wheel
    #   Part 2 : right wheel
    #   Part 3 : LED  <-- recoloured by Robotarium.draw_robots() at runtime
    # ------------------------------------------------------------------
    vertices = np.vstack([body_verts, left_wheel, right_wheel, led_verts])

    # Build faces as a list of 0-based index arrays (one per part)
    part_sizes = [n_body, 4, 4, 4]
    faces = []
    offset = 0
    for sz in part_sizes:
        faces.append(np.arange(offset, offset + sz, dtype=int))
        offset += sz

    # ------------------------------------------------------------------
    # Colours (one RGB row per part)
    # ------------------------------------------------------------------
    gt_gold   = np.array([179, 163, 105]) / 255.0
    dark_grey = np.array([0.20, 0.20, 0.20])
    led_white = np.array([1.00, 1.00, 1.00])

    colors = np.vstack([gt_gold, dark_grey, dark_grey, led_white])

    return GTERNALPatchData(vertices=vertices, faces=faces, colors=colors)


class GTERNALRobotPatch:
    """
    Matplotlib artist wrapper for a single GTernal robot.

    Creates and owns the :class:`~matplotlib.patches.PathPatch` objects
    for the body, wheels, and LED of one robot.  Call
    :meth:`set_pose` each frame to move the robot and :meth:`set_led`
    to update the LED colour.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        The axes on which to draw this robot.
    pose : array-like, shape (3,)
        Initial ``[x, y, theta]`` pose in world coordinates.
    patch_data : GTERNALPatchData, optional
        Pre-computed patch data from :func:`gternal_patch`.  If omitted
        a new one is generated (useful when every robot shares the same
        template geometry).
    """

    LED_FACE_IDX = 3  # 0-based index of the LED face in patch_data

    def __init__(
        self,
        axes: plt.Axes,
        pose: NDArray[np.floating],
        patch_data: Optional[GTERNALPatchData] = None,
    ):
        if patch_data is None:
            patch_data = gternal_patch()

        self._data = patch_data
        self._axes = axes
        self._patches: list[PathPatch] = []
        self._led_patch_idx = self.LED_FACE_IDX  # which PathPatch is the LED

        # Build one PathPatch per face
        for face_idx, face in enumerate(patch_data.faces):
            verts = patch_data.vertices[face]          # (n_verts, 2) in body frame
            # Transform to world frame
            world_verts = self._transform(verts, pose)

            codes = (
                [Path.MOVETO]
                + [Path.LINETO] * (len(world_verts) - 1)
                + [Path.CLOSEPOLY]
            )
            path_verts = np.vstack([world_verts, world_verts[0]])  # close the path
            path = Path(path_verts, codes)

            color = patch_data.colors[face_idx]
            patch = PathPatch(path, facecolor=color, edgecolor="none", zorder=2)
            axes.add_patch(patch)
            self._patches.append(patch)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_pose(self, pose: NDArray[np.floating]) -> None:
        """Update all patches to reflect the new ``[x, y, theta]`` pose."""
        for face_idx, (face, patch) in enumerate(
            zip(self._data.faces, self._patches)
        ):
            verts = self._data.vertices[face]
            world_verts = self._transform(verts, pose)

            codes = (
                [Path.MOVETO]
                + [Path.LINETO] * (len(world_verts) - 1)
                + [Path.CLOSEPOLY]
            )
            path_verts = np.vstack([world_verts, world_verts[0]])
            self._patches[face_idx].set_path(Path(path_verts, codes))

    def set_led(self, rgb: NDArray[np.floating]) -> None:
        """
        Set the LED colour.

        Parameters
        ----------
        rgb : array-like, shape (3,)
            Red, green, blue components in ``[0, 255]``.
        """
        color = np.asarray(rgb, dtype=float) / 255.0
        self._patches[self.LED_FACE_IDX].set_facecolor(color)

    def set_zorder(self, zorder: int) -> None:
        """Bring all patches to ``zorder``."""
        for patch in self._patches:
            patch.set_zorder(zorder)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _transform(
        body_verts: NDArray[np.floating],
        pose: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """
        Apply a 2-D rigid-body transform to ``body_verts``.

        Rotates by ``theta - pi/2`` so that the robot's +x axis (forward)
        maps to the display's "up" direction when ``theta = 0``.

        Parameters
        ----------
        body_verts : (N, 2) array
            Vertices in the robot body frame.
        pose : (3,) array
            ``[x, y, theta]`` world pose.

        Returns
        -------
        world_verts : (N, 2) array
        """
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
        th = theta - np.pi / 2

        cos_th, sin_th = np.cos(th), np.sin(th)
        R = np.array([[cos_th, -sin_th],
                      [sin_th,  cos_th]])

        # Homogeneous: (N, 2) @ (2, 2)^T + [x, y]
        return body_verts @ R.T + np.array([x, y])