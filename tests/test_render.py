import numpy as np

from camera_motion import dataset_generator as D
from camera_motion import geometry as G
from camera_motion import render as Rd


def test_lookat_camera_projects_subject_to_image_centre():
    ds = D.generate_e1_audit("E1a", n_cond=1, group=8)
    for R, p, s in zip(ds["R"], ds["p"], ds["subject"]):
        uv, z = Rd.project(s[None], R[5], p[5], f=400.0, cx=240, cy=135)
        np.testing.assert_allclose(uv[0], [240, 135], atol=1e-6)
        assert z[0] > 0  # subject in front of the camera


def test_world_up_points_up_in_image():
    ds = D.generate_e1_audit("E1a", n_cond=1, group=4)
    R, p, s = ds["R"][0][0], ds["p"][0][0], ds["subject"][0]
    uv, _ = Rd.project(np.stack([s, s + [0, 0, 0.5]]), R, p, 400.0, 240, 135)
    assert uv[1, 1] < uv[0, 1]  # higher in the world -> smaller v (image y points down)


def test_near_plane_clipping():
    poly = np.array([[0, 0, -1.0], [1, 0, 2.0], [0, 1, 2.0]])
    out = Rd._clip_near(poly, 0.1)
    assert np.all(out[:, 2] >= 0.1 - 1e-12) and len(out) == 4


def test_render_view_returns_finite_image():
    ds = D.generate_e1_audit("E1b", n_cond=1, group=2)
    faces = Rd.build_scene(ds["subject"][0])
    img = np.asarray(Rd.render_view(faces, ds["R"][0][15], ds["p"][0][15], size=(160, 90)))
    assert img.shape == (90, 160, 3) and img.dtype == np.uint8 and img.std() > 5


def test_upsampling_reproduces_keyframes_and_valid_rotations():
    ds = D.generate_e1_audit("E1b", n_cond=1, group=1)
    R, p = ds["R"][0], ds["p"][0]
    Ru, pu = Rd.upsample_poses(R, p, factor=3)
    assert len(pu) == 3 * (len(p) - 1) + 1
    np.testing.assert_allclose(pu[::3], p, atol=1e-9)
    np.testing.assert_allclose(Ru[::3], R, atol=1e-9)
    assert G.is_valid_rotation(Ru, 1e-9)
