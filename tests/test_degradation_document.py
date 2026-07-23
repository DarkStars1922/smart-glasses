from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DOCUMENT = Path("docs/Shouldersurfing Smart Glass .md")
PARAMETERS = Path("analysis/results/v2_parameters.json")
PARAMETERS_V21 = Path("analysis/results/v2_1_parameters.json")
PARAMETERS_V3 = Path("analysis/results/v3_parameters.json")
PARAMETERS_V4 = Path("analysis/results/v4_parameters.json")
PARAMETERS_V5 = Path("analysis/results/v5_parameters.json")
PARAMETERS_V6 = Path("analysis/results/v6_parameters.json")


def test_document_uses_generated_core_fit_values() -> None:
    document = DOCUMENT.read_text(encoding="utf-8")
    parameters = json.loads(PARAMETERS.read_text(encoding="utf-8"))
    grids = parameters["fit"]["geometry_and_spatial_psf"]["point_grids"]
    temporal = parameters["fit"]["temporal_psf"]
    background = parameters["fit"]["background_and_noise"]

    expected = [
        f"{grids['W']['local_scale_normalized_47mm']['median'][0]:.4f}",
        f"{grids['W']['local_scale_normalized_47mm']['median'][1]:.4f}",
        f"{grids['W']['psf_fwhm_normalized_47mm']['median'][0]:.2f}",
        f"{grids['W']['psf_fwhm_normalized_47mm']['median'][1]:.2f}",
        f"{temporal['psf_fwhm_camera_px']['median'][0]:.2f}",
        f"{temporal['psf_fwhm_camera_px']['median'][1]:.2f}",
        f"{background['background_rgb']['median'][0]:.4f}",
    ]
    assert all(value in document for value in expected)


def test_document_uses_followup_fit_values() -> None:
    document = DOCUMENT.read_text(encoding="utf-8")
    parameters = json.loads(PARAMETERS_V21.read_text(encoding="utf-8"))
    grids = parameters["fit"]["large_color_point_grids"]
    response = parameters["fit"]["permuted_photometry"]

    expected = [
        f"{grids['G']['point_contrast']['median']:.3f}",
        f"{grids['B']['point_contrast']['median']:.3f}",
        f"{grids['G']['psf_fwhm_camera_px']['median'][0]:.2f}",
        f"{grids['B']['psf_fwhm_camera_px']['median'][1]:.2f}",
        f"{response['gray_fit_mae']:.3f}",
        f"{response['color_fit_mae']:.3f}",
    ]
    assert all(value in document for value in expected)


def test_document_contains_complete_model_and_identifiability_sections() -> None:
    document = DOCUMENT.read_text(encoding="utf-8")

    assert "I_{blur}" not in document
    assert "ψ\\_blur" not in document
    assert "JPEG 压缩 quality ∈" not in document
    assert "\\mathcal W" in document
    assert "\\mathcal D" in document
    assert "\\mathcal J" in document
    assert "当前可识别参数" in document
    assert "仍不可识别的参数" in document
    assert "v2_1_parameters.json" in document
    assert "二次几何残差场" in document
    assert "15 x 15" in document
    assert "31 x 31" in document
    assert "F11 截掉" in document
    assert "f_c(0)=0" in document
    assert "特征级多帧融合识别模型" in document


def test_core_capture_fit_resolves_the_remaining_photometric_terms() -> None:
    parameters = json.loads(PARAMETERS_V3.read_text(encoding="utf-8"))
    response = parameters["fit"]["joint_photometry"]
    background = parameters["fit"]["temporal_quadratic_background"]
    edges = parameters["fit"]["gb_slanted_edge_psf"]

    assert parameters["dataset"]["frame_count"] == 9
    assert response["gray_design_rank"] == response["gray_design_columns"]
    assert response["tone_monotonic"] == [True, True, True]
    assert all(type(value) is bool for value in response["tone_monotonic"])
    assert background["frame_count"] == 2
    assert edges["accepted_count"] >= 8
    assert set(edges["by_color"]) == {"G", "B"}


def test_document_uses_core_fit_as_the_latest_model() -> None:
    document = DOCUMENT.read_text(encoding="utf-8")
    parameters = json.loads(PARAMETERS_V3.read_text(encoding="utf-8"))
    response = parameters["fit"]["joint_photometry"]
    edges = parameters["fit"]["gb_slanted_edge_psf"]["by_color"]

    expected = [
        "v3_parameters.json",
        "c_core",
        f"{response['gray_fit_mae']:.3f}",
        f"{response['color_fit_mae']:.3f}",
        f"{edges['G']['median']:.2f}",
        f"{edges['B']['median']:.2f}",
    ]
    assert all(value in document for value in expected)


def test_research_outputs_do_not_expose_internal_hashes() -> None:
    parameters = json.loads(PARAMETERS.read_text(encoding="utf-8"))
    report = Path("analysis/results/v2_report.md").read_text(encoding="utf-8")

    assert "qtable_sha256" not in parameters["dataset"]["jpeg"]
    assert "SHA-256" not in report


def test_color_operator_fit_uses_an_independent_holdout_chart() -> None:
    parameters = json.loads(PARAMETERS_V4.read_text(encoding="utf-8"))
    fit = parameters["fit"]
    training = fit["training_node_fit"]
    holdout = fit["holdout_validation"]
    operator = fit["color_operator"]

    assert parameters["dataset"]["frame_count"] == 6
    assert training["frame_count"] == 3
    assert training["design_rank"] == training["design_columns"]
    assert holdout["frame_count"] == 1
    assert holdout["used_for_training"] is False
    assert holdout["validation_color_count"] == 24
    assert operator["lut_shape"] == [3, 3, 3, 3]
    assert operator["selected_model"] == "constrained_linear_matrix"
    assert operator["linear_has_negative_entries"] is False
    assert operator["holdout_jpeg_mae_linear"] < operator["holdout_jpeg_mae_lut"]
    assert fit["anchor_tone_normalization"]["holdout"]["color_patches_used"] is False


def test_document_uses_color_operator_holdout_results() -> None:
    document = DOCUMENT.read_text(encoding="utf-8")
    parameters = json.loads(PARAMETERS_V4.read_text(encoding="utf-8"))
    operator = parameters["fit"]["color_operator"]

    expected = [
        "v4_parameters.json",
        "c_color",
        "h_{d,p,j,t}",
        f"{operator['holdout_jpeg_mae_linear']:.4f}",
        f"{operator['holdout_jpeg_mae_lut']:.4f}",
    ]
    assert all(value in document for value in expected)


def test_cross_pose_fit_uses_four_sessions_and_independent_holdouts() -> None:
    parameters = json.loads(PARAMETERS_V5.read_text(encoding="utf-8"))
    sessions = parameters["fit"]["sessions"]
    generalization = parameters["fit"]["cross_pose_generalization"]

    assert parameters["dataset"]["session_count"] == 4
    assert parameters["dataset"]["frame_count"] == 24
    assert set(sessions) == {"baseline", "repeat", "pose_left", "pose_right"}
    assert sessions["pose_right"]["orientation_quarter_turns_ccw"] == 1
    assert all(
        session["training_node_fit"]["design_rank"]
        == session["training_node_fit"]["design_columns"]
        for session in sessions.values()
    )
    assert all(
        session["holdout_validation"]["used_for_training"] is False
        for session in sessions.values()
    )
    assert len(generalization["leave_one_session_out"]) == 4
    assert np.asarray(generalization["pooled_color_matrix"]).shape == (3, 3)
    assert generalization["status"] in {
        "supported_across_sampled_poses",
        "pose_conditioning_required",
    }


def test_document_uses_cross_pose_generalization_results() -> None:
    document = DOCUMENT.read_text(encoding="utf-8")
    parameters = json.loads(PARAMETERS_V5.read_text(encoding="utf-8"))
    generalization = parameters["fit"]["cross_pose_generalization"]
    pooled = generalization["pooled_matrix_on_all_holdouts"]

    expected = [
        "v5_parameters.json",
        "c_color_pose_left",
        "c_color_pose_right",
        generalization["status"],
        f"{pooled['mean_jpeg_mae']:.4f}",
        f"{generalization['leave_one_session_out_max_jpeg_mae']:.4f}",
    ]
    assert all(value in document for value in expected)


def test_vertical_down_fit_uses_five_sessions_and_independent_holdouts() -> None:
    parameters = json.loads(PARAMETERS_V6.read_text(encoding="utf-8"))
    sessions = parameters["fit"]["sessions"]
    generalization = parameters["fit"]["cross_pose_generalization"]
    down = sessions["pose_down"]

    assert parameters["dataset"]["session_count"] == 5
    assert parameters["dataset"]["frame_count"] == 30
    assert set(sessions) == {
        "baseline",
        "repeat",
        "pose_left",
        "pose_right",
        "pose_down",
    }
    assert down["orientation_quarter_turns_ccw_by_role"] == {
        "black_start": 0,
        "train_A": 1,
        "train_B": 1,
        "train_C": 1,
        "holdout": 1,
        "black_end": 1,
    }
    assert all(
        session["training_node_fit"]["design_rank"]
        == session["training_node_fit"]["design_columns"]
        for session in sessions.values()
    )
    assert all(
        session["holdout_validation"]["used_for_training"] is False
        for session in sessions.values()
    )
    assert len(generalization["leave_one_session_out"]) == 5


def test_document_uses_vertical_down_generalization_results() -> None:
    document = DOCUMENT.read_text(encoding="utf-8")
    parameters = json.loads(PARAMETERS_V6.read_text(encoding="utf-8"))
    generalization = parameters["fit"]["cross_pose_generalization"]
    pooled = generalization["pooled_matrix_on_all_holdouts"]

    expected = [
        "v6_parameters.json",
        "c_color_pose_down",
        generalization["status"],
        f"{pooled['mean_jpeg_mae']:.4f}",
        f"{generalization['leave_one_session_out_max_jpeg_mae']:.4f}",
    ]
    assert all(value in document for value in expected)
