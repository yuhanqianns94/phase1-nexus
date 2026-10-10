from pathlib import Path
from streamlit.testing.v1 import AppTest
from test_prediction_pipeline import _article


def _app(tmp_path, monkeypatch, state=None):
    monkeypatch.setenv("PHASE1_V3_EXTRACTION_ROOTS", str(tmp_path))
    monkeypatch.setenv("PHASE1_V3_STATE_ROOT", str(tmp_path / "web_state"))
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    app = AppTest.from_string("from phase1_web import app as native\nfrom phase1_web.parity_app import main\nmain(native)", default_timeout=30)
    for key, value in (state or {}).items():
        app.session_state[key] = value
    return app.run()


def test_workflow_loads_v3_library_with_v1_navigation_and_no_curve_cache(tmp_path, monkeypatch):
    _article(tmp_path, "PMC123", rows=[(.1, 3, 0), (.3, 6, 2)])
    before = {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()}
    app = _app(tmp_path, monkeypatch)
    assert not app.exception
    assert app.title[0].value == "Literature Library"
    assert [b.label for b in app.button if str(b.key).startswith("nav_")] == [
        "1  ·  Library", "Article comparison", "2  ·  Target trial", "3  ·  Results", "4  ·  OC simulation"]
    after = {p.relative_to(tmp_path) for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    assert not list(tmp_path.rglob("*curve*cache*"))


def test_target_form_matches_v1_defaults_and_structured_fields(tmp_path, monkeypatch):
    _article(tmp_path, "PMC123", rows=[(.1, 3, 0), (.3, 6, 2)])
    app = _app(tmp_path, monkeypatch)
    app.button(key="nav_Target Trial").click().run()
    assert not app.exception
    assert app.number_input(key="similarity_threshold_input").value == .3
    assert app.text_input(key="sim_weight_text_drugs").value == "1/6"
    assert app.text_input(key="sim_weight_text_structure").value == "0"
    assert app.radio(key="target_curve_method").value == "article_line_weighted"
    assert app.radio(key="embedding_backend_select").value == "local"
    assert not any(
        widget.key == "embedding_model_select" for widget in app.selectbox
    )
    assert not any(
        widget.key == "random_intercept_scale_prior_mean"
        for widget in app.number_input
    )
    assert not any(widget.key == "combo_uplift_mean" for widget in app.number_input)
    assert app.number_input(key="opt_age_min").value is None
    assert app.number_input(key="opt_dlt_eva_days").value == 0
    assert app.text_area(key="planned_doses_text").label == "Planned doses (optional)"
    assert any(w.key == "opt_gene_targets" for w in app.text_input)
    assert not any("accelerated" in str(w.label).lower() for w in app.get("number_input"))


def test_target_form_switches_between_tfidf_and_openai_embeddings(
    tmp_path, monkeypatch
):
    _article(tmp_path, "PMC123", rows=[(.1, 3, 0), (.3, 6, 2)])
    app = _app(tmp_path, monkeypatch)
    app.button(key="nav_Target Trial").click().run()

    app.radio(key="embedding_backend_select").set_value("openai").run()
    assert app.selectbox(key="embedding_model_select").value == (
        "text-embedding-3-small"
    )

    app.radio(key="embedding_backend_select").set_value("local").run()
    assert not any(
        widget.key == "embedding_model_select" for widget in app.selectbox
    )


def test_target_form_shows_only_controls_relevant_to_method_and_trial_type(
    tmp_path, monkeypatch
):
    _article(tmp_path, "PMC123", rows=[(.1, 3, 0), (.3, 6, 2)])
    app = _app(tmp_path, monkeypatch)
    app.button(key="nav_Target Trial").click().run()

    app.radio(key="target_curve_method").set_value(
        "power_likelihood_random_intercept_slope"
    ).run()
    assert app.number_input(key="random_intercept_scale_prior_mean").value == -1.1
    assert app.number_input(key="random_slope_scale_prior_mean").value == -1.5
    assert not any(widget.key == "combo_uplift_mean" for widget in app.number_input)

    import json
    app.session_state['native_target_rows_json']=json.dumps([
        {'scenario':'Main','drug':'A','role':'escalated','presence':'present'},
        {'scenario':'Main','drug':'B','role':'combination_partner','presence':'present'}])
    app.run()
    for method in (
        "article_line_weighted",
        "article_posterior_draw_pool",
        "power_likelihood_random_intercept_slope",
    ):
        app.radio(key="target_curve_method").set_value(method).run()
        assert not app.exception
        assert not any(
            widget.key in {"combo_uplift_mean", "combo_uplift_sd", "apply_combo_prior"}
            for widget in [*app.number_input, *app.checkbox]
        )
        assert any("Treatment type: combination" in c.value for c in app.caption)


def test_old_combination_uplift_draft_fields_are_not_restored_or_saved(tmp_path):
    import json
    from datetime import datetime, timedelta, timezone
    from phase1_web.reference.target_trial_draft import (
        TARGET_TRIAL_DRAFT_FORMAT,
        load_target_trial_draft,
        save_target_trial_draft,
        target_trial_default_values,
    )

    now = datetime.now(timezone.utc)
    path = tmp_path / "target_trial_draft.json"
    supported = {"target_drug": "A", "trial_type_control": "combination"}
    retired = {"apply_combo_prior": True, "combo_uplift_mean": .3, "combo_uplift_sd": .2}
    path.write_text(json.dumps({
        "format": TARGET_TRIAL_DRAFT_FORMAT,
        "saved_at_utc": now.isoformat(),
        "expires_at_utc": (now + timedelta(hours=24)).isoformat(),
        "values": {**supported, **retired},
    }))
    draft = load_target_trial_draft(path=path, now=now)
    assert draft is not None and draft.values == supported
    save_target_trial_draft({**supported, **retired}, path=path, now=now)
    assert json.loads(path.read_text())["values"] == supported
    defaults = target_trial_default_values(
        default_prior="weakly_informative", default_weights={}, default_threshold=.3
    )
    assert not retired.keys() & defaults.keys()


def test_web_predicts_three_methods_and_runs_priority_boin_search(tmp_path, monkeypatch):
    _article(tmp_path, "PMC123", rows=[(.1, 3, 0), (.3, 6, 2)])
    app = _app(tmp_path, monkeypatch)
    app.button(key="nav_Target Trial").click().run()
    app.radio(key="target_curve_method").set_value("article_line_weighted")
    app.text_input(key="target_drug").set_value("IPH2101")
    app.text_input(key="target_dose_unit").set_value("mg/kg")
    app.text_area(key="planned_doses_text").set_value(".05, .1, .2")
    next(b for b in app.button if b.label.startswith("Run toxicity prediction")).click().run()
    assert not app.exception and not app.error, [e.value for e in app.error]
    raw = app.session_state["curve_result"]
    assert raw["selected_curve_method"] == "article_line_weighted"
    assert raw["embedding_backend_requested"] == "local"
    assert raw["embedding_backend_used"] == "local_token_cosine"
    assert all(m["status"] == "fitted" for m in raw["toxicity_curve"].values())
    assert len(raw["toxicity_curve"]) == 3
    assert not any(widget.key == "prediction_curve_method" for widget in app.radio)
    assert not list(tmp_path.rglob("*curve*cache*"))
    # A fresh browser view avoids stale, already-unmounted target widgets in
    # AppTest's rerun tree; it retains exactly the application result/context.
    app = _app(tmp_path, monkeypatch, {"ui_page": "OC Simulation", "curve_result": raw,
                                      "v3_reference_context": app.session_state["v3_reference_context"]})
    assert not app.exception and not app.error, [e.value for e in app.error]
    assert not any(
        widget.key == "boin_search_truth_curve_method" for widget in app.selectbox
    )
    assert len([s for s in app.selectbox if str(s.key).startswith("boin_search_priority_")]) == 4
    assert not any("accelerated" in str(w.label).lower() for w in [*app.number_input, *app.multiselect, *app.checkbox])
    app.text_input(key="boin_search_calibration_text").set_value("automatic")
    app.text_input(key="boin_search_elimination_cutoff_text").set_value("0.90, 0.95")
    app.text_input(key="boin_search_stop_rule_text").set_value("off")
    app.number_input(key="boin_search_posterior_curve_count").set_value(4)
    app.number_input(key="boin_search_trials_per_scenario").set_value(2)
    next(b for b in app.button if b.label.startswith("Run BOIN parameter search")).click().run()
    assert not app.exception and not app.error, [e.value for e in app.error]
    result = app.session_state["oc_result"]
    assert result["candidate_count"] == 2
    assert result["ranking_priority"] == ["correct_decision", "overtoxic_selection", "safety_stop", "mean_total_sample_size"]
    assert result["base_config"]["boin_elimination_min_n"] == 1
