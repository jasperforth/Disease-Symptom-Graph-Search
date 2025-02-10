"""
./backend/tests/test_scoring.py

Contains tests for the scoring function in backend/app/services/scoring.py.

    mode : str
        "linear", "multiplicative_simple", "nonlinear_exponential", or "nonlinear_sigmoid"
"""

from backend.app.services.scoring import calculate_score
from backend.app.core.config import settings

def test_calculate_score_invalid_mode():
    try:
        calculate_score(
            coocc_norm=0.2,
            tfidf_norm=0.5,
            ratio_norm=0.3,
            pmi_norm=0.4,
            npmi_norm=0.6,
            llr_norm=0.0,
            base_weight=0.4,
            local_weight=0.2,
            global_weight=0.4,
            param_coocc=0.4,
            param_pmi=0.0,
            param_npmi=1.0,
            param_llr=0.0,
            mode="unknown_mode"
        )
        print("test_calculate_score_invalid_mode: failed")
        assert False, "Expected ValueError for unknown mode."
    except ValueError as ve:
        print(f"test_calculate_score_invalid_mode: {ve}")
        assert str(ve) == "[calculate_score] Unknown mode: unknown_mode"


def test_calculate_score_linear():
    score = calculate_score(
        coocc_norm=0.2,
        tfidf_norm=0.5,
        ratio_norm=0.3,
        pmi_norm=0.4,
        npmi_norm=0.6,
        llr_norm=0.0,
        base_weight=0.4,
        local_weight=0.2,
        global_weight=0.4,
        param_coocc=0.4,
        param_pmi=0.0,
        param_npmi=1.0,
        param_llr=0.0,
        mode="linear"
    )
    print(f"test_calculate_score_linear: {score}")
    assert 0.0 <= score <= 1.0, "Score should be within [0,1] for linear mode."


def test_calculate_score_multiplicative_simple():
    score = calculate_score(
        coocc_norm=0.2,
        tfidf_norm=0.5,
        ratio_norm=0.3,
        pmi_norm=0.4,
        npmi_norm=0.6,
        llr_norm=0.0,
        base_weight=0.4,
        local_weight=0.2,
        global_weight=0.4,
        param_coocc=0.4,
        param_pmi=0.0,
        param_npmi=1.0,
        param_llr=0.0,
        mode="multiplicative_simple"
    )
    print(f"test_calculate_score_multiplicative_simple: {score}")
    assert 0.0 <= score <= 1.0, "Score should be within [0,1] for multiplicative_simple mode."


def test_calculate_score_nonlinear_exponential():
    score = calculate_score(
        coocc_norm=0.2,
        tfidf_norm=0.5,
        ratio_norm=0.3,
        pmi_norm=0.4,
        npmi_norm=0.6,
        llr_norm=0.0,
        base_weight=0.4,
        local_weight=0.2,
        global_weight=0.4,
        param_coocc=0.4,
        param_pmi=0.0,
        param_npmi=1.0,
        param_llr=0.0,
        mode="nonlinear_exponential"
    )
    print(f"test_calculate_score_nonlinear_exponential: {score}")
    assert 0.0 <= score <= 1.0, "Score should be within [0,1] for nonlinear_exponential mode."


def test_calculate_score_nonlinear_sigmoid():
    score = calculate_score(
        coocc_norm=0.2,
        tfidf_norm=0.5,
        ratio_norm=0.3,
        pmi_norm=0.4,
        npmi_norm=0.6,
        llr_norm=0.0,
        base_weight=0.4,
        local_weight=0.2,
        global_weight=0.4,
        param_coocc=0.4,
        param_pmi=0.0,
        param_npmi=1.0,
        param_llr=0.0,
        mode="nonlinear_sigmoid"
    )
    print(f"test_calculate_score_nonlinear_sigmoid: {score}")
    assert 0.0 <= score <= 1.0, "Score should be within [0,1] for nonlinear_sigmoid mode."


if __name__ == "__main__":
    test_calculate_score_invalid_mode()
    test_calculate_score_linear()
    test_calculate_score_multiplicative_simple()
    test_calculate_score_nonlinear_exponential()
    test_calculate_score_nonlinear_sigmoid()
    