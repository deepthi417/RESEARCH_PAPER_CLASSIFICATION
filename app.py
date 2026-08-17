"""
Research Project Intelligence — Streamlit application.

This file is presentation only. All classification, taxonomy lookup,
keyword extraction, problem detection, recommendation, project
generation, and similarity search logic lives in src/ and is imported,
not reimplemented, here.
"""

import streamlit as st

from src.classifier import load_classifier, classify, ClassifierLoadError
from src.taxonomy import get_domain_info
from src.keyword_extractor import load_keyword_model, extract_keywords
from src.problem_detector import detect_problem_and_data_type
from src.recommender import full_recommendation
from src.project_generator import generate_project
from src.similarity_search import SimilaritySearchService


st.set_page_config(page_title="Research Paper Classification", layout="wide")


# =========================================================
# CACHED RESOURCE LOADERS — each loads exactly once per session
# =========================================================

@st.cache_resource
def get_classifier():
    return load_classifier()


@st.cache_resource
def get_keyword_model():
    return load_keyword_model()


@st.cache_resource
def get_similarity_service():
    service = SimilaritySearchService()
    service.load()
    return service


# =========================================================
# PIPELINE ORCHESTRATION
# Calls into src/ modules in the fixed order from the design.
# No pipeline logic is implemented here — only sequencing and
# error containment, so one failing stage doesn't take down the rest.
# =========================================================

def run_full_pipeline(title, abstract):
    result = {
        'title': title,
        'abstract': abstract,
        'domain': None,
        'keywords': None,
        'problem_type': None,
        'data_type': None,
        'phase5': None,
        'project': None,
        'similarity': None,
        'errors': [],
    }

    # --- Classification + taxonomy ---
    try:
        vectorizer, clf = get_classifier()
        classification = classify(title, abstract, vectorizer, clf)
        broad_domain, subdomain = get_domain_info(classification['primary_category'])
        result['domain'] = {
            **classification,
            'broad_domain': broad_domain,
            'subdomain': subdomain,
        }
    except ClassifierLoadError as e:
        result['errors'].append(f"Classifier unavailable: {e}")
        return result
    except Exception as e:
        result['errors'].append(f"Classification failed unexpectedly: {e}")
        return result

    # --- Keywords ---
    try:
        kw_model = get_keyword_model()
        result['keywords'] = extract_keywords(kw_model, title, abstract)
    except Exception as e:
        result['errors'].append(f"Keyword extraction unavailable: {e}")
        result['keywords'] = []

    # --- Problem/data type ---
    try:
        problem_type, data_type = detect_problem_and_data_type(title, abstract)
        result['problem_type'] = problem_type
        result['data_type'] = data_type
    except Exception as e:
        result['errors'].append(f"Problem/data-type detection failed: {e}")
        result['problem_type'] = 'Unclear / Not specified'
        result['data_type'] = 'Tabular'

    # --- Recommendations (Phase 5) ---
    try:
        result['phase5'] = full_recommendation(
            result['problem_type'], result['data_type'],
            result['domain']['broad_domain'], result['domain']['subdomain']
        )
    except Exception as e:
        result['errors'].append(f"Recommendation engine failed: {e}")

    # --- Suggested project (Phase 6) ---
    if result['phase5'] is not None:
        try:
            result['project'] = generate_project(
                title, abstract,
                result['domain']['broad_domain'], result['domain']['subdomain'],
                result['problem_type'], result['data_type'],
                result['keywords'] or [], result['phase5']
            )
        except Exception as e:
            result['errors'].append(f"Project generation failed: {e}")

    # --- Similarity search (Phase 7) — independent path, own failure mode ---
    try:
        sim_service = get_similarity_service()
        result['similarity'] = sim_service.search(title, abstract, k=5)
    except Exception as e:
        result['similarity'] = {
            'similar_papers': [], 'status': 'unavailable',
            'error_message': str(e), 'query_excluded_self_match': False
        }

    return result


# =========================================================
# UI
# =========================================================

st.title("Research Paper Classification")
st.caption("Turn research papers into actionable project ideas.")

if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

with st.form("analyze_form"):
    title_input = st.text_input("Research Paper Title")
    abstract_input = st.text_area("Research Paper Abstract", height=180)
    submitted = st.form_submit_button("Analyze Research")

if submitted:
    if not title_input.strip() or not abstract_input.strip():
        st.error("Please enter both a title and an abstract.")
    else:
        word_count = len(abstract_input.split())
        if word_count < 15:
            st.warning(
                f"The abstract is quite short ({word_count} words) — results may be "
                f"less reliable, but analysis will proceed."
            )
        with st.spinner("Analyzing..."):
            st.session_state.analysis_result = run_full_pipeline(title_input, abstract_input)

result = st.session_state.analysis_result

if result is not None:
    if result['errors'] and result['domain'] is None:
        st.error("Analysis could not be completed:")
        for err in result['errors']:
            st.error(err)
    else:
        if result['errors']:
            with st.expander("Some parts of the analysis had issues"):
                for err in result['errors']:
                    st.warning(err)

        tabs = st.tabs([
            "Overview", "Domain & Classification", "Keywords", "Recommendations",
            "Suggested Project", "Resources", "Similar Papers", "Evaluation & Difficulty"
        ])

        # --- Overview ---
        with tabs[0]:
            st.subheader(result['title'])
            col1, col2, col3 = st.columns(3)
            col1.metric("Domain", result['domain']['broad_domain'])
            col2.metric("Subdomain", result['domain']['subdomain'])
            col3.metric("Confidence", f"{result['domain']['confidence']*100:.1f}%")

            col4, col5 = st.columns(2)
            col4.metric("Problem Type", result['problem_type'])
            col5.metric("Data Type", result['data_type'])

            if result['project']:
                st.markdown("---")
                st.markdown(f"**Suggested Project:** {result['project']['project_title']}")
                st.markdown(f"**Difficulty:** {result['project']['difficulty']}")
                if result['project'].get('generation_warnings'):
                    for w in result['project']['generation_warnings']:
                        st.info(w)

        # --- Domain & Classification ---
        with tabs[1]:
            st.write(f"**Broad Domain:** {result['domain']['broad_domain']}")
            st.write(f"**Subdomain:** {result['domain']['subdomain']}")
            st.write(f"**Primary Category (arXiv):** {result['domain']['primary_category']}")
            st.write(f"**Confidence:** {result['domain']['confidence']*100:.1f}%")
            if result['domain'].get('runner_up'):
                ru = result['domain']['runner_up']
                st.caption(f"Runner-up: {ru['category']} ({ru['confidence']*100:.1f}%)")

        # --- Keywords ---
        with tabs[2]:
            if result['keywords']:
                for kw in result['keywords']:
                    st.markdown(f"- {kw}")
            else:
                st.info("No keywords could be extracted for this input.")

        # --- Recommendations ---
        with tabs[3]:
            if result['phase5']:
                if result['phase5'].get('kb_coverage_warning'):
                    st.warning(result['phase5']['kb_coverage_warning'])

                st.markdown("### Recommended Technologies")
                st.table(result['phase5']['technologies'])

                st.markdown("### Recommended Algorithms")
                st.table(result['phase5']['algorithms'])
            else:
                st.info("Recommendations are unavailable for this analysis.")

        # --- Suggested Project ---
        with tabs[4]:
            p = result['project']
            if p:
                st.markdown(f"## {p['project_title']}")
                st.markdown(f"**Problem Statement:** {p['problem_statement']}")
                st.markdown(f"**Objective:** {p['objective']}")
                st.markdown(f"**Proposed Solution:** {p['proposed_solution']}")
                st.markdown(f"**Input:** {p['input']}")
                st.markdown(f"**Expected Output:** {p['expected_output']}")
                st.markdown(f"**Recommended Architecture:** {p['recommended_architecture']}")
                st.markdown(f"**Difficulty:** {p['difficulty']}")
                st.markdown("**Estimated Development Complexity:**")
                for r in p['estimated_development_complexity']:
                    st.markdown(f"- {r}")
                st.markdown("**Key Features:**")
                for f in p['key_features']:
                    st.markdown(f"- {f}")
                st.markdown("**Future Enhancements:**")
                for f in p['future_enhancements']:
                    st.markdown(f"- {f}")
                st.markdown(f"**Why This Project Is Suitable:** {p['why_suitable']}")
                st.markdown(f"**How This Differs From the Source Research:** {p['differs_from_source']}")
            else:
                st.info("Suggested project is unavailable for this analysis.")

        # --- Resources ---
        with tabs[5]:
            if result['phase5']:
                res = result['phase5']['resources']
                st.markdown("### Datasets")
                for d in res['datasets']:
                    st.markdown(f"- {d}")
                st.markdown("### Software")
                for s in res['software']:
                    st.markdown(f"- {s}")
                st.markdown("### Hardware")
                st.json(res['hardware'])
            else:
                st.info("Resource recommendations are unavailable for this analysis.")

        # --- Similar Papers ---
        with tabs[6]:
            sim = result['similarity']
            if sim and sim['status'] == 'ok':
                if sim['similar_papers']:
                    st.table(sim['similar_papers'])
                else:
                    st.info("No similar papers found.")
            else:
                st.warning(
                    "Similarity search is currently unavailable"
                    + (f": {sim['error_message']}" if sim and sim.get('error_message') else ".")
                )

        # --- Evaluation & Difficulty ---
        with tabs[7]:
            if result['phase5']:
                st.markdown("### Evaluation Metrics")
                st.table(result['phase5']['metrics'])
                st.markdown("### Difficulty")
                st.write(result['phase5']['difficulty']['difficulty'])
                for r in result['phase5']['difficulty']['reasons']:
                    st.markdown(f"- {r}")
            else:
                st.info("Evaluation and difficulty details are unavailable for this analysis.")
else:
    st.info("Enter a title and abstract above, then click Analyze Research.")
