"""Multilingual evaluation dataset and benchmark harness for CiteCat.

Provides bilingual (French and English) test records and evaluated queries
covering exact matches, semantic paraphrases, cross-lingual retrieval (FR <-> EN),
accented queries, stop words, distractors, and unanswerable queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from citecat.models import ExtractedPage, PDFExtractionResult, Record


@dataclass
class EvalQuestion:
    """A single evaluation question."""

    question: str
    answerable: bool
    category: str
    language: str
    expected_record_ids: list[str] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Standard Benchmark Records (Bilingual English & French Corpus)
# ---------------------------------------------------------------------------
BENCHMARK_RECORDS: list[Record] = [
    # 1. English - Climate & Terrestrial Biodiversity
    Record(
        id="hal-eval-en-climate",
        title="Climate Change Impacts on Terrestrial Biodiversity",
        snippet=(
            "Global climate change has significant adverse effects on terrestrial "
            "biodiversity, leading to habitat loss, shifts in species distribution, "
            "and increased vulnerability of endangered ecosystems worldwide."
        ),
        source_url="https://hal.science/hal-eval-en-climate",
        license="https://creativecommons.org/licenses/by/4.0/",
        language="en",
        publication_date="2023",
        retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
    ),
    # 2. French - Climate & Marine Biodiversity
    Record(
        id="hal-eval-fr-climate",
        title=(
            "Impact du réchauffement climatique et de l'acidification des océans "
            "sur la biodiversité marine"
        ),
        snippet=(
            "Le réchauffement global et l'acidification des océans altèrent profondément "
            "les écosystèmes marins, provoquant le blanchissement corallien, la perte de "
            "biodiversité marine et la dégradation irréversible des chaînes trophiques aquatiques."
        ),
        source_url="https://hal.science/hal-eval-fr-climate",
        license="https://creativecommons.org/licenses/by/4.0/",
        language="fr",
        publication_date="2023",
        retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
    ),
    # 3. English - Deep Learning in NLP
    Record(
        id="hal-eval-en-nlp",
        title="Advances in Deep Learning for Natural Language Processing",
        snippet=(
            "Modern machine learning models, specifically transformer architectures, "
            "substantially improve natural language processing capabilities "
            "across translation, sentiment analysis, and question answering benchmarks."
        ),
        source_url="https://hal.science/hal-eval-en-nlp",
        license="https://creativecommons.org/licenses/by/4.0/",
        language="en",
        publication_date="2024",
        retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
    ),
    # 4. French - Deep Learning in NLP
    Record(
        id="hal-eval-fr-nlp",
        title=(
            "Mécanismes d'apprentissage profond et architectures de transformeurs "
            "pour le traitement automatique des langues"
        ),
        snippet=(
            "Les méthodes d'apprentissage automatique profond et les modèles de fondation "
            "permettent des avancées remarquables en linguistique informatique, facilitant "
            "la traduction neuronale et la compréhension du langage naturel par les machines."
        ),
        source_url="https://hal.science/hal-eval-fr-nlp",
        license="https://creativecommons.org/licenses/by/4.0/",
        language="fr",
        publication_date="2024",
        retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
    ),
    # 5. English - Protein Structure Prediction
    Record(
        id="hal-eval-en-protein",
        title="Computational Methods for Protein Structure Prediction",
        snippet=(
            "Computational methods for protein structure prediction combine deep neural "
            "networks, evolutionary co-variance analysis, and physical energy minimization "
            "to determine 3D structures from primary amino acid sequences."
        ),
        source_url="https://hal.science/hal-eval-en-protein",
        license="https://creativecommons.org/licenses/by/4.0/",
        language="en",
        publication_date="2023",
        retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
    ),
    # 6. French - Protein Structure Prediction
    Record(
        id="hal-eval-fr-protein",
        title=(
            "Méthodes informatiques et algorithmes pour la prédiction de la "
            "structure tridimensionnelle des protéines"
        ),
        snippet=(
            "La modélisation bioinformatique et l'apprentissage supervisé permettent de prédire "
            "la structure tridimensionnelle des macromolécules biologiques à partir de leurs "
            "séquences d'acides aminés avec une précision atomique."
        ),
        source_url="https://hal.science/hal-eval-fr-protein",
        license="https://creativecommons.org/licenses/by/4.0/",
        language="fr",
        publication_date="2023",
        retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
    ),
    # 7. English - Quantum Cryptography (Key Distribution)
    Record(
        id="hal-eval-en-quantum",
        title="Quantum Cryptography and Entangled Photon Key Distribution",
        snippet=(
            "Quantum key distribution utilizes quantum superposition and photon entanglement "
            "to establish unconditionally secure cryptographic communication channels, "
            "detecting eavesdropping via fundamental principles of quantum mechanics."
        ),
        source_url="https://hal.science/hal-eval-en-quantum",
        license="https://creativecommons.org/licenses/by/4.0/",
        language="en",
        publication_date="2024",
        retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
    ),
    # 8. English - Distractor (Quantum Chemistry Simulation, not Cryptography)
    Record(
        id="hal-eval-en-distractor",
        title="Quantum Chemical Simulations of Complex Molecular Systems",
        snippet=(
            "Density functional theory and quantum chemical simulations model the electronic "
            "structures and reaction kinetics of transition metal catalysts without empirical "
            "parameters or cryptographic security protocols."
        ),
        source_url="https://hal.science/hal-eval-en-distractor",
        license="https://creativecommons.org/licenses/by/4.0/",
        language="en",
        publication_date="2024",
        retrieved_at=datetime(2024, 1, 1, tzinfo=UTC),
    ),
]


# ---------------------------------------------------------------------------
# Evaluation Queries
# ---------------------------------------------------------------------------
EVAL_DATASET: list[EvalQuestion] = [
    # 1. Exact match (EN)
    EvalQuestion(
        question="What are the effects of climate change on terrestrial biodiversity?",
        answerable=True,
        category="exact_match",
        language="en",
        expected_record_ids=["hal-eval-en-climate"],
        notes="High lexical overlap with English climate record.",
    ),
    # 2. Exact match (FR)
    EvalQuestion(
        question="Quel est l'impact de l'acidification des océans sur la biodiversité marine ?",
        answerable=True,
        category="exact_match",
        language="fr",
        expected_record_ids=["hal-eval-fr-climate"],
        notes="High lexical overlap with French marine record.",
    ),
    # 3. Semantic paraphrase (EN)
    EvalQuestion(
        question="How do automated algorithms comprehend human language using neural networks?",
        answerable=True,
        category="semantic_paraphrase",
        language="en",
        expected_record_ids=["hal-eval-en-nlp"],
        notes="Paraphrase: 'comprehend human language' vs 'natural language processing'.",
    ),
    # 4. Semantic paraphrase (FR)
    EvalQuestion(
        question=(
            "De quelle manière les réseaux neuronaux modélisent-ils la linguistique informatique ?"
        ),
        answerable=True,
        category="semantic_paraphrase",
        language="fr",
        expected_record_ids=["hal-eval-fr-nlp"],
        notes="Paraphrase in French for NLP models.",
    ),
    # 5. Cross-lingual FR -> EN (French query, English evidence)
    EvalQuestion(
        question=(
            "Quelles sont les méthodes informatiques pour prédire la "
            "structure tridimensionnelle des protéines ?"
        ),
        answerable=True,
        category="cross_lingual_fr_to_en",
        language="fr",
        expected_record_ids=["hal-eval-en-protein", "hal-eval-fr-protein"],
        notes="French query matching English and French protein papers.",
    ),
    # 6. Cross-lingual EN -> FR (English query, French evidence)
    EvalQuestion(
        question="How does global warming affect ocean coral bleaching and marine biodiversity?",
        answerable=True,
        category="cross_lingual_en_to_fr",
        language="en",
        expected_record_ids=["hal-eval-fr-climate", "hal-eval-en-climate"],
        notes="English query matching French marine paper ('blanchissement corallien').",
    ),
    # 7. Accented & punctuation query (FR)
    EvalQuestion(
        question="Mécanismes d'apprentissage profond : comment modélise-t-on le langage ?",
        answerable=True,
        category="accented_and_punct",
        language="fr",
        expected_record_ids=["hal-eval-fr-nlp"],
        notes="Tests diacritics (é, è) and colon/apostrophe tokenization.",
    ),
    # 8. Multiple relevant records (EN + FR)
    EvalQuestion(
        question="How does global climate change harm ecosystems and biodiversity?",
        answerable=True,
        category="multi_record",
        language="en",
        expected_record_ids=["hal-eval-en-climate", "hal-eval-fr-climate"],
        notes="Both English terrestrial and French marine records are relevant.",
    ),
    # 9. Stop-word rich query
    EvalQuestion(
        question="What is the way to do this in protein structure prediction?",
        answerable=True,
        category="stop_words",
        language="en",
        expected_record_ids=["hal-eval-en-protein"],
        notes="Query dominated by interrogatives and function words.",
    ),
    # 10. Distractor query (Quantum cryptography vs Quantum chemistry)
    EvalQuestion(
        question="How is quantum cryptography implemented using entangled photon states?",
        answerable=True,
        category="distractor",
        language="en",
        expected_record_ids=["hal-eval-en-quantum"],
        notes="Must retrieve quantum cryptography, not the chemistry distractor.",
    ),
    # 11. Unanswerable (Cooking recipe)
    EvalQuestion(
        question="What is the recipe for the best chocolate cake?",
        answerable=False,
        category="unanswerable",
        language="en",
        expected_record_ids=[],
        notes="Non-scientific cooking query; must abstain.",
    ),
    # 12. Unanswerable (Sports future prediction)
    EvalQuestion(
        question="Who will win the 2030 World Cup?",
        answerable=False,
        category="unanswerable",
        language="en",
        expected_record_ids=[],
        notes="Future speculative query; must abstain.",
    ),
    # 13. Unanswerable (Personal information)
    EvalQuestion(
        question="What is the phone number of the HAL administrator?",
        answerable=False,
        category="unanswerable",
        language="en",
        expected_record_ids=[],
        notes="Personal data; must abstain.",
    ),
    # 14. Unanswerable (French daily life)
    EvalQuestion(
        question="Où puis-je acheter les meilleurs billets de train pour Paris en réduction ?",
        answerable=False,
        category="unanswerable",
        language="fr",
        expected_record_ids=[],
        notes="Daily life French query with no scientific coverage; must abstain.",
    ),
]


def populate_benchmark_corpus(store) -> int:
    """Populate an IngestStore with the standard benchmark records."""
    count = 0
    for record in BENCHMARK_RECORDS:
        store.ingest_record(record)
        count += 1
    return count


def evaluate(
    store,
    retriever=None,
    min_score: float = 0.35,
    k: int = 5,
) -> dict:
    """Run the evaluation benchmark against the store.

    Returns comprehensive metrics including:
    - Answerable accuracy & abstention rate
    - False positive rate (hallucinations on unanswerable queries)
    - Recall@k on expected records
    - Mean Reciprocal Rank (MRR)
    """
    from citecat.citations import build_answer_with_citations

    total = len(EVAL_DATASET)
    answerable_count = sum(1 for q in EVAL_DATASET if q.answerable)
    unanswerable_count = total - answerable_count

    correct_answers = 0
    correct_abstentions = 0
    incorrect_answers = 0  # False positive: answered when unanswerable
    incorrect_abstentions = 0  # False negative: abstained when answerable

    reciprocal_ranks: list[float] = []
    recalls_at_k: list[float] = []
    details: list[dict] = []

    for eq in EVAL_DATASET:
        ans_res = build_answer_with_citations(
            eq.question, store, retriever=retriever, k=k, min_score=min_score
        )
        has_answer = ans_res.answer is not None

        # Check citation record IDs retrieved
        retrieved_ids = [c.record_id for c in ans_res.citations]

        # Calculate rank & recall for answerable questions
        if eq.answerable:
            expected = set(eq.expected_record_ids)
            matched = expected.intersection(retrieved_ids)
            recall = len(matched) / len(expected) if expected else 0.0
            recalls_at_k.append(recall)

            rr = 0.0
            for rank_idx, r_id in enumerate(retrieved_ids, start=1):
                if r_id in expected:
                    rr = 1.0 / rank_idx
                    break
            reciprocal_ranks.append(rr)
        else:
            recall = 1.0 if not has_answer else 0.0

        if eq.answerable and has_answer:
            status = "correct_answer"
            correct_answers += 1
        elif not eq.answerable and not has_answer:
            status = "correct_abstention"
            correct_abstentions += 1
        elif eq.answerable and not has_answer:
            status = "incorrect_abstention"
            incorrect_abstentions += 1
        else:
            status = "incorrect_answer"
            incorrect_answers += 1

        details.append(
            {
                "question": eq.question,
                "category": eq.category,
                "language": eq.language,
                "answerable": eq.answerable,
                "got_answer": has_answer,
                "status": status,
                "retrieved_ids": retrieved_ids,
                "expected_ids": eq.expected_record_ids,
            }
        )

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    mean_recall = sum(recalls_at_k) / len(recalls_at_k) if recalls_at_k else 0.0
    ans_accuracy = correct_answers / answerable_count if answerable_count else 0.0
    abst_rate = correct_abstentions / unanswerable_count if unanswerable_count else 0.0
    false_positive_rate = incorrect_answers / unanswerable_count if unanswerable_count else 0.0

    return {
        "total": total,
        "answerable": answerable_count,
        "unanswerable": unanswerable_count,
        "correct_answer": correct_answers,
        "correct_abstention": correct_abstentions,
        "incorrect_answer": incorrect_answers,
        "incorrect_abstention": incorrect_abstentions,
        "answerable_accuracy": round(ans_accuracy, 4),
        "correct_abstention_rate": round(abst_rate, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "recall_at_k": round(mean_recall, 4),
        "mrr": round(mrr, 4),
        "details": details,
    }


def evaluate_comparison(
    store,
    lexical_retriever,
    hybrid_retriever,
    lexical_min_score: float = 0.5,
    hybrid_min_score: float = 0.35,
) -> dict[str, dict]:
    """Run side-by-side evaluation of BM25 vs Hybrid retrieval on the benchmark."""
    lexical_metrics = evaluate(store, retriever=lexical_retriever, min_score=lexical_min_score)
    hybrid_metrics = evaluate(store, retriever=hybrid_retriever, min_score=hybrid_min_score)
    return {
        "bm25": lexical_metrics,
        "hybrid": hybrid_metrics,
    }


# ---------------------------------------------------------------------------
# Phase 3 Full-Text PDF Benchmark Data & Evaluation
# ---------------------------------------------------------------------------
BENCHMARK_PDF_EXTRACTIONS: dict[str, PDFExtractionResult] = {
    "hal-eval-en-climate": PDFExtractionResult(
        record_id="hal-eval-en-climate",
        pdf_url="https://hal.science/hal-eval-en-climate/file/climate_biodiversity.pdf",
        status="success",
        total_pages=3,
        extracted_pages=3,
        total_words=320,
        ocr_required=False,
        pages=[
            ExtractedPage(
                page_number=1,
                text=(
                    "Climate Change Impacts on Terrestrial Biodiversity: "
                    "A Longitudinal Synthesis.\n"
                    "Abstract: Global climate change has significant adverse effects on "
                    "terrestrial biodiversity, leading to habitat loss, shifts in species "
                    "distribution, and increased vulnerability of endangered ecosystems "
                    "worldwide.\n"
                    "Section 1. Introduction: Terrestrial ecosystems face accelerating "
                    "anthropogenic pressures."
                ),
                word_count=45,
                has_images=False,
            ),
            ExtractedPage(
                page_number=2,
                text=(
                    "Section 3. Longitudinal Field Experiments and Soil Dynamics:\n"
                    "In our longitudinal field experiments across the alpine tundra transect, "
                    "soil microbial respiration increased by 23.4% under a 2°C warming "
                    "treatment, while mycorrhizal fungi biomass declined precipitously by 38.7% "
                    "over a five-year monitoring cycle. Enzymatic assays demonstrated that "
                    "accelerated nutrient cycling depletes organic carbon stocks in subalpine "
                    "soils."
                ),
                word_count=55,
                has_images=False,
            ),
            ExtractedPage(
                page_number=3,
                text=(
                    "Section 5. Ecosystem Modeling and Policy Conclusions:\n"
                    "Dynamic global vegetation models project that without aggressive carbon "
                    "mitigation, up to 42% of alpine endemic plant species will experience "
                    "severe range contraction by 2080. Conservation frameworks must prioritize "
                    "microclimatic refugia and assisted migration corridors."
                ),
                word_count=40,
                has_images=False,
            ),
        ],
        full_text="",
    ),
    "hal-eval-fr-climate": PDFExtractionResult(
        record_id="hal-eval-fr-climate",
        pdf_url="https://hal.science/hal-eval-fr-climate/file/climat_ocean.pdf",
        status="success",
        total_pages=3,
        extracted_pages=3,
        total_words=310,
        ocr_required=False,
        pages=[
            ExtractedPage(
                page_number=1,
                text=(
                    "Impact du réchauffement climatique et de l'acidification des océans "
                    "sur la biodiversité marine.\n"
                    "Résumé : Le réchauffement global et l'acidification des océans altèrent "
                    "profondément les écosystèmes marins, provoquant le blanchissement "
                    "corallien, la perte de biodiversité marine et la dégradation irréversible "
                    "des chaînes trophiques aquatiques."
                ),
                word_count=42,
                has_images=False,
            ),
            ExtractedPage(
                page_number=2,
                text=(
                    "Section 2. Méthodologie d'échantillonnage océanographique :\n"
                    "Les campagnes océanographiques ont déployé des sondes bathymétriques CTD "
                    "jusqu'à six mille mètres de profondeur le long des rifts abyssaux pour "
                    "échantillonner les masses d'eaux profondes."
                ),
                word_count=30,
                has_images=False,
            ),
            ExtractedPage(
                page_number=3,
                text=(
                    "Section 4. Résultats bathymétriques et biogéochimiques :\n"
                    "Les mesures bathymétriques réalisées dans la fosse des Mariannes ont "
                    "révélé une diminution du pH de 0.15 unité et une concentration de "
                    "carbonate de calcium réduite de 18 millimoles par mètre cube. Cette "
                    "sous-saturation en aragonite inhibe sévèrement la biominéralisation "
                    "chez les mollusques ptéropodes."
                ),
                word_count=46,
                has_images=False,
            ),
        ],
        full_text="",
    ),
    "hal-eval-en-nlp": PDFExtractionResult(
        record_id="hal-eval-en-nlp",
        pdf_url="https://hal.science/hal-eval-en-nlp/file/nlp_transformers.pdf",
        status="success",
        total_pages=3,
        extracted_pages=3,
        total_words=330,
        ocr_required=False,
        pages=[
            ExtractedPage(
                page_number=1,
                text=(
                    "Advances in Deep Learning for Natural Language Processing.\n"
                    "Abstract: Modern machine learning models, specifically transformer "
                    "architectures, substantially improve natural language processing "
                    "capabilities across translation, sentiment analysis, and question "
                    "answering benchmarks."
                ),
                word_count=32,
                has_images=False,
            ),
            ExtractedPage(
                page_number=2,
                text=(
                    "Section 3. Attention Head Pruning and Efficiency Ablations:\n"
                    "Pruning 12 out of 16 attention heads in the cross-attention layers "
                    "reduced downstream task latency by 41.2% while perplexity increased "
                    "by merely 0.35 points across language modeling validation splits. We "
                    "observed that redundant heads primarily attend to syntactic delimiter "
                    "tokens rather than semantic content."
                ),
                word_count=50,
                has_images=False,
            ),
            ExtractedPage(
                page_number=3,
                text=(
                    "Section 5. Conclusion: Efficient transformer variants enable on-device "
                    "real-time inference without compromising contextual semantic fidelity."
                ),
                word_count=18,
                has_images=False,
            ),
        ],
        full_text="",
    ),
}

# Questions whose answers require full-text article passages (not found in abstracts)
EVAL_FULLTEXT_DATASET: list[EvalQuestion] = [
    # 1. Full-text specific fact (EN)
    EvalQuestion(
        question=(
            "By what percentage did mycorrhizal fungi biomass decline "
            "in the alpine tundra transect?"
        ),
        answerable=True,
        category="fulltext_specific_fact",
        language="en",
        expected_record_ids=["hal-eval-en-climate"],
        notes="Fact appears exclusively on Page 2 of full-text PDF, not in abstract.",
    ),
    # 2. Full-text specific fact (FR)
    EvalQuestion(
        question=(
            "Quelle diminution du pH a été enregistrée lors des mesures bathymétriques "
            "dans la fosse des Mariannes ?"
        ),
        answerable=True,
        category="fulltext_specific_fact",
        language="fr",
        expected_record_ids=["hal-eval-fr-climate"],
        notes="Fact appears exclusively on Page 3 of full-text PDF, not in abstract.",
    ),
    # 3. Full-text ablation latency metric (EN)
    EvalQuestion(
        question=(
            "By what percentage was downstream task latency reduced when pruning attention heads?"
        ),
        answerable=True,
        category="fulltext_ablation_metric",
        language="en",
        expected_record_ids=["hal-eval-en-nlp"],
        notes="Fact appears on Page 2 of NLP paper, not in abstract.",
    ),
    # 4. Cross-lingual full-text query (FR -> EN)
    EvalQuestion(
        question=(
            "Quel pourcentage d'augmentation de la respiration microbienne du sol "
            "a été mesuré dans la toundra alpine ?"
        ),
        answerable=True,
        category="cross_lingual_fulltext",
        language="fr",
        expected_record_ids=["hal-eval-en-climate"],
        notes="French query targeting English full-text Page 2.",
    ),
    # 5. Unanswerable control (EN)
    EvalQuestion(
        question="What is the exact recipe for chocolate chip cookies in the research laboratory?",
        answerable=False,
        category="unanswerable",
        language="en",
        expected_record_ids=[],
        notes="Must abstain on both abstract and full-text.",
    ),
    # 6. Unanswerable control (FR)
    EvalQuestion(
        question="Quel est le tarif d'un billet de train pour Marseille en période de Noël ?",
        answerable=False,
        category="unanswerable",
        language="fr",
        expected_record_ids=[],
        notes="Must abstain on both abstract and full-text.",
    ),
]


def populate_benchmark_fulltext_corpus(store) -> int:
    """Populate store with full-text PDF extractions for benchmark records."""
    count = 0
    for record in BENCHMARK_RECORDS:
        if record.id in BENCHMARK_PDF_EXTRACTIONS:
            extraction = BENCHMARK_PDF_EXTRACTIONS[record.id]
            store.ingest_pdf(record, extraction)
        else:
            store.ingest_record(record)
        count += 1
    return count


def evaluate_abstract_vs_fulltext(
    abstract_store,
    fulltext_store,
    lexical_retriever,
    hybrid_retriever=None,
    lexical_min_score: float = 0.4,
    hybrid_min_score: float = 0.30,
) -> dict[str, dict]:
    """Compare retrieval performance between abstract-only and full-text stores.

    Evaluates:
    - Abstract-only BM25
    - Full-text BM25
    - Full-text Hybrid (if retriever provided)
    - Citation verification rate for all returned citations against stored source text.
    """
    from citecat.citations import build_answer_with_citations, verify_citation

    def _eval_dataset_on_store(store, retriever, min_score: float) -> dict:
        answerable_count = sum(1 for q in EVAL_FULLTEXT_DATASET if q.answerable)
        unanswerable_count = len(EVAL_FULLTEXT_DATASET) - answerable_count

        correct_answers = 0
        correct_abstentions = 0
        incorrect_answers = 0
        incorrect_abstentions = 0
        reciprocal_ranks: list[float] = []
        recalls: list[float] = []
        verified_citations_count = 0
        total_citations_count = 0

        for eq in EVAL_FULLTEXT_DATASET:
            ans = build_answer_with_citations(
                eq.question, store, retriever=retriever, k=5, min_score=min_score
            )
            has_answer = ans.answer is not None
            retrieved_ids = [c.record_id for c in ans.citations]

            # Verify citations against stored source text
            for cit in ans.citations:
                total_citations_count += 1
                source_text = store.get_record_text(cit.record_id) or ""
                v_res = verify_citation(cit.quote, source_text)
                if v_res.verified:
                    verified_citations_count += 1

            if eq.answerable:
                expected = set(eq.expected_record_ids)
                matched = expected.intersection(retrieved_ids)
                recall = len(matched) / len(expected) if expected else 0.0
                recalls.append(recall)

                rr = 0.0
                for rank_idx, r_id in enumerate(retrieved_ids, start=1):
                    if r_id in expected:
                        rr = 1.0 / rank_idx
                        break
                reciprocal_ranks.append(rr)
            else:
                recall = 1.0 if not has_answer else 0.0

            if eq.answerable and has_answer:
                correct_answers += 1
            elif not eq.answerable and not has_answer:
                correct_abstentions += 1
            elif eq.answerable and not has_answer:
                incorrect_abstentions += 1
            else:
                incorrect_answers += 1

        citation_verification_rate = (
            verified_citations_count / total_citations_count if total_citations_count > 0 else 1.0
        )

        return {
            "total": len(EVAL_FULLTEXT_DATASET),
            "answerable": answerable_count,
            "unanswerable": unanswerable_count,
            "correct_answer": correct_answers,
            "correct_abstention": correct_abstentions,
            "incorrect_abstention": incorrect_abstentions,
            "incorrect_answer": incorrect_answers,
            "answerable_accuracy": round(correct_answers / answerable_count, 4)
            if answerable_count
            else 0.0,
            "correct_abstention_rate": round(correct_abstentions / unanswerable_count, 4)
            if unanswerable_count
            else 0.0,
            "false_positive_rate": round(incorrect_answers / unanswerable_count, 4)
            if unanswerable_count
            else 0.0,
            "recall_at_k": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
            "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)
            if reciprocal_ranks
            else 0.0,
            "citation_verification_rate": round(citation_verification_rate, 4),
        }

    abstract_bm25_metrics = _eval_dataset_on_store(
        abstract_store, lexical_retriever, min_score=lexical_min_score
    )
    fulltext_bm25_metrics = _eval_dataset_on_store(
        fulltext_store, lexical_retriever, min_score=lexical_min_score
    )

    result = {
        "abstract_bm25": abstract_bm25_metrics,
        "fulltext_bm25": fulltext_bm25_metrics,
    }

    if hybrid_retriever is not None:
        fulltext_hybrid_metrics = _eval_dataset_on_store(
            fulltext_store, hybrid_retriever, min_score=hybrid_min_score
        )
        result["fulltext_hybrid"] = fulltext_hybrid_metrics

    return result
