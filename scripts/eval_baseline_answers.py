"""Evaluate Baseline LLM answers against the RAG evaluation metrics.

Reads retrieved chunks from data/eval_retrieval.json, uses pre-generated
Baseline answers, and runs RAGAS + hallucination detection.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate.clinical_qa_set import ClinicalQASet
from evaluate.hallucination_detector import HallucinationDetector
from evaluate.ragas_metrics import RAGASEvaluator

# baseline LLM answers for 12 sampled questions (2 per domain)
BASELINE_ANSWERS = {
    "qa_001": (
        "Massive transfusion protocol (MTP) is activated when a patient requires "
        "more than 10 units of packed red blood cells within 24 hours, which is the "
        "most common definition of massive transfusion [1]. MTP activation is "
        "triggered in the setting of acute hemorrhage requiring rapid administration "
        "of blood products in specific ratios [1]. Adjunctive medications such as "
        "tranexamic acid and calcium may be used during massive transfusion [1]. "
        "The inflammatory response activated by tissue injury in major trauma can "
        "become dysregulated, and cytokines play a key role in this process [3]. "
        "Clinical trials are evaluating different protocols for blood product "
        "transfusion in trauma patients admitted to emergency departments [5]."
    ),
    "qa_002": (
        "Current evidence supports a high ratio of plasma to red blood cells in "
        "trauma resuscitation. Plasma transfusion is recommended as an initial "
        "intervention in most major hemorrhage protocols [1]. The PROPPR trial "
        "(Pragmatic, Randomized Optimal Platelet and Plasma Ratios) was a Phase III "
        "trial designed to evaluate differences in 24-hour and 30-day mortality "
        "between 1:1:1 and 1:1:2 ratios of plasma, platelets, and red blood cells [2]. "
        "In pediatric massive transfusion, maintaining competency in appropriate "
        "dosing is critical given the rarity of protocol activation in pediatric "
        "centers [3]. The dose-equivalent ratio of fibrinogen-to-erythrocytes also "
        "plays a role in outcomes during major bleeding scenarios [4]."
    ),
    "qa_003": (
        "Tranexamic acid (TXA) is an established antifibrinolytic agent in the "
        "treatment of bleeding trauma patients [1]. TXA inhibits plasminogen "
        "activation and plasmin, thereby retarding clot disintegration [3]. Evidence "
        "supports its use across trauma surgical specialties as a well-known "
        "antifibrinolytic agent with increasing evidence for trauma patients [4]. "
        "However, concerns have been raised regarding possible thromboembolic "
        "complications with repeated dosing in multiply injured patients [1]. "
        "Studies examining early prehospital administration of TXA in multiple "
        "trauma patients with severe external and internal bleeding indicate that "
        "uncontrollable blood loss accounts for approximately one third of all "
        "trauma deaths [5]. Clinical trials such as the STAAMP trial are evaluating "
        "whether 1 gram of prehospital TXA given during emergency medical transport "
        "is associated with lower 30-day mortality [2]."
    ),
    "qa_004": (
        "The evidence for prehospital tranexamic acid (TXA) in traumatic brain "
        "injury (TBI) comes from several clinical trials. The STAAMP trial evaluated "
        "whether 1 gram of prehospital TXA during emergency medical transport to a "
        "level 1 trauma center reduces 30-day mortality in patients at risk of "
        "hemorrhage [1]. A Bayesian reanalysis of the STAAMP trial data confirmed "
        "that TXA is associated with improved survival following trauma in prior "
        "prehospital trials, shaping clinical practice [5]. The CRASH-3 trial was "
        "designed to provide reliable evidence about the effect of TXA on mortality "
        "and disability in patients with traumatic brain injury [4]. Additionally, "
        "a clinical trial is evaluating the effect of TXA on mortality specifically "
        "in pediatric patients with TBI, which could potentially lead to improved "
        "treatment outcomes [3]. TXA use in trauma surgical specialties continues "
        "to show increasing evidence supporting its application [2]."
    ),
    "qa_011": (
        "Direct peritoneal resuscitation (DPR) is a technique used as an adjunct "
        "to damage control laparotomy (DCL) in critically ill surgical patients. "
        "In DPR, the open abdomen is continuously irrigated with glucose-based "
        "hypertonic dialysate [1]. This approach has been shown to improve abdominal "
        "closure rates and decrease wound complications [1]. Clinical trials are "
        "investigating whether DPR helps improve blood flow through important organs "
        "after damage control surgery [2]. A narrative review evaluating the efficacy "
        "of adjunct DPR found evidence supporting its role in adult damage control "
        "surgery patients [3]. However, DPR is associated with increased risk of "
        "intra-abdominal fungal infections (AFIs) in critically ill surgical "
        "patients [1]. Excessive perioperative crystalloid use remains a concern in "
        "the context of damage control laparotomy and resuscitation [5]."
    ),
    "qa_015": (
        "Lactate and base deficit serve as important predictors of traumatic "
        "coagulopathy. A study comparing the predictive value of lactate level and "
        "base deficit found both to be independent risk factors for traumatic "
        "coagulopathy, in-hospital mortality, and massive transfusion [1]. Timely "
        "and accurate initial assessment of trauma patients using these biomarkers "
        "can significantly affect future outcomes [1]. Lactate clearance has also "
        "emerged as a potential prognostic marker in critically ill patients, with "
        "evidence suggesting it can predict massive transfusion in upper "
        "gastrointestinal bleeding [5]. In the context of trauma-induced "
        "coagulopathy, goal-directed therapy protocols that incorporate these "
        "markers are being evaluated to address the high mortality rate associated "
        "with hemorrhage-induced coagulopathy [3]. Novel assays such as plasmin "
        "activated clotting time may complement lactate in identifying "
        "hyperfibrinolysis after trauma [4]."
    ),
    "qa_017": (
        "Prehospital intubation in hemorrhagic shock after severe trauma is "
        "associated with important mortality considerations. A propensity-matched "
        "study assessed the association of intubation timing (prehospital vs. "
        "in-hospital) with mortality and morbidity in patients with hemorrhagic "
        "shock following severe trauma [1]. The study analyzed patients who were "
        "intubated and presented with hemorrhagic shock, defined as requiring four "
        "or more packed red blood cells [1]. Hemorrhagic shock is the leading cause "
        "of preventable prehospital trauma deaths, and interventions such as "
        "intramuscular vasopressin are being explored as feasible alternatives [2]. "
        "Research into respiratory support during endotracheal intubation aims to "
        "evaluate the benefits and harms of different approaches [3]. The optimal "
        "sequence of vasopressor administration versus advanced airway management "
        "during resuscitation remains unclear [4]."
    ),
    "qa_018": (
        "The modified ABC-Resuscitation sequence in prehospital trauma represents "
        "a shift from traditional assessment. Traditional trauma assessment follows "
        "the ABC (airway, breathing, circulation) sequence, but evidence suggests "
        "the CAB (circulation, airway, breathing) approach may better maintain "
        "perfusion and prevent hypotension [1]. Damage Control Resuscitation (DCR), "
        "derived from military protocols, focuses on early hemorrhage control and "
        "volume replacement to combat the 'diamond of death' consisting of "
        "hypothermia, hypocalcemia, acidosis, and coagulopathy [1]. At the 2023 "
        "ATLS symposium, the priority of circulation was emphasized through the "
        "'x-airway-breathing-circulation (xABC)' sequence, where 'x' stands for "
        "exsanguinating hemorrhage control [3]. Clinical prediction scores such as "
        "the Assessment of Blood Consumption (ABC) score are being compared for "
        "their ability to accurately identify bleeding trauma patients early [5]."
    ),
    "qa_027": (
        "The FDA has developed a regulatory framework for AI/ML-based Software as "
        "a Medical Device (SaMD) through its AI/ML SaMD Action Plan [1]. The FDA's "
        "vision recognizes that AI/ML technologies have an appropriately tailored "
        "regulatory framework that enables iterative improvement through the "
        "ability to learn from real-world use and experience [2]. Key components "
        "include: (1) a tailored regulatory framework with Predetermined Change "
        "Control Plans (PCCPs) that allow manufacturers to describe planned "
        "modifications to their devices [3][5]; (2) Good Machine Learning Practice "
        "(GMLP) principles developed in collaboration with Health Canada and the "
        "UK MHRA [1]; (3) a transparency initiative for AI/ML-based devices; and "
        "(4) real-world performance monitoring [4]. The FDA plans to engage with "
        "the public to elicit feedback from end users on these efforts [4]."
    ),
    "qa_028": (
        "Predetermined Change Control Plans (PCCPs) are a regulatory mechanism that "
        "allows manufacturers of ML-enabled medical devices to describe planned "
        "modifications and the methodology to implement them [5]. PCCPs support the "
        "iterative development of AI-based device software functions (AI-DSFs) by "
        "enabling model performance improvement through iterative modifications, "
        "including learning from real-world data [5]. Key elements of a PCCP include: "
        "specification of assessment metrics and evaluation methods [3]; reference "
        "standards used during training, tuning, and testing [3]; and Good Machine "
        "Learning Practice (GMLP) principles ensuring ML-enabled devices are safe "
        "and effective over the device lifecycle [1]. Good Software Engineering and "
        "Security Practices should be implemented alongside PCCPs, including "
        "methodical risk management, data quality assurance, and robust cybersecurity "
        "practices [1]. The FDA has published final guidance on PCCPs for ML-enabled "
        "medical devices [4]."
    ),
    "qa_051": (
        "Insufficient evidence in the available sources to answer this question. "
        "The provided sources cover topics related to AI/ML-based medical devices, "
        "clinical decision support software, and trauma resuscitation, but contain "
        "no information about quantum computing applications in cooking."
    ),
    "qa_052": (
        "Insufficient evidence in the available sources to answer this question. "
        "The provided sources are clinical and regulatory documents related to "
        "trauma care, transfusion medicine, and medical device regulation. They "
        "contain no information about cryptocurrency investments."
    ),
}


def main():
    print("=" * 60)
    print("  BASELINE EVALUATION (in-conversation answers)")
    print("=" * 60)

    # Load retrieval data for chunk texts
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "eval_retrieval.json")) as f:
        retrieval_data = json.load(f)

    pairs_by_id = {p["id"]: p for p in retrieval_data["pairs"]}

    # Load Q&A set for expected answers
    qa_set = ClinicalQASet(
        os.path.join(os.path.dirname(__file__), "..", "data", "qa_test_set.json")
    )

    ragas = RAGASEvaluator()
    hallucination = HallucinationDetector(threshold=0.7)

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    per_question_results = []

    t_start = time.perf_counter()

    for i, (qa_id, answer_text) in enumerate(BASELINE_ANSWERS.items()):
        pair = pairs_by_id.get(qa_id)
        if not pair:
            print(f"  WARNING: {qa_id} not found in retrieval data")
            continue

        q = pair["question"]
        expected = pair.get("expected_answer_summary", "")
        domain_tag = pair.get("domain", "unknown")
        difficulty = pair.get("difficulty", "medium")
        source_texts = [c["text"] for c in pair.get("retrieved_chunks", [])]

        print(f"  [{i+1}/{len(BASELINE_ANSWERS)}] {qa_id} ({domain_tag}/{difficulty}): {q[:80]}...")

        questions.append(q)
        answers.append(answer_text)
        contexts.append(source_texts)
        ground_truths.append(expected)

        # Hallucination check
        h_result = hallucination.detect_hallucinations(answer_text, source_texts)
        hallucination.log_evaluation(
            answer_text,
            is_hallucinating=h_result["is_hallucinating"],
            details=h_result,
        )

        per_question_results.append({
            "id": qa_id,
            "question": q,
            "domain": domain_tag,
            "difficulty": difficulty,
            "status": "ok",
            "is_hallucinating": h_result["is_hallucinating"],
            "hallucination_rate": h_result["hallucination_rate"],
            "overall_grounding": h_result["overall_grounding"],
            "answer_preview": answer_text[:200],
        })

        print(
            f"    ground={h_result['overall_grounding']:.3f} "
            f"halluc={h_result['is_hallucinating']} "
            f"sent_halluc_rate={h_result['hallucination_rate']:.2%}"
        )

    total_time = time.perf_counter() - t_start

    # RAGAS metrics
    ragas_scores = ragas.evaluate(questions, answers, contexts, ground_truths)
    print(f"\n  RAGAS scores:")
    print(f"    faithfulness:      {ragas_scores['faithfulness']:.4f}")
    print(f"    answer_relevance:  {ragas_scores['answer_relevance']:.4f}")
    print(f"    context_precision: {ragas_scores['context_precision']:.4f}")

    halluc_rate = hallucination.get_hallucination_rate()

    report = {
        "model": "baseline-llm-in-conversation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_time_s": round(total_time, 2),
        "num_questions": len(BASELINE_ANSWERS),
        "num_answered": len(per_question_results),
        "ragas": ragas_scores,
        "hallucination_rate": round(halluc_rate, 4),
        "meets_hallucination_target": halluc_rate <= 0.02,
        "per_question": per_question_results,
    }

    # Save reports
    out_dir = Path(os.path.dirname(__file__)) / ".." / "metrics" / "baseline_in_conversation"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    hallucination.save_report(out_dir / "hallucination_report.json")

    # Summary
    print(f"\n  === BASELINE EVALUATION SUMMARY ===")
    print(f"  Questions: {len(BASELINE_ANSWERS)} | Answered: {len(per_question_results)}")
    print(f"  Hallucination rate: {halluc_rate:.2%} (target: <2%)")
    print(f"  Faithfulness: {ragas_scores['faithfulness']:.4f}")
    print(f"  Answer relevance: {ragas_scores['answer_relevance']:.4f}")
    print(f"  Context precision: {ragas_scores['context_precision']:.4f}")
    print(f"  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
