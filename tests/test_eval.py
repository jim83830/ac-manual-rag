from eval.eval import Outcome, Question, judge, load_questions, summarize
from rag.chunk import Chunk
from rag.retrieve import RankedHit, RetrievalResult


def result(found, pages_scores):
    ranked = [
        RankedHit(chunk=Chunk(id=f"ac:{p}", doc_id="ac", doc_name="客廳冷氣", pdf_page=p, printed_page=None,
                              section="S", text="t"), score=s, vector_rank=1)
        for p, s in pages_scores
    ]
    return RetrievalResult(query="q", candidates=[], ranked=ranked, found=found)


def test_load_questions(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text(
        "- q: 強力運轉怎麼開？\n  expect: [{doc_id: ac, pdf_page: 11}]\n- q: 可以烘衣服嗎？\n  expect: []\n",
        encoding="utf-8",
    )
    assert load_questions(path) == [Question("強力運轉怎麼開？", [("ac", 11)]), Question("可以烘衣服嗎？", [])]


def test_judge_answerable_hit():
    outcome = judge(Question("q", [("ac", 11)]), result(True, [(12, 2.0), (11, 1.0)]))
    assert outcome.correct
    assert outcome.top_score == 2.0
    assert outcome.pages == [("ac", 12), ("ac", 11)]


def test_judge_answerable_miss_when_refused():
    assert not judge(Question("q", [("ac", 11)]), result(False, [(11, -3.0)])).correct


def test_judge_unanswerable():
    question = Question("q", [])
    assert judge(question, result(False, [(11, -3.0)])).correct
    assert not judge(question, result(True, [(11, 2.0)])).correct


def test_judge_empty_ranked_has_no_score():
    assert judge(Question("q", []), result(False, [])).top_score is None


def test_summarize_counts():
    outcomes = [
        Outcome(Question("a", [("ac", 1)]), True, 2.0, [("ac", 1)], True),
        Outcome(Question("b", [("ac", 2)]), True, 0.5, [("ac", 3)], False),
        Outcome(Question("c", []), False, -4.0, [("ac", 3)], True),
    ]
    text = summarize(outcomes)
    assert "檢索命中率 1/2 = 50%" in text
    assert "拒答正確率 1/1 = 100%" in text
    assert "❌" in text and "b" in text
