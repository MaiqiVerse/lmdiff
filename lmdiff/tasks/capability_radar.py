from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lmdiff.tasks.base import BaseEvaluator, Task, TaskResult
from lmdiff.tasks.evaluators import ContainsAnswer

if TYPE_CHECKING:
    from lmdiff._engine import Engine
    from lmdiff.engine import InferenceEngine
    from lmdiff.probes.loader import ProbeSet


@dataclass(frozen=True)
class DomainRadarResult:
    """Per-domain measurement for one engine."""
    domain: str
    n_probes: int
    accuracy: float | None
    """Correct / **scorable**, or ``None`` when nothing was scorable.

    .. versionchanged:: 0.4.4
       Was correct / ``n_probes``, counting unjudgeable probes as
       wrong. On ``v01`` under the ``multiple_choice`` evaluator that
       rendered as a clean ``0.0`` across all 90 probes, because none
       of them carries the metadata that rule needs."""
    bd_vs_baseline: float | None = None
    n_unscorable: int = 0
    """Probes the evaluator could not judge (v0.4.4). Excluded from
    ``accuracy``. Non-zero means the evaluator does not fit these
    probes, which is a different statement from a low score."""

    @property
    def n_scorable(self) -> int:
        """What ``accuracy`` rests on."""
        return self.n_probes - self.n_unscorable

    @property
    def fully_scorable(self) -> bool:
        return self.n_unscorable == 0


@dataclass
class RadarResult:
    """Capability radar result for a pair of configs (or single config)."""
    engine_a_name: str
    engine_b_name: str | None
    domains: list[str]
    a_by_domain: dict[str, DomainRadarResult]
    b_by_domain: dict[str, DomainRadarResult] | None
    bd_by_domain: dict[str, float] | None
    bd_healthy_by_domain: dict[str, float | None] | None
    degeneracy_rates: dict[str, dict[str, float]] | None
    metadata: dict = field(default_factory=dict)

    def summary_table(self) -> list[dict]:
        """Flat list of rows for report rendering."""
        rows: list[dict] = []
        for d in self.domains:
            a = self.a_by_domain[d]
            row: dict[str, Any] = {
                "domain": d,
                "n_probes": a.n_probes,
                # Emitted next to the accuracy, always, so no consumer
                # can render the number without the denominator it
                # rests on -- the failure this column exists for.
                "n_scorable_a": a.n_scorable,
                "n_unscorable_a": a.n_unscorable,
                "accuracy_a": a.accuracy,
            }
            if self.b_by_domain is not None:
                b = self.b_by_domain[d]
                row["n_scorable_b"] = b.n_scorable
                row["n_unscorable_b"] = b.n_unscorable
                row["accuracy_b"] = b.accuracy
                row["delta_acc"] = (
                    None if a.accuracy is None or b.accuracy is None
                    else b.accuracy - a.accuracy
                )
            if self.bd_by_domain is not None:
                row["bd"] = self.bd_by_domain[d]
            if self.bd_healthy_by_domain is not None:
                row["bd_healthy"] = self.bd_healthy_by_domain[d]
            if self.degeneracy_rates is not None:
                row["degen_a"] = self.degeneracy_rates[d]["a"]
                row["degen_b"] = self.degeneracy_rates[d]["b"]
            rows.append(row)
        return rows


class CapabilityRadar:
    """Multi-domain capability + distribution comparison.

    Runs task evaluation (accuracy) AND behavioral distance (BD)
    per domain for a pair of configs. Two views of the same underlying
    generations.
    """

    def __init__(
        self,
        probes: ProbeSet,
        evaluator: BaseEvaluator | None = None,
        max_new_tokens: int = 16,
    ) -> None:
        """``evaluator`` is the **fallback** from v0.4.4 onward.

        Probes carrying a ``scoring`` field are judged by the evaluator
        they name; this one covers the rest. Before v0.4.4 it judged
        everything, which is why a mixed-format set could not be scored
        correctly by any single choice -- see PHASE_PLAN §5.1.
        """
        self.probes = probes
        self.evaluator = evaluator or ContainsAnswer()
        self.max_new_tokens = max_new_tokens

        domains = probes.domains
        if len(domains) < 2:
            raise ValueError(
                f"CapabilityRadar requires at least 2 domains, got {len(domains)}: {domains}"
            )

    def _run_task_for_domain(
        self,
        domain: str,
        domain_probes: ProbeSet,
        engine: "Engine",
        outputs: list[str] | None = None,
        prefix_text: str = "",
        generate_kwargs: dict | None = None,
    ) -> TaskResult:
        task = Task(
            name=f"radar_{domain}",
            probes=domain_probes,
            evaluator=self.evaluator,
            max_new_tokens=self.max_new_tokens,
        )
        return task.run(
            engine, outputs=outputs, prefix_text=prefix_text,
            generate_kwargs=generate_kwargs,
        )

    def run_single(
        self,
        engine: "Engine",
        *,
        prefix_text: str = "",
        generate_kwargs: dict | None = None,
    ) -> RadarResult:
        """Accuracy-only radar for one engine.

        .. versionchanged:: 0.4.5
           Speaks the ``Engine`` Protocol. ``prefix_text`` and
           ``generate_kwargs`` must be supplied for any config with a
           ``system_prompt`` or non-greedy decode — the Protocol's
           engines are stateless, and omitting them measures a different
           configuration without raising. Build both from a ``Config``
           with ``lmdiff._prompting``.
        """
        by_domain = self.probes.by_domain()
        domains = sorted(by_domain.keys())

        a_results: dict[str, DomainRadarResult] = {}
        for d in domains:
            tr = self._run_task_for_domain(
                d, by_domain[d], engine,
                prefix_text=prefix_text, generate_kwargs=generate_kwargs,
            )
            a_results[d] = DomainRadarResult(
                domain=d,
                n_probes=tr.n_probes,
                accuracy=tr.accuracy,
                n_unscorable=tr.n_unscorable,
            )

        return RadarResult(
            engine_a_name=engine.name,
            engine_b_name=None,
            domains=domains,
            a_by_domain=a_results,
            b_by_domain=None,
            bd_by_domain=None,
            bd_healthy_by_domain=None,
            degeneracy_rates=None,
        )

    def run_pair(
        self, engine_a: "InferenceEngine", engine_b: "InferenceEngine",
    ) -> RadarResult:
        """Full radar: accuracy per engine + BD per domain.

        .. deprecated:: 0.4.5
           Not ported to the ``Engine`` Protocol and **removed in
           v0.5.0**. It requires a v0.2.x ``InferenceEngine``.

           The reason it is not ported rather than merely unported: it
           exists to pair accuracy with ``BehavioralDistance`` on shared
           generations, and the live path does not compute BD at all —
           ``_pipeline`` derives δ directly via ``engine.score``,
           deliberately (see ``geometry`` module docstring). Its only
           in-tree caller, ``ModelDiff.capability_radar``, is on the same
           removal list. Porting it would mean porting BD, which reaches
           for ``engine.tokenizer`` — a model object, which metrics are
           not allowed to see.

           ``run_single`` is ported and is the part worth keeping. If
           accuracy paired with δ on shared generations is wanted, it
           belongs in ``_pipeline``, where both engines and the shared
           generations already are — which is what v0.4.5 does for
           ``accuracy_by_variant``.
        """
        import warnings

        from lmdiff.metrics.output.behavioral_distance import BehavioralDistance

        warnings.warn(
            "CapabilityRadar.run_pair is deprecated since v0.4.5 and will "
            "be removed in v0.5.0. It requires the deprecated "
            "InferenceEngine. Use run_single for per-domain accuracy, or "
            "lmdiff.family() for accuracy paired with change geometry.",
            DeprecationWarning,
            stacklevel=2,
        )

        by_domain = self.probes.by_domain()
        domains = sorted(by_domain.keys())
        bd_metric = BehavioralDistance()

        a_results: dict[str, DomainRadarResult] = {}
        b_results: dict[str, DomainRadarResult] = {}
        bd_by_domain: dict[str, float] = {}
        bd_healthy_by_domain: dict[str, float | None] = {}
        degeneracy_rates: dict[str, dict[str, float]] = {}

        for d in domains:
            domain_probes = by_domain[d]

            # Generate once per engine, share the outputs between the task
            # evaluator and BD. Under sampling decode, two independent
            # generate() calls diverge; accuracy and BD would end up
            # describing different samples. See L-010.
            gen_a = engine_a.generate(
                domain_probes.texts, n_samples=1, max_new_tokens=self.max_new_tokens,
            )
            gen_b = engine_b.generate(
                domain_probes.texts, n_samples=1, max_new_tokens=self.max_new_tokens,
            )

            # Legacy GenerationResult -> the list[str] Task.run now takes.
            tr_a = self._run_task_for_domain(
                d, domain_probes, engine_a,
                outputs=[c[0] for c in gen_a.completions],
            )
            tr_b = self._run_task_for_domain(
                d, domain_probes, engine_b,
                outputs=[c[0] for c in gen_b.completions],
            )

            bd_result = bd_metric.compute(
                engine_a, engine_b, domain_probes.texts,
                max_new_tokens=self.max_new_tokens,
                pre_gen_a=gen_a, pre_gen_b=gen_b,
            )

            a_results[d] = DomainRadarResult(
                domain=d,
                n_probes=tr_a.n_probes,
                accuracy=tr_a.accuracy,
                n_unscorable=tr_a.n_unscorable,
                # bd_vs_baseline deliberately None: BD is symmetric,
                # lives in top-level bd_by_domain only.
            )
            b_results[d] = DomainRadarResult(
                domain=d,
                n_probes=tr_b.n_probes,
                accuracy=tr_b.accuracy,
                n_unscorable=tr_b.n_unscorable,
            )

            bd_by_domain[d] = bd_result.value
            bd_healthy_by_domain[d] = bd_result.details.get("bd_healthy")
            degeneracy_rates[d] = {
                "a": bd_result.details.get("degeneracy_rate_a", 0.0),
                "b": bd_result.details.get("degeneracy_rate_b", 0.0),
            }

        return RadarResult(
            engine_a_name=engine_a.model_name,
            engine_b_name=engine_b.model_name,
            domains=domains,
            a_by_domain=a_results,
            b_by_domain=b_results,
            bd_by_domain=bd_by_domain,
            bd_healthy_by_domain=bd_healthy_by_domain,
            degeneracy_rates=degeneracy_rates,
        )
