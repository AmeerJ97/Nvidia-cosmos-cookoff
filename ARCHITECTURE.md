# CLASP Architecture

CLASP is a research prototype for epistemic independence in multi-agent systems. The architecture treats robotic handoff release as a stopping-time POMDP: the true physical state is hidden, each agent receives a partial observation, each agent privately proposes `ACT` or `THINK`, and the orchestrator commits release only when the stopping rule is satisfied.

---

## Formal Model

At frame `t`, the latent state `s_t` includes human grip readiness, gripper-object contact geometry, object stability, occlusion, velocity divergence, and distance from the safe release interval. This state is not directly observable.

Each agent `i` receives an observation:

```text
o_t^i = phi_i(frame_t, LiveKV_t, ArchiveKV_t, Oracle_t)
```

where `phi_i` is the agent's identity-specific projection from the P x T x M asymmetry matrix:

- `P`: prompt bias, such as conservative, speed-optimized, kinematic skeptic, or archive-loyalist.
- `T`: temporal stride over the LiveKV window.
- `M`: modality mask, such as full, gripper, or velocity.

Each agent returns a private action:

```text
a_t^i in {THINK, ACT}
```

The stopping time is:

```text
tau = inf { t : Consensus_t({a_t^i}) = ACT and Oracle_t does not veto }
```

The release decision is therefore not an agent-local classification. It is a stopping-time policy over independent partial beliefs, external consensus, asymmetric loss, and hard physical constraints.

---

## Design Invariant: Epistemic Independence

The agents are intentionally blind to one another:

- They do not receive peer decisions.
- They do not know the consensus threshold.
- They do not know whether their proposal will be pivotal.
- They do not receive global reward traces during a live frame.

This isolates belief formation from coordination strategy. The orchestrator can use consensus for safety without creating incentives for agents to game a voting protocol.

---

## Frame Lifecycle

1. The orchestrator receives a trajectory frame.
2. The vision bridge extracts physical constraint signals and may issue a hard veto.
3. Cosmos Embed produces a 768-dimensional frame embedding.
4. LiveKV stores the frame summary for local temporal continuity.
5. ArchiveKV retrieves top-k distilled memories from prior successful releases.
6. Each living agent receives its own masked prompt context and returns `ACT` or `THINK`.
7. The scorer applies O(1) kinematic evaluation against the safe release interval.
8. The dynamic consensus threshold decides whether enough independent `ACT` proposals exist.
9. If release is correct, distilled reasoning is archived and SFT records are emitted.
10. If agents die, Hyper-GRPO samples replacement identities.

---

## Spectating Burn-In

Spectating Burn-In is the formal anti-imitation mechanism embedded in `clasp_pkg/agents.py`. It supplies agents with failure-only cases before decision time:

- fatal early `ACT`;
- repeated premature `ACT`;
- Life-Points collapse and replacement by Hyper-GRPO.

No success cases are shown. The intent is to prevent behavioral cloning from positive demonstrations and to force release behavior to emerge from survival pressure. The burn-in examples define absorbing failure boundaries; they do not define a template for when release should occur.

This matters because a stopping-time controller should not simply imitate an observed release frame. It must infer whether the current partially observed physical state has entered the safe release set.

---

## Survival Dynamics

The Life-Points state `L_i` is updated every frame:

```text
THINK:          L_i <- L_i - gamma_think
wrong ACT:      L_i <- L_i - gamma_wrong
early wrong ACT:L_i <- L_i - 2 * gamma_wrong
correct ACT:    L_i <- L_max
```

With the current configuration:

- `gamma_think = 2`
- `gamma_wrong = 33`
- early wrong `ACT = 66`
- `L_max = 100`

The loss structure encodes a safety preference: delay is costly, but premature release is much more costly. An agent can survive caution longer than unsafe decisiveness.

---

## Hyper-GRPO Identity Search

Hyper-GRPO is a gradient-free categorical policy over the 36 identities in the P x T x M matrix. When an agent dies, its identity logit is updated from accumulated reward relative to reward history. Replacement samples are drawn from the updated distribution.

The update is deliberately identity-level rather than weight-level. CLASP searches over epistemic roles and observation projections while leaving model weights fixed during evaluation. This keeps online adaptation auditable and bounded.

---

## Consensus Schedule

Consensus is frame-adaptive:

| Frame region | Required ACT fraction | Rationale |
|---|---:|---|
| Early frames (`t < 8`) | 100% | Minimize premature-release risk under sparse evidence |
| Mid frames (`8 <= t < 15`) | 85% | Require strong agreement as evidence accumulates |
| Late frames (`t >= 15`) | 66% | Reduce excessive non-release once temporal context is richer |

The threshold is invisible to agents. It is part of the external stopping-time policy, not the agent prompt contract.

---

## Memory Architecture

LiveKV provides local temporal continuity through a Redis-backed sliding window. ArchiveKV stores distilled successful-release principles in a FAISS index keyed by Cosmos embeddings. Agents receive only the memories selected for their masked context; this preserves the asymmetry assumption.

ArchiveKV is intentionally secondary to live evidence. Retrieved memories are treated as prior evidence, not proof that the current trajectory has reached the safe set.

---

## Safety Boundary

CLASP's primary safety boundary is conservative release control:

- physics veto before VLM inference when signals are clearly unsafe;
- independent agent belief formation;
- dynamic consensus outside the agents;
- asymmetric Life-Points penalties;
- failure-only burn-in;
- replacement rather than in-place online weight mutation.

The resulting behavior is intentionally fail-safe: non-release is acceptable when uncertainty remains, while premature release is treated as the dominant failure mode.
