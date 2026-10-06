# Local self-play and Jev-inspired learning methods

Research date: 2026-10-06. Resolves the research question in [Investigate local self-play and Jev-inspired learning methods](https://github.com/finger-six/tm_bot/issues/4); it does not select an algorithm, architecture, or training budget.

## Scope and evidence boundary

The training format is two-player Terraforming Mars with drafting and Prelude only, across Tharsis, Elysium, Hellas, Vastitas Borealis, and Amazonis Planitia. The main learning path is human-data warm-start followed by self-play; pure self-play is a secondary comparison. The Jev-inspired agent is an independently trained local comparison agent. These are project requirements, not conclusions of the cited papers. See [CONTEXT.md](../../CONTEXT.md).

The resource envelope supplied for this investigation is one RTX 4080 Super with 16 GB VRAM and 32 GB system RAM, with no paid cloud by default. It has not been benchmarked here. All Terraforming Mars adaptations and experiment designs below are **proposals/inferences**. Published results establish methods in their original domains, not feasibility or convergence in this training format.

## What survives from AlphaZero, and what needs adaptation

AlphaZero combines a neural policy and value prediction with MCTS. Self-play produces root search visit distributions as policy targets and terminal win/draw/loss outcomes as value targets. Its network learns both using cross-entropy and squared-error losses. It masks illegal moves and renormalizes the remaining probabilities. Its demonstrated games are chess, shogi, and Go; these results do not demonstrate handling private hands. [AlphaZero paper, algorithm and supplementary action representation](https://arxiv.org/pdf/1712.01815)

The reusable pattern is search-guided policy improvement and value learning. The following are proposed adaptations, not an assertion that adding a mask makes vanilla AlphaZero sound:

| Format property | Proposed adaptation and check |
| --- | --- |
| Hidden hands and previously seen draft packs | Define the actor input as its own observation/action history, including private observations it legitimately retained. Never feed the opponent's actual hidden hand or future deck order. Test that changing only unobserved information cannot change the actor's direct prediction with the same inference randomness. |
| Private drafting choices | Preserve the information timing of the rules; serializing decisions in a simulator must not reveal an earlier player's private selection to the later player. Sample possible hands jointly, respecting cards already observed, revealed, removed, and passed. |
| Chance draws | Keep actual environment randomness separate from hypothetical search samples. Sample without replacement under the remaining-deck constraints, or enumerate small chance branches where practical. A lucky sampled draw is not a choice the player can force. |
| Variable and compound choices | Use an exact legal-choice schema and either enumerate complete candidate actions or factor a choice into conditional stages with legality checked at every stage. Masking must cover targets, payments, selections, and placement, not just action names. Count auxiliary decisions when comparing search budgets. |
| Long games and delayed payoffs | Compare terminal targets with bootstrapped/truncated-return targets; record the bias/variance tradeoff and early/mid/late-game performance. Do not silently turn a time-limit truncation into a draw or replace the winner objective with production growth. |
| Turn structure and reward | Back up from the relevant player's perspective; do not blindly negate after every simulator step when the same player has another choice. Define terminal utility and tie handling before collecting data. Two-player win/loss utility can be made zero-sum; two independent raw-score rewards generally are not. |
| Five maps | Include map identity and map-specific public state. Report results per map, not only an aggregate. Geometric augmentation is a hypothesis to verify against map rules, not an automatic symmetry. |

Imperfect-information search has an additional strategic problem: an action's value can depend on the probability of choosing it and the beliefs it induces. ReBeL explicitly explains why substituting ordinary state values into perfect-information search is unsound in this setting. A recurrent observation encoder alone does not establish that the resulting search is sound. [ReBeL paper, sections 1–4](https://proceedings.neurips.cc/paper_files/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf)

## Defensible information-aware candidates

These are options to investigate, not a ranking or an adopted design.

| Candidate | Published basis | Local adaptation to investigate | Material limitation |
| --- | --- | --- | --- |
| Neural information-set MCTS (ISMCTS) | Cowling et al. search information-set trees and discuss weaknesses of separately solving determinizations, including strategy fusion: pretending one can choose differently in indistinguishable states. [ISMCTS paper](https://eprints.whiterose.ac.uk/id/eprint/75048/1/CowlingPowleyWhitehouse2012.pdf) | Add observation-conditioned policy priors and leaf estimates; sample consistent hidden-card assignments; aggregate only across decisions that share the acting player's information. Measure benefit against a direct policy and simple determinization baseline. | A heuristic candidate; no Nash-equilibrium claim. Sharing a root across samples does not by itself fix every opponent-information or belief problem. Test the opponent node's information rather than allowing root-player knowledge to leak. |
| Belief-particle planning against a specified opponent model | POMCP combines sampled belief updates with history-tree Monte Carlo planning using a black-box simulator for a POMDP. [POMCP paper](https://papers.nips.cc/paper_files/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf) | Treat a fixed or sampled opponent policy as part of the transition model; retain particles constrained by the actor's draft history and public play. | POMCP is single-agent planning, not an equilibrium solver for two adapting players. Inferred hand distributions depend on the opponent model; test robustness to misspecification and particle impoverishment. |
| Public-belief learning and depth-limited subgame solving | ReBeL learns policy/value functions over public belief states and combines them with imperfect-information search. Its theory is for two-player zero-sum games. [ReBeL paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf) DeepStack combines continual re-solving with learned counterfactual-value estimates in heads-up poker. [DeepStack paper](https://poker.cs.ualberta.ca/publications/17science.pdf) | Investigate a small restricted subgame or compressed belief representation before full-game training. Private draft histories and correlated card assignments must remain representable. | Enumerating private histories/ranges and learning range-conditioned values may dominate local memory and compute. An approximation or action abstraction needs separate validation. Poker results and theoretical assumptions do not supply a local Terraforming Mars convergence guarantee. |
| Search-light self-play with an average strategy | NFSP learns an approximate best response with reinforcement learning and an average strategy by supervised learning from its own past best-response behavior, using separate memories and anticipatory mixing. [NFSP paper](https://arxiv.org/pdf/1603.01121) | An information-state candidate scorer with value/Q prediction can compare direct play with search-assisted play without paying search cost at every training decision. | Its reported poker results do not establish success here. Human imitation data are not the historical-best-response reservoir that defines NFSP; adding them changes the method. Approximate function learning and long horizons still matter. |
| Search-light actor-critic | PPO alternates sampled interaction with multiple minibatch updates of a surrogate policy objective. [PPO paper](https://arxiv.org/abs/1707.06347) | A masked policy over legal structured candidates, conditioned on legitimate history, is a useful self-play comparator with a value critic. | PPO's original benchmarks are not equilibrium results for hidden-information board games. This path can exploit its training opponents while remaining vulnerable to others. |
| Neural regret minimization | Deep CFR approximates CFR with neural networks rather than an explicit information abstraction. [Deep CFR paper](https://proceedings.mlr.press/v97/brown19b.html) | A research alternative if regret traversals and information-state targets can be collected locally; profile traversal and replay costs first. | Full-game traversals, wide compound choices, and multiple memories can be expensive. Its formulation and experiments are not evidence that this game fits the machine or learns within an unstated budget. |

A compact hand-belief approximation may be useful, but independent per-card inclusion probabilities do not encode all correlations created by without-replacement draws and drafting. That is a modeling limitation of the proposed approximation, to be tested with consistency checks and toy games.

## Human-data warm-start and opponent sampling

Human imitation followed by self-play is supported as a precedent by the original AlphaGo, which combined supervised learning from expert games with reinforcement learning from self-play. AlphaZero's tabula-rasa procedure is a different experimental path. Neither result predicts which path will perform better with this project's data. [AlphaGo paper](https://www.nature.com/articles/nature16961), [AlphaZero paper](https://arxiv.org/pdf/1712.01815)

Proposed warm-start targets and limits:

- **Policy:** supervised likelihood of the recorded action, conditioned on that player's reconstructed information and legal candidates. A chosen action is a behavioral label, not proof it was best. Retain losing games unless a declared filtering experiment justifies removal; training only winners changes the data distribution.
- **Value:** recorded terminal utility from the acting player's perspective. This estimates outcomes under recorded play, not optimal value. Self-play later changes the continuation policies; version and distinguish human and self-play targets.
- **Structured factual heads:** use only reconstructable events with explicit timing and labels. For example, predicting a player's final win under a named opponent/continuation policy is different from predicting whether an opponent holds a card. A public log may provide neither private-hand labels nor complete drafting observations.
- **Availability gate:** audit game count, complete versus partial records, player strength, maps, rules/version, Prelude coverage, draft-pack visibility, payment/target detail, and data usage rights. Do not invent unseen hands or missing choices. Mark incomplete examples, exclude them from targets they cannot support, and quantify coverage.
- **Splits:** hold out complete games before extracting positions. Keep repeated/duplicated games together; consider player/time splits. Adjacent positions from one game across train/test would give a misleading test of generalization.

Sequential imitation has distribution shift because the learner's actions affect its future observations. DAgger addresses that with interactive expert labeling of learner-induced states; static historical logs alone do not provide its feedback loop. Here it informs the limitation of warm-start, not a proposal to claim DAgger guarantees without an expert oracle. [Ross et al., DAgger](https://proceedings.mlr.press/v15/ross11a.html)

Training exclusively against the latest checkpoint can create narrow competence. PSRO trains responses to mixtures of policies and uses empirical game analysis for policy selection; the paper also identifies overfitting to training partners. [PSRO paper](https://arxiv.org/abs/1711.00832) Proposed local options are a bounded pool of frozen checkpoints plus fixed baseline opponents, with recorded sampling weights, or NFSP's average-strategy mechanism. A uniform checkpoint pool is a simpler heuristic, not an implementation of all PSRO machinery. Keep evaluation opponents fixed and outside training updates; evaluate the whole cross-play matrix to expose cycles rather than judging strength only against the previous checkpoint.

## What Jev actually supplies, and a local comparison option

TypeSafe's September 15 announcement describes Jev as producing typed probabilistic decisions and names a new architecture, parallel sampler, and Reinforcement Learning for Calibrated Decisions (RLCD). These are first-party product claims, not a published reproducible training algorithm. [Announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

The official model documentation says Jev is not fine-tuned or LoRA-adapted with customer data and serves the same weights for every account. It documents a hosted endpoint and request-based customization. The announcement, model reference, and documentation index reviewed for this report did not provide a reproducible architecture/training recipe or downloadable weights. That is a bounded finding, not proof that no publication could appear later. [Models](https://docs.typesafe.ai/models), [Documentation index](https://docs.typesafe.ai/llms.txt)

The public inspiration is a structured interface with probabilities. TypeSafe's Choice/Score confidence is derived from the output distribution; Noul is a yes/no probability without a separate confidence field. [Confidence documentation](https://docs.typesafe.ai/confidence) Bounded output types do not establish correct game judgments. No vendor speed, calibration, or intelligence claim is transferred to this project.

**Independent local design inference:** investigate a compact neural encoder of public state, private observation history, card/resource features, and map context, with a shared scorer for each legal typed candidate. Possible encoders include an MLP over engineered summaries, permutation-aware card-set aggregation, or a small attention/recurrent model. Deep Sets supplies a published basis for permutation-invariant/equivariant set modeling; it does not specify Jev's architecture. [Deep Sets paper](https://papers.nips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf)

The proposed interface would return legal action IDs/parameters plus a move-selection distribution and separately named value/event predictions. Train it independently from the same allowed human games and its own self-play; use declared imitation/RL losses and held-out calibration. A direct-policy form offers a fast-decision comparison; adding the same information-aware search offers an architecture comparison with search controlled. The local model need not ingest natural-language card descriptions or emulate a general-purpose language model. This is neither JEPA nor a reconstruction of proprietary Jev/RLCD.

## Three different meanings of probability

| Output | Meaning in the proposed experiment | Suitable validation |
| --- | --- | --- |
| Policy `pi(a | information)` | Probability of selecting a move; it may represent imitation, exploration, a mixed strategy, or search visits. | Legal mass, held-out action likelihood for imitation, entropy, and game outcomes. A move with probability 0.8 does not imply an 80% win chance. |
| Factual/event `P(event | information, conditions)` | Probability of a precisely labeled event, such as a win under specified continuation/opponents or a hidden-card event with labels. | Held-out negative log likelihood/Brier score and reliability diagrams, stratified by map, phase, and opponent. Disclose uncertainty when labels are sparse. |
| Action value `Q(information, action, continuation)` | Expected utility after taking an action under specified hidden-state beliefs, opponent policy, and subsequent play. | Held-out rollout estimates, ranking agreement, value error, and playing strength; attach sampling uncertainty and policy version. A softmax of Q is a selection rule, not factual probability calibration. |

Modern neural classifiers can be miscalibrated; temperature scaling is a published post-processing method evaluated by Guo et al. Its success on their benchmarks does not guarantee calibration under self-play distribution shift. Fit calibration only on a validation split and assess it on an untouched test split. [Calibration paper](https://proceedings.mlr.press/v70/guo17a.html) Calibration of action imitation is calibration against what humans choose, not against an unknowable single best action. Track event calibration and policy strength separately; either can improve while the other worsens.

## Fair comparisons and local measurements

The following is a proposed experiment framework, not a prescribed budget:

1. **Warm-start effect:** compare the same agent/search method with human-data warm-start and random initialization. Publish both total compute including pretraining and performance versus subsequent self-play compute. Match training format, evaluator, opponent pool, and stopping/checkpoint rule.
2. **Whole-agent strength:** compare independently trained AlphaZero-inspired and Jev-inspired agents at equal local training wall-time or hardware-hours, counting simulation, search-label creation, preprocessing, tuning, and optimization. If each generates different games, state that this measures the whole learning system.
3. **Architecture effect:** offer the same permitted human split and common training examples/targets to both, then use the same information-aware search and matched evaluation budget. Report any shared features and parameter counts. Sharing data is allowed; copying one agent's weights into the other would answer a different independence question.
4. **Search effect:** evaluate each direct policy and its search-assisted variant, at matched per-decision wall-time and separately matched simulator-call budgets. Equal simulations alone are not equal cost when leaf networks and action decomposition differ. Charge belief updates and chance sampling to the decision budget.
5. **Evaluation:** freeze checkpoints; balance seats and map frequencies; use paired initial seeds/deals with seat swaps where valid. Record chance-stream handling when decisions consume different numbers of draws. Keep hidden state inaccessible to actors. Report win/draw/loss, score margin as a secondary statistic, per-map results, failures, decision-time tails, and uncertainty over paired games and independent training runs.

RL evaluation with few runs can be unreliable; Agarwal et al. develop interval estimates and aggregate metrics to avoid overconfident comparisons. Use repeated independent training seeds where feasible and disclose the count; large evaluation-game counts do not substitute for training-run replication. [Statistical Precipice paper](https://arxiv.org/abs/2108.13264) A cross-play win rate is evidence of performance against that opponent suite, not measured Nash exploitability or proof of superhuman play.

Profile before deciding a training budget:

| Measurement | Record and why |
| --- | --- |
| Simulator and search | State transitions/second; clone/undo, legality, draw sampling, and belief update costs; search calls/nodes and depth; decision counts and completed game-length distribution. Distinguishes rule-engine cost from network cost. |
| Inference | Batch-size sweep; cold versus warmed calls; p50/p95/p99 decision latency including queue/transfer/search; observations and candidates per batch; GPU idle/utilization periods. Report direct policy and searched agent separately. |
| Training | Update time, samples per update, examples/second, replay age, target-generation time, validation time, and end-to-end completed games/hour only after actual measurement. Record concurrent actor/learner contention. |
| Memory | Peak GPU allocated and reserved memory, process and whole-system RAM, replay bytes/example, tree/particle bytes, worker duplication, disk growth, and paging/OOM events. Reserve operating-system/display headroom rather than treating all stated capacity as available. |
| Reproducibility | CPU, storage, OS, driver, framework versions, precision, model size, worker count, seed policy, map mix, and checkpoint identity. Those details are unknown here. |

PyTorch distinguishes tensor-allocated memory from caching-allocator-reserved memory; both are useful. Its profiler exposes CPU/CUDA activity, shapes, and tensor-memory profiling. Measure end-to-end wall-clock separately, with warmed GPU work and appropriate synchronization, and disclose profiler overhead. [CUDA memory notes](https://github.com/pytorch/docs/blob/site/stable/notes/cuda.md), [Profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

Candidate local tactics are bounded replay/tree/particle pools, batched inference, bounded CPU actors, and mixed precision after a correctness comparison. These are hypotheses to profile, not evidence that a particular model fits. Neither a model size, games/hour estimate, required game count, convergence time, nor affordable complete budget can be inferred from the GPU name. The actionable output of a feasibility spike is measured throughput and memory for representative early/mid/late decisions and complete games, including all five maps and training/inference contention.

## Open questions before algorithm selection

- What complete human observations and choices can actually be reconstructed, and on what maps/rules versions?
- Can the simulator cheaply clone or undo, separate public/private state, preserve draft information timing, and reproduce chance events without leaking future draws?
- Which legal-action representation keeps exact behavior while limiting branching and enumeration cost?
- Which belief approximation is consistent with observed draft packs and correlated hidden cards, and how badly does opponent-model mismatch affect decisions?
- What winner/tie utility, truncation policy, evaluation latency target, total local budget, and minimum meaningful improvement will the human select?
- What measured learning curves, cross-play outcomes, calibration errors, and failure rates justify continuing a candidate?

No candidate is established as convergent or practical on this hardware by this report. It supplies cited options and the checks needed to make a later choice reviewable.