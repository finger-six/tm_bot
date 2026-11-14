# Terraforming Mars AI

A project for learning and comparing agents that play two-player Terraforming Mars games.

## Language

**Agent**:
A decision maker that chooses moves for one player in a Terraforming Mars game.
_Avoid_: Model, simulator (these are components, not the complete decision maker).

**Self-play game**:
A training game in which agents controlled by the training system play both seats.

**Baseline opponent**:
A fixed agent used to measure whether a learned agent improves; its behavior does not change during an evaluation comparison.

**Evaluation game**:
A game used to measure an agent's playing strength against fixed opponents or another agent, rather than to train that agent.

**Training format**:
Two-player Terraforming Mars with drafting and Prelude as the only expansion, played on Tharsis, Elysium, Hellas, Vastitas Borealis, or Amazonis Planitia.

**Supported map**:
A map whose rules are verified against the project's accepted fidelity bar, and so is eligible for training and evaluation games. Currently Tharsis, Elysium, Hellas, and Vastitas Borealis (Nova variant).
_Avoid_: Available map, playable map.

**Actor observation**:
Everything an agent may use for one player's decision: public state, that player's own private cards, and that player's own history of seen and passed cards. Never the opponent's hand, deck order or RNG state.
_Avoid_: State (ambiguous with the engine's authoritative state).

**Decision**:
One engine prompt waiting on a player, answered by one complete response, built from a sequence of atomic choices.
_Avoid_: Move, turn (a turn contains several decisions).

**Candidate**:
One legal option for an atomic choice within a decision, enumerated by the adapter from the actor's own prompt.

**Jev-inspired agent**:
An independently trained local Terraforming Mars agent inspired by structured decisions and probability outputs; it is a comparison agent, not a reproduction of TypeSafe AI's proprietary Jev model.

**Human-data warm-start**:
Initial training from recorded human Terraforming Mars games before the agent improves through self-play.
