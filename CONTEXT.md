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

**Jev-inspired agent**:
An independently trained local Terraforming Mars agent inspired by structured decisions and probability outputs; it is a comparison agent, not a reproduction of TypeSafe AI's proprietary Jev model.

**Human-data warm-start**:
Initial training from recorded human Terraforming Mars games before the agent improves through self-play.
