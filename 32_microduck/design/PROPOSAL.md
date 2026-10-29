# Joy Lab II: Microduck first day and learning plan (owner's proposal)

Saved verbatim from the owner's plan (2026-10-24). This file is the full intent; each phase's build
scope was cut from it, and where they differ, the build scope wins.

---

Build one dependable Microduck in MuJoCo, give it a collection of tasks, and explore different ways to control and teach it. The robot, sensors, and environment provide a stable foundation. Different model configurations determine how it understands instructions, chooses actions, predicts consequences, and learns from experience.

The central question is whether experience makes the duck more capable—and whether those abilities carry into situations it has never encountered before.

We should be able to watch the answer: a duck that searches effectively, corrects a missed interaction, investigates an unfamiliar object, or uses yesterday's discovery in a different room.

# Environment and robot

Start with a small room containing a home mat, colored destinations, a partition, a ball, and a broad floor button connected to a lamp. Give Microduck reliable standing, walking, turning, looking, and stopping.

Support two ways to run the room:

- **Repeatable episodes**, where we restore starting conditions and compare controllers.
- **A continuing life**, where world state, memory, and permitted learning persist between sessions.

Keep the implementation in one self-contained ProjectJoy folder. Use MuJoCo for simulation and MLX for neural components, following the existing environment and resource rules. Validate movement-policy compatibility before building autonomous behavior on top.

# Tasks and task versions

Each task is a family of related situations. Training, development, and reserved test versions are designed together.

| Task | What the duck does | Useful variations |
| - | - | - |
| **Go there and come home** | Reach a requested destination, navigate around obstacles, and return home. | Starting pose, destination location, obstacle arrangement, route length. |
| **Set the lamp** | Observe the lamp, press the button when necessary, and verify the requested state. | Button and lamp positions, initial lamp state, approach direction. |
| **Find the missing ball** | Remember where a ball disappeared, obtain another view, and locate it again. | Occluders, disappearance direction, waiting time, camera viewpoint. |
| **Nudge into a target** | Approach a ball, make controlled contact, observe its motion, and correct subsequent attempts. | Target position, object size, rolling resistance, initial alignment. |
| **Discover what this does** | Investigate an unfamiliar button or object, retain the relationship, and use it later. | Button–lamp pairings, inactive buttons, spatial separation, mechanism combinations. |
| **Adjust and remember** | Adapt to changed object behavior, then remain competent when earlier conditions return. | Familiar–changed–familiar sequences, appearance changes, combinations of known properties. |
| **Explore before being asked** | Choose investigations while idle, then use discoveries to complete a later request. | Exploration time, available objects, distractors, subsequent goals. |

Begin with navigation, lamp interaction, and finding an occluded ball. Add controlled nudging after approach and contact work reliably. Mechanism discovery and adaptation build on those abilities.

Every task gets a short specification: starting conditions, observations, available actions, success and failure conditions, permitted variations, and a satisfying demonstration.

# Baseline bot and experience collection

Separate reliable movement from task completion.

The movement layer supplies the established gait and bounded commands. Freeze it during the initial controller comparisons.

Above that, build a reference bot using simple visual tracking, explicit memory, and a hand-written controller. It should finish the starter tasks, retry missed actions, search when a target disappears, and check outcomes. For discovery tasks, it can systematically try buttons and record what happens.

Then add a **small recurrent neural policy** as the first learned baseline. It uses observation history to select actions without requiring a language model or explicit world model.

An exact-simulator-state controller can help establish whether a task is physically solvable. Label that separately from results obtained through the robot's sensors.

## Collect experience worth learning from

The reference bot, manual control, and later model controllers can all produce demonstrations.

Record images, body state, instructions, timestamps, executed commands, observed outcomes, and which controller produced the trajectory. Keep corrections and recoveries as well as clean successes.

This gives the VLA and other policies a training source, and gives world models observations paired with the actions that actually occurred.

# Controller configurations

Start with a small set of named configurations, then explore combinations that answer useful questions.

| Configuration | What we investigate |
| - | - |
| **Simple controller + memory** | What reliable rules and observation history can accomplish. |
| **Recurrent policy** | How much a compact policy can learn directly from experience. |
| **Liquid decision model + existing skills** | Whether structured choices improve skill selection and recovery. |
| **Generative VLM + existing skills** | What language-based interpretation and planning contribute. |
| **VLA — SmolVLA** | Whether observations and instructions can directly produce useful learned action sequences. |
| **Decision model + VLA** | Whether structured goal or skill selection helps a VLA execute longer activities. |
| **Generative VLM + VLA** | Whether language-based planning and learned movement execution work well together. |
| **World model + planner** | Whether predicting consequences improves action selection. |
| **Decision model or VLM + world model** | Whether semantic reasoning and physical prediction help each other. |
| **VLA + world model** | Whether predicted consequences help select, assess, or revise action sequences. |

A later configuration can combine a high-level planner, a VLA executor, and a world model once the individual pieces work.

# SmolVLA and VLA combinations

SmolVLA is a primary candidate. Its architecture combines visual observations, robot state, and language with a flow-matching action expert that produces continuous action chunks. [SmolVLA architecture](https://huggingface.co/blog/smolvla)

For Microduck, the proposed starting interface is:

**Camera + body state + instruction → adapted SmolVLA → short movement and head-pose command sequences → established gait controller.**

This gives the VLA responsibility for approach, alignment, searching, and interaction while preserving established walking competence.

Microduck-specific demonstrations, action mapping, normalization, and a verified MLX implementation are part of this branch's work. Direct joint control can become a later experiment.

## Keep two control comparisons clear

In skill-selection experiments, controllers choose from the same available skills. In action-generation experiments, policies produce commands through the same lower-level interface.

We can also compare complete systems on task completion, but must record their different assistance and capabilities. A VLM calling a finished navigation skill and a VLA learning navigation from movement commands are solving different portions of the problem.

# World models JEPA and action generation

Begin with a small predictor trained on recorded interactions. Compare predictions built on frozen visual features with representations learned through a JEPA-style objective.

The predictor should answer an actionable question: given recent observations and a proposed action, what is likely to happen next?

Its predictions must influence a decision—an approach, a nudge, a retry, or a choice to look again. Useful prediction is demonstrated through behavior as well as prediction error.

## Action generation experiments

SmolVLA already includes a flow-based action expert. We can compare that approach with a straightforward action head and, later, an NCM-inspired hierarchy that generates movement sequences from coarse structure to finer commands.

The released NCM system addresses text generation. Applying that approach to robot trajectories would be an experimental adaptation within Joy Lab. [NCM repository](https://github.com/HazyResearch/ncm)

# Generalization and held out task versions

A different random seed gives us another instance. We also want deliberately reserved forms of novelty.

| Test family | What we reserve | What it tests |
| - | - | - |
| **New instances** | Unseen starting states and seeds under familiar rules. | Reliability beyond memorized episodes. |
| **New appearances** | Colors, textures, lighting, and object appearances. | Dependence on accidental visual cues. |
| **New arrangements** | Layouts, routes, button locations, and occluder placements. | Spatial transfer. |
| **New physical conditions** | Object properties within a defined operating range. | Unfamiliar dynamics. |
| **New combinations** | Familiar properties combined in unfamiliar ways. | Recombination of learned knowledge. |
| **New wording** | Reserved paraphrases and equivalent instructions. | Language generalization. |
| **New compositions** | Familiar abilities required in a new sequence. | Reuse of skills toward longer goals. |

For example, the duck trains with several button-and-lamp arrangements. Reserved versions move the button, change its appearance, begin with the lamp already on, or place the button behind a partition.

For an unfamiliar mechanism, give it an opportunity to investigate. Success includes obtaining the evidence needed to act.

Vary appearance and physics independently where practical. A particular color should not always reveal which object rolls farther.

## Training development and reserved test collections

1. **Training worlds:** available for demonstrations, exploration, parameter updates, and replay.
2. **Development worlds:** used to select checkpoints and improve the design.
3. **Reserved test worlds:** evaluated after those choices are made.

Record split assignments and task versions. Reserve layouts, properties, or combinations according to the question being tested. Avoid splitting adjacent frames from the same experience across training and evaluation.

If we repeatedly inspect a test case and tune against it, treat it as a development case and reserve fresh cases for subsequent claims.

# Learning and adaptation

Learning is a separate configuration choice. The same controller can operate with:

- Fixed parameters and within-episode history.
- Persistent episodic memory.
- Replay updates between activities.
- Online adaptation of selected components.
- Consolidation using recent and older experience.
- Later adaptation of decision-model, VLA, or language-model components.

Label what changed: **memory added, predictor updated, policy updated, or language/decision model adapted**.

This lets us distinguish better recall from better predictions or newly learned movement.

## Evaluate generalization and adaptation separately

For generalization, freeze learned parameters and specify the permitted starting memory. Ordinary observation history and replanning remain available during an episode.

For adaptation, deliberately allow learning. Measure performance on arrival, experience required to improve, and competence when familiar conditions return.

Reserved test experiences stay out of training and replay during the frozen evaluation. Continuing-life experiments preserve experience according to their documented sequence.

# Earlier research and experiments

Use the earlier research as the experiment collection. Terra contributes navigation; Physics Detective contributes probing; Object Permanence contributes occlusion; Daydream Engine contributes prediction; Two Brains contributes adaptation; Sleeping Robot contributes replay and retention.

Curiosity experiments ask whether choosing experiences helps more than simple exploration. Grounded-language experiments ask whether physical experience improves fresh predictions or reasoning inside the model itself.

These become activities and configurations for the same duck.

# Measurements and replay

Keep the visible scorecard small:

- Completion rate and attempts required.
- Falls, collisions, and missed interactions.
- Simulated task time and actual inference latency.
- Familiar versus reserved-task performance.
- Improvement after experience and retention afterward.
- Whether predictions distinguish actions and improve decisions.

Use common starting scenes and interaction budgets for comparable runs. Record timing behavior consistently. Use multiple scene seeds, and multiple training seeds when drawing conclusions about a learning method.

The replay should show **the goal, selected action, expected outcome, observed outcome, and what was retained or updated**.

# First satisfying version and next experiments

A stable Microduck completes the three starter tasks with the reference bot. Each task has training, development, and reserved versions. Saved configurations reproduce runs, and controllers can be exchanged without rebuilding the robot or room.

The first learned experiments compare a recurrent policy, a decision model, and adapted SmolVLA. Predictive planning and longer-term learning follow as additional configurations.

Add contact tasks, unfamiliar mechanisms, purposeful exploration, changing dynamics, learned representations, and new action generators as individual experiments.

The experience to pursue is a duck entering an unfamiliar room, finding a button in a new location, checking what it does, satisfying your request, and later applying that experience somewhere else. Save its missed presses, strange detours, and successful recoveries alongside the clean demonstrations.
