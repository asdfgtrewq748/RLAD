## ADDED Requirements

### Requirement: Comprehensive Literature Review

The manuscript SHALL provide a comprehensive literature review that establishes the research context, identifies gaps, and positions Enhanced RLAD as a novel contribution.

#### Scenario: TSAD landscape overview

- **WHEN** Section 1.3 presents the time series anomaly detection (TSAD) literature
- **THEN** the section SHALL cite authoritative survey papers (e.g., "Deep Learning for Time Series Anomaly Detection: A Survey")
- **AND** the section SHALL classify existing methods into categories:
  - Prediction-based methods
  - Reconstruction-based methods
  - Distance-based methods
  - RL-based methods
- **AND** the section SHALL demonstrate broad knowledge of the field

#### Scenario: Industrial label scarcity emphasis

- **WHEN** Section 1.3 discusses anomaly detection in industrial settings
- **THEN** the section SHALL explicitly address the challenge of label scarcity
- **AND** the section SHALL cite literature demonstrating this is a widespread industrial problem
- **AND** the section SHALL connect this challenge to the motivation for semi-supervised learning
- **AND** the section SHALL provide a strong theoretical foundation for the ensemble bootstrapping approach

#### Scenario: Existing RLAD positioning

- **WHEN** Section 1.3 introduces related RL-based anomaly detection work
- **THEN** the section SHALL explicitly cite Golchin & Rekabdar's RLAD paper
- **AND** the section SHALL acknowledge their contributions (pioneering RL+AL framework)
- **AND** the section SHALL identify their limitations:
  - Reliance on single VAE model for supervision signals
  - Use of standard/symmetric reward functions
  - Lack of domain-specific adaptation
- **AND** the section SHALL transition naturally to Enhanced RLAD as the solution

#### Scenario: Active learning context

- **WHEN** Section 1.3 or 2.4 discusses active learning
- **THEN** the section SHALL cite active learning survey papers
- **AND** the section SHALL discuss the value of active learning in resource-constrained industrial settings
- **AND** the section SHALL connect to "human-in-the-loop" paradigms
- **AND** the section SHALL position Margin Sampling as an established, effective strategy

### Requirement: Theoretical Foundation Strengthening

The manuscript SHALL strengthen theoretical sections by connecting concepts to applications and citing authoritative literature.

#### Scenario: MDP modeling motivation

- **WHEN** Section 2.1 describes modeling anomaly detection as an MDP
- **THEN** the section SHALL explain the motivation: converting indirect threshold-based tasks to direct policy-based classification
- **AND** the section SHALL emphasize this is more robust in dynamic environments
- **AND** the section SHALL cite classic RL survey papers (e.g., Sutton & Barto, or recent RL surveys)
- **AND** the section SHALL reference applications of RL to anomaly detection

#### Scenario: DQN stability mechanisms

- **WHEN** Section 2.2 explains DQN and target networks
- **THEN** the section SHALL clearly describe the target network's role in stabilizing training
- **AND** the section SHALL explain how fixed Q targets prevent training oscillation/divergence
- **AND** the section SHALL cite the original DQN paper and relevant follow-up work
- **AND** the section SHALL connect stability to reliable anomaly detection performance

#### Scenario: PER for anomaly detection

- **WHEN** Section 2.3 describes Priority Experience Replay
- **THEN** the section SHALL explain why PER is especially important for anomaly detection
- **AND** the section SHALL emphasize anomaly samples are sparse but high-value
- **AND** the section SHALL explain how uniform sampling would drown out rare anomalies
- **AND** the section SHALL describe how prioritizing high TD-error samples ensures learning from critical events
- **AND** the section SHALL cite the PER paper and applications to imbalanced learning

## ADDED Requirements

### Requirement: Citation Completeness

The manuscript SHALL include comprehensive citations for all key concepts, demonstrating familiarity with the state-of-the-art.

#### Scenario: Key paper inclusion

- **WHEN** the reference list is finalized
- **THEN** it SHALL include the following mandatory references:
  - Golchin & Rekabdar RLAD paper (arXiv:2504.02999)
  - "Deep Learning for Time Series Anomaly Detection: A Survey" (arXiv:2211.05244)
  - At least one classic RL survey or textbook
  - At least one active learning survey
  - Original DQN paper (Mnih et al., Nature 2015)
  - Original PER paper (Schaul et al., ICLR 2016)
  - At least one XAI survey paper (for future work)
  - Industrial safety / predictive maintenance industry reports
- **AND** all references SHALL be in IEEE format
- **AND** all references SHALL be complete (all authors, full title, venue, year, DOI/URL)

#### Scenario: Citation integration

- **WHEN** concepts are introduced in the text
- **THEN** each concept SHALL have appropriate citations:
  - Method descriptions: cite original papers
  - Problem statements: cite motivating literature
  - Comparisons: cite the works being compared
  - Claims: cite supporting evidence
- **AND** citations SHALL be integrated naturally into the narrative flow
- **AND** over-citation SHALL be avoided (typically 1-3 citations per claim)
