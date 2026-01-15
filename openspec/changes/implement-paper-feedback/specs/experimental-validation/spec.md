## ADDED Requirements

### Requirement: Dataset Characterization

The manuscript SHALL provide comprehensive characterization of the private dataset to enable reproducibility assessment and comparison with public benchmarks.

#### Scenario: Dataset description completeness

- **WHEN** Section 4.1 describes the experimental dataset
- **THEN** the section SHALL include a new subsection (4.1.1) titled "Dataset Characterization"
- **AND** the section SHALL document the following statistics:
  - Total duration of data collection
  - Total number of anomaly events
  - Average duration of anomaly events
  - Sampling frequency
  - Number of normal vs. anomalous windows
- **AND** the section SHALL describe the types of anomalies present:
  - Sudden pressure spikes (equipment malfunction)
  - Gradual pressure drift (wear and tear)
  - Abnormal cycles (operational issues)
- **AND** the section SHALL provide summary statistics (mean, std, min, max) for key features

#### Scenario: Benchmark dataset comparison

- **WHEN** Section 4.1.1 presents the dataset
- **THEN** the section SHALL compare the dataset to established public benchmarks:
  - Numenta Anomaly Benchmark (NAB): specifically "machine_temperature_system_failure.csv"
  - Server Machine Dataset (SMD): relevant machine subset
- **AND** the section SHALL explain similarities in anomaly patterns
- **AND** the section SHALL state: "Like [NAB dataset], our data contains anomaly patterns that are precursors to catastrophic equipment failure"
- **AND** the section SHALL emphasize the scientific problem is universal despite private data

#### Scenario: Sensor and collection methodology

- **WHEN** the dataset is described
- **THEN** the section SHALL document:
  - Sensor type and specifications (pressure sensors)
  - Installation location (hydraulic support)
  - Data collection protocol
  - Any preprocessing applied
- **AND** the section SHALL enable readers to understand the data's physical context

### Requirement: Experimental Rigor Enhancement

The manuscript SHALL strengthen experimental validation through clearer presentation of results and stronger ablation study conclusions.

#### Scenario: Performance interpretation

- **WHEN** Section 4.2 presents quantitative performance results
- **THEN** the section SHALL explicitly connect high recall (0.915) to the asymmetric reward function design
- **AND** the section SHALL state this is an expected outcome of the risk-averse optimization
- **AND** the section SHALL compare results to baseline methods with statistical significance tests if possible
- **AND** the section SHALL highlight that no manual threshold tuning was required (practical advantage)

#### Scenario: Ablation study conclusions

- **WHEN** Section 4.4 presents ablation study results
- **THEN** each ablation experiment SHALL have an explicit conclusion
- **AND** the section SHALL identify which component removal caused the largest performance drop
- **AND** the section SHALL state: "Removing the asymmetric reward function caused F1 to drop from 0.933 to 0.871, the largest decrease"
- **AND** the section SHALL conclude: "This confirms encoding domain risk knowledge into the reward function is critical for safety-oriented detection"
- **AND** the section SHALL draw similar explicit conclusions for other ablations

### Requirement: Reproducibility Documentation

The manuscript SHALL provide sufficient documentation to enable other researchers to understand, replicate, and build upon the experimental results.

#### Scenario: Experimental protocol documentation

- **WHEN** experimental setup is described
- **THEN** the section SHALL document:
  - Train/validation/test split proportions
  - Random seed values used for reproducibility
  - Hyperparameter values for all algorithms
  - Hardware specifications (CPU/GPU, memory)
  - Software versions (Python, PyTorch, etc.)
  - Training duration and convergence criteria
- **AND** the documentation SHALL be sufficient to reproduce the experiments

#### Scenario: Code availability documentation

- **WHEN** reproducibility is discussed
- **THEN** the manuscript SHALL reference the code repository location
- **AND** the repository SHALL include:
  - README with installation instructions
  - Usage examples with expected outputs
  - Citation information for the paper
- **AND** the manuscript SHALL mention any dependencies or requirements
- **AND** the manuscript SHALL state whether the full dataset or samples are available

## ADDED Requirements

### Requirement: Comparative Analysis Table

The manuscript SHALL include a comprehensive comparison table that clearly differentiates Enhanced RLAD from existing approaches.

#### Scenario: Comparison table structure

- **WHEN** Section 1.4 presents contributions
- **THEN** Table I SHALL be included with the following structure:
  - Rows: Different anomaly detection frameworks (Existing RLAD, Enhanced RLAD, Traditional methods, Other deep learning methods)
  - Columns: Key feature categories
    - Supervision Source (e.g., Single VAE, Multi-method Ensemble, Reconstruction Error, etc.)
    - Core Algorithm (e.g., DQN+Standard Reward, DQN+Asymmetric Reward, Threshold-based, etc.)
    - Learning Objective (e.g., Balanced Accuracy, Risk-Aware, Reconstruction Minimization, etc.)
    - Industrial Adaptation (e.g., None, Domain-Aware Reward, General Purpose, etc.)
- **AND** Enhanced RLAD's innovations SHALL be highlighted in bold
- **AND** the table SHALL use clear, concise language for each cell
- **AND** the table SHALL be referenced in the text as "Table I shows..."

#### Scenario: Table formatting compliance

- **WHEN** Table I is formatted
- **THEN** the table caption SHALL be placed above the table (IEEE standard)
- **AND** the table SHALL use Roman numeral numbering (Table I)
- **AND** the caption SHALL be descriptive: "Table I: Comparison of Anomaly Detection Frameworks"
- **AND** the table SHALL be readable and clear at publication size
- **AND** the table SHALL fit within page margins
