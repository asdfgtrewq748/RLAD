## ADDED Requirements

### Requirement: IEEE Citation Format Compliance

The manuscript SHALL use IEEE citation format exclusively for all in-text citations and references.

#### Scenario: In-text citation format

- **WHEN** citing sources in the text
- **THEN** all citations SHALL use square bracket format: [1], [2-4], [5], [6-8]
- **AND** citations SHALL NOT use other formats (author-year, footnotes, etc.)
- **AND** multiple citations SHALL be combined in sequential order: [1, 3, 5] or [1-3, 5]
- **AND** citation numbers SHALL correspond to the reference list order
- **AND** each reference SHALL have a unique number (use [6a, 6b] if necessary)

#### Scenario: Reference list format

- **WHEN** the reference list is presented
- **THEN** all references SHALL follow IEEE format examples:
  - Journal paper: A. Author, "Title of paper," Abbrev. Title of Journal, vol. x, no. x, pp. xxx-xxx, Abbrev. Month, Year.
  - Conference paper: A. Author, "Title of paper," in Abbrev. Title of Conf., City of Conf., Country (optional), Year, pp. xxx-xxx.
  - ArXiv preprint: A. Author, "Title of paper," arXiv:xxxx.xxxxx, Year.
- **AND** all references SHALL include: authors, full title, source, year, DOI/URL
- **AND** author names SHALL be formatted: Initial. Surname
- **AND** paper titles SHALL be in quotation marks
- **AND** publication names SHALL be in italics
- **AND** references SHALL be numbered sequentially in order of citation

#### Scenario: Citation completeness verification

- **WHEN** the manuscript is finalized
- **THEN** all in-text citations SHALL have corresponding entries in the reference list
- **AND** all reference list entries SHALL be cited in the text
- **AND** all reference information SHALL be complete and accurate
- **AND** DOIs SHALL be included where available
- **AND** URLs SHALL be included for arXiv preprints and online resources

### Requirement: IEEE TII Template Compliance

The manuscript SHALL use the official IEEE Transactions on Industrial Informatics template and comply with all formatting requirements.

#### Scenario: Template application

- **WHEN** the manuscript is formatted
- **THEN** the latest IEEE TII LaTeX or Word template SHALL be used
- **AND** all template placeholder text SHALL be replaced with actual content
- **AND** template formatting SHALL NOT be altered (margins, fonts, spacing)
- **AND** the template SHALL be the most recent version from IEEE TII website

#### Scenario: Section formatting

- **WHEN** sections are formatted
- **THEN** section headings SHALL follow template hierarchy:
  - Level 1: 1. Introduction (centered, bold)
  - Level 2: 1.1. Problem Statement (left-aligned, bold)
  - Level 3: 1.1.1. Background (left-aligned, bold, italic)
- **AND** section numbering SHALL be sequential and correct
- **AND** all sections SHALL be properly indented and spaced according to template

#### Scenario: Equation formatting

- **WHEN** equations are included
- **THEN** all equations SHALL be numbered sequentially: (1), (2), (3)
- **AND** equation numbers SHALL be right-aligned
- **AND** equations SHALL be centered
- **AND** all mathematical notation SHALL be clear and consistent
- **AND** variables SHALL be defined in text or in equation captions

### Requirement: Figure and Table Formatting

The manuscript SHALL format all figures and tables according to IEEE standards.

#### Scenario: Figure formatting

- **WHEN** figures are included in the manuscript
- **THEN** all figures SHALL have minimum 300 DPI resolution
- **THEN** figure captions SHALL be placed below each figure
- **AND** figures SHALL be numbered sequentially with Arabic numerals: Figure 1, Figure 2, Figure 3
- **AND** figure captions SHALL be descriptive and self-contained
- **AND** all text in figures SHALL be readable at publication size
- **AND** figures SHALL be clear and professional quality
- **AND** color schemes SHALL be colorblind-friendly if color is used

#### Scenario: Table formatting

- **WHEN** tables are included in the manuscript
- **THEN** table captions SHALL be placed above each table
- **AND** tables SHALL be numbered sequentially with Roman numerals: Table I, Table II, Table III
- **AND** table captions SHALL be descriptive and explain the table content
- **AND** tables SHALL be formatted cleanly with clear borders
- **AND** tables SHALL fit within page margins
- **AND** table text SHALL be readable and properly sized
- **AND** tables SHALL use consistent formatting throughout

#### Scenario: Figure and table references

- **WHEN** figures and tables are mentioned in text
- **THEN** each figure/table SHALL be referenced at least once in the text
- **AND** references SHALL use the format: "Fig. 1" or "Figure 1", "Table I" or "Table I"
- **AND** references SHALL be placed before the figure/table appears or in the same location
- **AND** references SHALL clearly indicate what the reader should look for

### Requirement: Blind Review Compliance

The manuscript submitted for review SHALL remove all identifying information to comply with IEEE TII's double-blind review policy.

#### Scenario: Author information removal

- **WHEN** the blind review version is prepared
- **THEN** all author names SHALL be removed from the manuscript
- **AND** all institutional affiliations SHALL be removed
- **AND** acknowledgments sections SHALL be removed
- **AND** any identifying information in footnotes SHALL be removed
- **AND** author bio sections SHALL be removed
- **AND** any mentions of "our previous work" SHALL be rephrased to third person

#### Scenario: Metadata anonymization

- **WHEN** the manuscript file is prepared for submission
- **THEN** file properties SHALL be checked for author information
- **AND** document metadata (author, organization) SHALL be removed
- **AND** file names SHALL NOT contain author names (use manuscript title instead)
- **AND** no hidden text or comments with identifying information SHALL be present

#### Scenario: Separate author information

- **WHEN** submitting to IEEE TII
- **THEN** author information SHALL be provided separately in the submission system
- **AND** a cover letter SHALL be prepared separately (not in the manuscript file)
- **AND** author information SHALL be complete and accurate in the submission form
- **AND** corresponding author contact information SHALL be clearly provided

### Requirement: Cover Letter Quality

The cover letter SHALL emphasize the core contributions and industrial relevance of Enhanced RLAD.

#### Scenario: Cover letter content

- **WHEN** the cover letter is written
- **THEN** it SHALL emphasize the two core innovations prominently
- **AND** it SHALL explain the industrial relevance (hydraulic support monitoring, mining safety)
- **AND** it SHALL clearly differentiate from existing RLAD work
- **AND** it SHALL highlight the novelty (domain-aware asymmetric reward, ensemble bootstrapping)
- **AND** it SHALL be concise (1 page maximum)
- **AND** it SHALL use professional, formal tone
- **AND** it SHALL address the editor by name if known

#### Scenario: Cover letter structure

- **WHEN** the cover letter is formatted
- **THEN** it SHALL follow standard business letter format
- **AND** it SHALL include: date, recipient (editor name, journal title), salutation, body, closing, signature
- **AND** the opening SHALL state the manuscript title and submission intent
- **AND** the body SHALL present the key contributions and fit for the journal
- **AND** the closing SHALL express appreciation and willingness to revise
- **AND** author information SHALL be included in the signature block

### Requirement: Submission Package Completeness

All required files for IEEE TII submission SHALL be prepared and checked for completeness.

#### Scenario: Required files checklist

- **WHEN** preparing the submission package
- **THEN** the following files SHALL be prepared:
  - Main manuscript file (blind review version, PDF or source)
  - Cover letter (separate file)
  - Figures (if separate files are required)
  - Tables (if separate files are required)
  - Supplementary materials (if any)
- **AND** all files SHALL be named clearly and appropriately
- **AND** all files SHALL meet size requirements
- **AND** all files SHALL be in acceptable formats

#### Scenario: Supplementary materials

- **WHEN** supplementary materials are included
- **THEN** they SHALL be clearly labeled as "Supplementary Materials"
- **AND** they SHALL include:
  - Dataset characterization document
  - Additional experimental results (if applicable)
  - Reproducibility information
  - Code repository link
- **AND** the main manuscript SHALL reference the supplementary materials appropriately
- **AND** the supplementary materials SHALL be well-organized and easy to navigate
