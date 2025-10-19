PRAGMA foreign_keys = ON;

-- 1) Game execution info (one row per run)
CREATE TABLE IF NOT EXISTS executions (
  execution_id        INTEGER PRIMARY KEY,
  player_class        TEXT,          -- starting class/archetype
  automated_player_class        TEXT,          -- Class name of the players.py, e.g. GeminiCorporationPlayer
  config_json     TEXT,          -- full config snapshot
  LOG_LEVEL     TEXT,
  ENABLE_PARALLELISM  TEXT,
  AUTOMATED_AGENT_MAX_EXCHANGES INTEGER,
  scenario    TEXT,
  win_threshold     INTEGER,
  max_rounds      INTEGER,
  roll_success_threshold      INTEGER,
  notes               TEXT           -- optional tag/cohort label
);

-- 2) Actions 
CREATE TABLE IF NOT EXISTS actions (
  action_id           INTEGER PRIMARY KEY,
  execution_id        INTEGER NOT NULL,              -- FK → executions
  actor               TEXT,                          -- 'player' or NPC/faction id
  title               TEXT,                          -- short label shown to player
  # TODO for Codex: add all field of OptionResponse class
  option_json     TEXT,                          -- full option payload
  FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);

-- 3) Assessments 
CREATE TABLE IF NOT EXISTS assessments (
  assessment_id       INTEGER PRIMARY KEY,
  execution_id                INTEGER NOT NULL,      -- FK → executions
  action_id                   INTEGER NOT NULL,      -- FK → actions
  scenario    TEXT,
  # TODO for Codex: dynamically create the columns for each triplet of each faction, present in the scenario yaml.
  final_weighted_score  INTEGER,
  FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE,
  FOREIGN KEY (action_id)    REFERENCES actions(action_id)       ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_outcomes_exec_action ON outcomes(execution_id, action_id);

-- 4) Credibility
CREATE TABLE IF NOT EXISTS credibility (
  credibility_vector_id       INTEGER PRIMARY KEY,
  execution_id                INTEGER NOT NULL,      -- FK → executions
  action_id                   INTEGER NOT NULL,      -- FK → actions
  cost                   INTEGER NOT NULL,
  reroll_attempt_count    INTEGER NOT NULL, -- start from 0 as an action without reroll already costs
  # TODO for Codex: add one column for each relevant credibility matrix element. (e.g. the row of the CivilSociety player). 
);