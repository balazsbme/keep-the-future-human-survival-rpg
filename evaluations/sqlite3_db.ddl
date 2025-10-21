PRAGMA foreign_keys = ON;

-- 1) Game execution info (one row per run)
CREATE TABLE IF NOT EXISTS executions (
  execution_id                    INTEGER PRIMARY KEY,
  player_class                    TEXT,
  automated_player_class          TEXT,
  config_json                     TEXT,
  log_level                       TEXT,
  enable_parallelism              TEXT,
  automated_agent_max_exchanges   INTEGER,
  scenario                        TEXT,
  win_threshold                   INTEGER,
  max_rounds                      INTEGER,
  roll_success_threshold          INTEGER,
  notes                           TEXT,
  created_at                      TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 2) Actions recorded for an execution
CREATE TABLE IF NOT EXISTS actions (
  action_id             INTEGER PRIMARY KEY,
  execution_id          INTEGER NOT NULL,
  actor                 TEXT,
  title                 TEXT,
  option_text           TEXT,
  option_type           TEXT,
  related_triplet       INTEGER,
  related_attribute     TEXT,
  success               INTEGER NOT NULL,
  roll_total            INTEGER,
  actor_score           INTEGER,
  player_score          INTEGER,
  effective_score       INTEGER,
  credibility_cost      INTEGER,
  credibility_gain      INTEGER,
  targets_json          TEXT,
  failure_text          TEXT,
  round_number          INTEGER,
  option_json           TEXT,
  created_at            TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);

-- 3) Assessments recorded after each action
CREATE TABLE IF NOT EXISTS assessments (
  assessment_id         INTEGER PRIMARY KEY,
  execution_id          INTEGER NOT NULL,
  action_id             INTEGER NOT NULL,
  scenario              TEXT,
  final_weighted_score  INTEGER,
  assessment_json       TEXT,
  FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE,
  FOREIGN KEY (action_id)    REFERENCES actions(action_id)       ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_assessments_exec_action
  ON assessments(execution_id, action_id);

-- 4) Credibility snapshots per action
CREATE TABLE IF NOT EXISTS credibility (
  credibility_vector_id INTEGER PRIMARY KEY,
  execution_id          INTEGER NOT NULL,
  action_id             INTEGER NOT NULL,
  cost                  INTEGER NOT NULL,
  reroll_attempt_count  INTEGER NOT NULL,
  credibility_json      TEXT,
  FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE,
  FOREIGN KEY (action_id)    REFERENCES actions(action_id)       ON DELETE CASCADE
);

-- 5) Results recorded per execution
CREATE TABLE IF NOT EXISTS results (
  execution_id          INTEGER PRIMARY KEY,
  successful_execution  INTEGER NOT NULL,
  result                TEXT,
  error_info            TEXT,
  created_at            TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);
