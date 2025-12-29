PRAGMA foreign_keys = ON;

-- 1) Game execution info (one row per run)
CREATE TABLE IF NOT EXISTS executions (
  execution_id                    TEXT PRIMARY KEY,
  session_id                      TEXT,
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
  action_time_cost_years          REAL,
  format_prompt_character_limit   INTEGER,
  conversation_force_action_after INTEGER,
  log_filename                    TEXT,
  notes                           TEXT,
  created_at                      TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 2) Actions recorded for an execution
CREATE TABLE IF NOT EXISTS actions (
  action_id             TEXT PRIMARY KEY,
  execution_id          TEXT NOT NULL,
  session_id            TEXT,
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
  assessment_id         TEXT PRIMARY KEY,
  execution_id          TEXT NOT NULL,
  action_id             TEXT NOT NULL,
  session_id            TEXT,
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
  credibility_vector_id TEXT PRIMARY KEY,
  execution_id          TEXT NOT NULL,
  action_id             TEXT NOT NULL,
  session_id            TEXT,
  cost                  INTEGER NOT NULL,
  reroll_attempt_count  INTEGER NOT NULL,
  credibility_json      TEXT,
  FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE,
  FOREIGN KEY (action_id)    REFERENCES actions(action_id)       ON DELETE CASCADE
);

-- 5) Results recorded per execution
CREATE TABLE IF NOT EXISTS results (
  execution_id          TEXT PRIMARY KEY,
  session_id            TEXT,
  successful_execution  INTEGER NOT NULL,
  result                TEXT,
  error_info            TEXT,
  log_warning_count     INTEGER DEFAULT 0,
  log_error_count       INTEGER DEFAULT 0,
  created_at            TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (execution_id) REFERENCES executions(execution_id) ON DELETE CASCADE
);
