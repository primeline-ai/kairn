-- Kairn Per-Workspace Schema (ws_{id}.db)
-- Version: 1.0

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Nodes (knowledge graph)
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'knowledge',
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    properties JSON,
    tags JSON,
    created_by TEXT,
    visibility TEXT DEFAULT 'workspace',
    source_type TEXT,
    source_ref TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    deleted_at TEXT
);

-- Edges (relationships)
CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT NOT NULL REFERENCES nodes(id),
    target_id TEXT NOT NULL REFERENCES nodes(id),
    type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    properties JSON,
    created_by TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, type)
);

-- Experiences (temporal, decaying)
CREATE TABLE IF NOT EXISTS experiences (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'knowledge',
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    context TEXT,
    confidence TEXT DEFAULT 'high',
    score REAL NOT NULL DEFAULT 1.0,
    decay_rate REAL NOT NULL,
    tags JSON,
    properties JSON,
    created_by TEXT,
    access_count INTEGER DEFAULT 0,
    promoted_to_node_id TEXT REFERENCES nodes(id),
    created_at TEXT NOT NULL,
    last_accessed TEXT
);

-- Projects
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    phase TEXT DEFAULT 'planning',
    goals JSON,
    active BOOLEAN DEFAULT FALSE,
    created_by TEXT,
    stakeholders JSON,
    success_metrics JSON,
    created_at TEXT NOT NULL,
    updated_at TEXT
);

-- Progress/Failure Log
CREATE TABLE IF NOT EXISTS progress (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id),
    type TEXT NOT NULL,
    action TEXT NOT NULL,
    result TEXT,
    next_step TEXT,
    created_by TEXT,
    created_at TEXT NOT NULL
);

-- Ideas
CREATE TABLE IF NOT EXISTS ideas (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT DEFAULT 'draft',
    category TEXT,
    score REAL,
    properties JSON,
    created_by TEXT,
    visibility TEXT DEFAULT 'private',
    created_at TEXT NOT NULL,
    updated_at TEXT
);

-- Context Router
CREATE TABLE IF NOT EXISTS routes (
    keyword TEXT NOT NULL PRIMARY KEY,
    node_ids JSON NOT NULL,
    confidence REAL NOT NULL
);

-- Activity Log
CREATE TABLE IF NOT EXISTS activity_log (
    id TEXT PRIMARY KEY,
    user_id TEXT,
    activity_type TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_nodes_ns_type ON nodes(namespace, type) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_nodes_created_by ON nodes(created_by) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_nodes_visibility ON nodes(visibility) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_experiences_type ON experiences(type);
CREATE INDEX IF NOT EXISTS idx_experiences_score ON experiences(score);
CREATE INDEX IF NOT EXISTS idx_progress_project ON progress(project_id);
CREATE INDEX IF NOT EXISTS idx_activity_time ON activity_log(created_at DESC);
