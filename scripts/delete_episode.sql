-- SQL script to delete all data related to a specific episode
-- Usage in DBeaver: 
--   1. Replace :episode_number with the actual episode number (e.g., 522)
--   2. Or DBeaver will prompt you to enter the value when you execute
-- 
-- Example: Change :episode_number to 522, or leave it and enter 522 when prompted

-- Begin transaction
BEGIN;

-- Step 1: Delete search_results that reference sentences from this episode
-- (search_results.sentence_id references sentences.id)
DELETE FROM search_results
WHERE sentence_id IN (
    SELECT s.id
    FROM sentences s
    INNER JOIN segments seg ON s.segment_id = seg.id
    INNER JOIN episodes e ON seg.episode_id = e.id
    WHERE e.episode_number = :episode_number
);

-- Step 2: Delete sentences that belong to segments from this episode
DELETE FROM sentences
WHERE segment_id IN (
    SELECT seg.id
    FROM segments seg
    INNER JOIN episodes e ON seg.episode_id = e.id
    WHERE e.episode_number = :episode_number
);

-- Step 3: Delete segments that belong to this episode
DELETE FROM segments
WHERE episode_id IN (
    SELECT e.id
    FROM episodes e
    WHERE e.episode_number = :episode_number
);

-- Step 4: Delete speaker_episode links for this episode
DELETE FROM speaker_episode
WHERE episode_id IN (
    SELECT e.id
    FROM episodes e
    WHERE e.episode_number = :episode_number
);

-- Step 5: Delete the episode itself
DELETE FROM episodes
WHERE episode_number = :episode_number;

-- Commit transaction
COMMIT;

-- Summary: All data for episode :episode_number has been deleted

