import type {SerializedGame} from './engine';

export interface NormalizationOptions {
  /**
   * When comparing runs whose simultaneous research submission order was alternated,
   * log messages reflect the order in which players clicked Submit.
   */
  ignoreGameLog?: boolean;
}

/**
 * Strips wall-clock, unseeded metadata, and DB save-counter fields from a SerializedGame
 * so deterministic gameplay state can be compared strictly across runs.
 *
 * Stripped / normalized fields:
 * - `createdTimeMs`: wall-clock Date.now() captured in Game.newInstance()
 * - `lastSaveId`: database save counter
 * - `id`: caller-supplied game ID (normalized to '<GAME_ID>')
 * - `name`: unseeded random game name generated via UnseededRandom.INSTANCE
 * - `spectatorId`: caller-supplied spectator ID (normalized to '<SPECTATOR_ID>')
 * - `gameLog[*].timestamp`: wall-clock Date.now() captured in LogMessage constructor
 * - `gameLog[*].data[*].value`: occurrences of the raw `game.id` string (logged at Phase.END in Game.gotoEndGame) normalized to '<GAME_ID>'
 * - `players[*].timer`: wall-clock Timer state (startedAt, sumElapsed, lastStoppedAt, running, afterFirstAction)
 */
export function normalizeSerializedGame(
  raw: SerializedGame,
  opts: NormalizationOptions = {},
): unknown {
  // Deep JSON round-trip first so live shared references / Sets / class instances become plain JSON values.
  const copy = JSON.parse(JSON.stringify(raw)) as Record<string, unknown>;
  const rawGameId = typeof copy.id === 'string' ? copy.id : undefined;

  delete copy.createdTimeMs;
  delete copy.lastSaveId;
  copy.id = '<GAME_ID>';
  copy.name = '<GAME_NAME>';
  copy.spectatorId = '<SPECTATOR_ID>';

  if (opts.ignoreGameLog) {
    delete copy.gameLog;
  } else if (Array.isArray(copy.gameLog)) {
    copy.gameLog = copy.gameLog.map((entry) => {
      if (entry && typeof entry === 'object') {
        const clone = {...(entry as Record<string, unknown>)};
        delete clone.timestamp;
        if (rawGameId !== undefined && Array.isArray(clone.data)) {
          clone.data = clone.data.map((item) => {
            if (item && typeof item === 'object') {
              const d = item as Record<string, unknown>;
              if (d.value === rawGameId) {
                return {...d, value: '<GAME_ID>'};
              }
            }
            return item;
          });
        }
        return clone;
      }
      return entry;
    });
  }

  if (Array.isArray(copy.players)) {
    copy.players = copy.players.map((player) => {
      if (player && typeof player === 'object') {
        const clone = {...(player as Record<string, unknown>)};
        delete clone.timer;
        return clone;
      }
      return player;
    });
  }

  return copy;
}

/**
 * Finds the first JSON path where two normalized states differ, or returns null if equal.
 */
export function findFirstDiff(a: unknown, b: unknown, path = '$'): string | null {
  if (Object.is(a, b)) {
    return null;
  }
  if (typeof a !== typeof b || a === null || b === null) {
    return `${path}: ${JSON.stringify(a)} !== ${JSON.stringify(b)}`;
  }
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b)) {
      return `${path}: array mismatch`;
    }
    const minLen = Math.min(a.length, b.length);
    for (let i = 0; i < minLen; i++) {
      const diff = findFirstDiff(a[i], b[i], `${path}[${i}]`);
      if (diff !== null) {
        return diff;
      }
    }
    if (a.length !== b.length) {
      return `${path}.length: ${a.length} !== ${b.length}`;
    }
    return null;
  }
  if (typeof a === 'object' && typeof b === 'object') {
    const objA = a as Record<string, unknown>;
    const objB = b as Record<string, unknown>;
    const keys = Array.from(new Set([...Object.keys(objA), ...Object.keys(objB)])).sort();
    for (const key of keys) {
      if (!(key in objA)) {
        return `${path}.${key}: missing in A (B=${JSON.stringify(objB[key])})`;
      }
      if (!(key in objB)) {
        return `${path}.${key}: missing in B (A=${JSON.stringify(objA[key])})`;
      }
      const diff = findFirstDiff(objA[key], objB[key], `${path}.${key}`);
      if (diff !== null) {
        return diff;
      }
    }
    return null;
  }
  return `${path}: ${String(a)} !== ${String(b)}`;
}
