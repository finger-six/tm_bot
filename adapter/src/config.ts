import crypto from 'node:crypto';
import {
  BoardNameValue,
  GameOptions,
  RandomMAOptionType,
  constants,
} from './engine';
import {CreateGameOptions, SUPPORTED_MAPS, SupportedMap} from './types';

const SUPPORTED_MAP_SET: ReadonlySet<string> = new Set(SUPPORTED_MAPS);

/**
 * Hashes an arbitrary seedKey string deterministically into a normalized float in [0, 1)
 * that preserves all 32 bits inside the engine's SeededRandom (Math.floor(seed * 2^32)).
 */
export function hashSeedKey(seedKey: string): {seed: number; seedHex: string} {
  const digest = crypto.createHash('sha256').update(seedKey, 'utf8').digest();
  const u32 = digest.readUInt32BE(0);
  return {
    seed: u32 / 4294967296,
    seedHex: u32.toString(16).padStart(8, '0'),
  };
}

export function validateCreateGameOptions(raw: CreateGameOptions): {seedKey: string; map: SupportedMap} {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('createGame requires an options object { seedKey, map }');
  }

  for (const key of Object.keys(raw)) {
    if (key !== 'seedKey' && key !== 'map') {
      throw new Error(
        `Unsupported option "${key}": training format options are fixed and cannot be overridden`,
      );
    }
  }

  const {seedKey, map} = raw;
  if (typeof seedKey !== 'string' || seedKey.length === 0) {
    throw new Error('Invalid seedKey: must be a non-empty string');
  }

  if (typeof map !== 'string' || !SUPPORTED_MAP_SET.has(map)) {
    throw new Error(
      `Unsupported map "${String(map)}". Supported maps are: ${SUPPORTED_MAPS.join(', ')}`,
    );
  }

  return {seedKey, map: map as SupportedMap};
}

export function buildTrainingGameOptions(boardName: BoardNameValue): GameOptions {
  return {
    boardName,
    clonedGamedId: undefined,

    // Configuration
    undoOption: false,
    showTimers: false,
    fastModeOption: false,
    showOtherPlayersVP: false,

    // Legacy expansion booleans: Corporate Era + Prelude only
    corporateEra: true,
    venusNextExtension: false,
    coloniesExtension: false,
    preludeExtension: true,
    prelude2Expansion: false,
    turmoilExtension: false,
    promoCardsOption: false,
    communityCardsOption: false,
    aresExtension: false,
    aresHazards: false,
    aresExtremeVariant: false,
    politicalAgendasExtension: 'Standard',
    solarPhaseOption: false,
    removeNegativeGlobalEventsOption: false,
    moonExpansion: false,
    pathfindersExpansion: false,
    ceoExtension: false,
    starWarsExpansion: false,
    underworldExpansion: false,
    deltaProjectExpansion: false,

    // Expansions record (strictly aligned with legacy booleans above)
    expansions: {
      corpera: true,
      promo: false,
      venus: false,
      colonies: false,
      prelude: true,
      prelude2: false,
      turmoil: false,
      community: false,
      ares: false,
      moon: false,
      pathfinders: false,
      ceo: false,
      starwars: false,
      underworld: false,
      deltaProject: false,
    },

    // Variants
    draftVariant: true,
    initialDraftVariant: false,
    preludeDraftVariant: false,
    ceosDraftVariant: false,
    startingCorporations: constants.CORPORATION_CARDS_DEALT_PER_PLAYER,
    shuffleMapOption: false,
    randomMA: RandomMAOptionType.NONE,
    includeFanMA: false,
    modularMA: false,
    soloTR: false,
    customCorporationsList: [],
    bannedCards: [],
    includedCards: [],
    customColoniesList: [],
    customPreludes: [],
    customCeos: [],
    startingCeos: constants.CEO_CARDS_DEALT_PER_PLAYER,
    startingPreludes: constants.PRELUDE_CARDS_DEALT_PER_PLAYER,
    requiresMoonTrackCompletion: false,
    requiresVenusTrackCompletion: false,
    moonStandardProjectVariant: false,
    moonStandardProjectVariant1: false,
    altVenusBoard: false,
    escapeVelocity: undefined,
    twoCorpsVariant: false,
  };
}
