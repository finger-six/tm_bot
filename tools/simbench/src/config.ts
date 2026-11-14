import {
  BoardName,
  BoardNameValue,
  Game,
  GameId,
  GameOptions,
  IGame,
  IPlayer,
  Player,
  PlayerId,
  RandomMAOptionType,
  SpectatorId,
  constants,
  initializeEngine,
} from './engine';

export const TARGET_MAPS: ReadonlyArray<BoardNameValue> = [
  BoardName.THARSIS,
  BoardName.ELYSIUM,
  BoardName.HELLAS,
  BoardName.VASTITAS_BOREALIS_NOVA,
];

/**
 * Builds explicit GameOptions with zero reliance on engine defaults.
 * Keeps legacy extension booleans and the `expansions` record strictly consistent.
 */
export function buildExplicitGameOptions(boardName: BoardNameValue): GameOptions {
  return {
    boardName,
    clonedGamedId: undefined,

    // Configuration
    undoOption: false,
    showTimers: false,
    fastModeOption: false,
    showOtherPlayersVP: false,

    // Legacy expansion booleans
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

export interface CreateGameParams {
  boardName: BoardNameValue;
  seed: number;
  gameId?: GameId;
  spectatorId?: SpectatorId;
  firstSeatIndex?: 0 | 1;
}

/**
 * Creates a fully initialized 2-player game with explicit seats, options, seed, and first player.
 *
 * Hook for Stages 3-5: snapshot/clone entry points can accept the returned `IGame` directly
 * via `game.serialize()` / `Game.deserialize()`.
 */
export function createExplicitTwoPlayerGame(params: CreateGameParams): {
  game: IGame;
  players: [IPlayer, IPlayer];
  options: GameOptions;
} {
  initializeEngine();

  const gameId: GameId = params.gameId ?? (`g-sim-${params.seed}` as GameId);
  const spectatorId: SpectatorId = params.spectatorId ?? (`s-sim-${params.seed}` as SpectatorId);
  const p1 = new Player('Seat 1', 'red', false, 0, 'p-1' as PlayerId);
  const p2 = new Player('Seat 2', 'blue', false, 0, 'p-2' as PlayerId);
  const players: [IPlayer, IPlayer] = [p1, p2];
  const firstPlayer = players[params.firstSeatIndex ?? 0];
  const options = buildExplicitGameOptions(params.boardName);

  const game = Game.newInstance(gameId, players, firstPlayer, spectatorId, options, params.seed);
  return {game, players, options};
}
