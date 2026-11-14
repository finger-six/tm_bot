import path from 'node:path';

export type GameId = `g${string}`;
export type PlayerId = `p${string}`;
export type SpectatorId = `s${string}`;
export type SpaceId = `${string}`;
export type CardName = string;
export type Color = 'red' | 'green' | 'yellow' | 'blue' | 'black' | 'purple' | 'orange' | 'pink' | 'neutral' | 'bronze';

export type BoardNameValue =
  | 'tharsis'
  | 'hellas'
  | 'elysium'
  | 'utopia planitia'
  | 'vastitas borealis nova'
  | 'terra cimmeria nova'
  | 'arabia terra'
  | 'vastitas borealis'
  | 'amazonis p.'
  | 't. cimmeria'
  | 'Hollandia';

export type ExpansionKey =
  | 'corpera'
  | 'promo'
  | 'venus'
  | 'colonies'
  | 'prelude'
  | 'prelude2'
  | 'turmoil'
  | 'community'
  | 'ares'
  | 'moon'
  | 'pathfinders'
  | 'ceo'
  | 'starwars'
  | 'underworld'
  | 'deltaProject';

export type GameOptions = {
  boardName: BoardNameValue;
  clonedGamedId: GameId | undefined;
  undoOption: boolean;
  showTimers: boolean;
  fastModeOption: boolean;
  showOtherPlayersVP: boolean;
  corporateEra: boolean;
  venusNextExtension: boolean;
  coloniesExtension: boolean;
  preludeExtension: boolean;
  prelude2Expansion: boolean;
  turmoilExtension: boolean;
  promoCardsOption: boolean;
  communityCardsOption: boolean;
  aresExtension: boolean;
  aresHazards: boolean;
  aresExtremeVariant: boolean;
  politicalAgendasExtension: 'Standard' | 'Random' | 'Chairman';
  solarPhaseOption: boolean;
  removeNegativeGlobalEventsOption: boolean;
  moonExpansion: boolean;
  pathfindersExpansion: boolean;
  ceoExtension: boolean;
  starWarsExpansion: boolean;
  underworldExpansion: boolean;
  deltaProjectExpansion: boolean;
  expansions: Record<ExpansionKey, boolean>;
  draftVariant: boolean;
  initialDraftVariant: boolean;
  preludeDraftVariant: boolean;
  ceosDraftVariant: boolean;
  startingCorporations: number;
  shuffleMapOption: boolean;
  randomMA: string;
  includeFanMA: boolean;
  modularMA: boolean;
  soloTR: boolean;
  customCorporationsList: ReadonlyArray<CardName>;
  bannedCards: ReadonlyArray<CardName>;
  includedCards: ReadonlyArray<CardName>;
  customColoniesList: ReadonlyArray<string>;
  customPreludes: ReadonlyArray<CardName>;
  customCeos: ReadonlyArray<CardName>;
  startingCeos: number;
  startingPreludes: number;
  requiresMoonTrackCompletion: boolean;
  requiresVenusTrackCompletion: boolean;
  moonStandardProjectVariant: boolean;
  moonStandardProjectVariant1: boolean;
  altVenusBoard: boolean;
  escapeVelocity?: unknown;
  twoCorpsVariant: boolean;
};

export type SpendableResource =
  | 'megacredits'
  | 'steel'
  | 'titanium'
  | 'plants'
  | 'heat'
  | 'microbes'
  | 'floaters'
  | 'lunaArchivesScience'
  | 'spireScience'
  | 'seeds'
  | 'auroraiData'
  | 'graphene'
  | 'kuiperAsteroids';

export type Payment = {[k in SpendableResource]: number};
export type PaymentOptions = Partial<Record<Exclude<SpendableResource, 'megacredits'> | 'lunaTradeFederationTitanium', boolean>>;

export type Units = {
  megacredits: number;
  steel: number;
  titanium: number;
  plants: number;
  energy: number;
  heat: number;
};

export type InputResponse =
  | {type: 'option'}
  | {type: 'or'; index: number; response: InputResponse}
  | {type: 'and'; responses: Array<InputResponse>}
  | {type: 'initialCards'; responses: Array<InputResponse>}
  | {type: 'card'; cards: Array<CardName>}
  | {type: 'projectCard'; card: CardName; payment: Payment}
  | {type: 'space'; spaceId: SpaceId}
  | {type: 'player'; player: Color}
  | {type: 'amount'; amount: number}
  | {type: 'payment'; payment: Payment}
  | {type: 'productionToLose'; units: Units}
  | {type: 'resource'; resource: keyof Units}
  | {type: 'resources'; units: Units};

export interface EngineMessage {
  data: Array<{type?: number; value: string}>;
  message: string;
}

export interface ICard {
  name: CardName;
}

export interface ICorporationCard extends ICard {
  startingMegaCredits: number;
  cardCost?: number;
}

export interface IProjectCard extends ICard {
  cost: number;
}

export interface IStandardProjectCard extends ICard {
  canAct(player: IPlayer): boolean;
  canPayWith(player: IPlayer): {
    steel?: boolean;
    titanium?: boolean;
    seeds?: boolean;
    kuiperAsteroids?: boolean;
  };
  getAdjustedCost(player: IPlayer): number;
}

export interface PlayerInput {
  type: string;
  title?: string | EngineMessage;
  buttonLabel?: string;
  optional?: boolean;
}

export interface OrOptions extends PlayerInput {
  type: 'or';
  options: Array<PlayerInput>;
}

export interface AndOptions extends PlayerInput {
  type: 'and';
  options: Array<PlayerInput>;
}

export interface SelectCard<T extends ICard = ICard> extends PlayerInput {
  type: 'card';
  cards: ReadonlyArray<T>;
  config: {
    min: number;
    max: number;
    enabled?: ReadonlyArray<boolean>;
    played?: boolean | string;
  };
}

export interface SelectInitialCards extends PlayerInput {
  type: 'initialCards';
  options: Array<PlayerInput>;
  inputs: {
    corp: PlayerInput | undefined;
    project: PlayerInput | undefined;
    prelude: PlayerInput | undefined;
    ceo: PlayerInput | undefined;
  };
}

export interface SelectCardToPlay<T extends ICard = ICard> extends PlayerInput {
  type: 'projectCard';
  cards: Array<T>;
  enabled?: ReadonlyArray<boolean>;
  extras: Map<CardName, {reserveUnits: Readonly<Units>; overriddenCost?: number}>;
}

export type SelectProjectCardToPlay = SelectCardToPlay<IProjectCard>;
export type SelectStandardProjectToPlay = SelectCardToPlay<IStandardProjectCard>;

export interface SelectPayment extends PlayerInput {
  type: 'payment';
  amount: number;
  paymentOptions: Partial<PaymentOptions>;
  reserveUnits?: Units;
}

export interface SelectSpace extends PlayerInput {
  type: 'space';
  spaces: ReadonlyArray<{id: SpaceId; spaceType?: string; bonus?: ReadonlyArray<number>}>;
}

export interface SelectAmount extends PlayerInput {
  type: 'amount';
  min: number;
  max: number;
}

export interface SelectPlayer extends PlayerInput {
  type: 'player';
  players: ReadonlyArray<IPlayer>;
}

export interface SelectProductionToLose extends PlayerInput {
  type: 'productionToLose';
  unitsToLose: number;
  pairs?: number;
  player: IPlayer;
}

export interface SelectResource extends PlayerInput {
  type: 'resource';
  include: ReadonlyArray<keyof Units>;
}

export interface SelectResources extends PlayerInput {
  type: 'resources';
  count: number;
}

export interface EngineLogMessageData {
  type?: number;
  value: string | ReadonlyArray<string>;
}

export interface EngineLogMessage {
  playerId?: PlayerId;
  timestamp: number;
  type?: number;
  message: string;
  data: ReadonlyArray<EngineLogMessageData>;
}

export interface EngineCardModel {
  name: CardName;
  resources?: number;
  calculatedCost?: number;
  isDisabled?: boolean;
}

export interface EngineSpaceModel {
  id: SpaceId;
  x: number;
  y: number;
  spaceType: string;
  bonus: ReadonlyArray<number>;
  tileType?: number;
  color?: Color;
  highlight?: 'noctis' | 'volcanic';
  rotated?: true;
}

export interface EngineMilestoneModel {
  name: string;
  playerName: string | undefined;
  color: Color | undefined;
  scores: ReadonlyArray<{
    color: Color;
    score: number;
    claimable: boolean;
  }>;
}

export interface EngineAwardModel {
  name: string;
  playerName: string | undefined;
  color: Color | undefined;
  scores: ReadonlyArray<{
    color: Color;
    score: number;
  }>;
}

export interface EnginePublicPlayerModel {
  actionsTakenThisRound: number;
  actionsThisGeneration: ReadonlyArray<CardName>;
  actionsTakenThisGame: number;
  availableBlueCardActionCount: number;
  cardCost: number;
  cardDiscount: number;
  cardsInHandNbr: number;
  citiesCount: number;
  color: Color;
  energy: number;
  energyProduction: number;
  heat: number;
  heatProduction: number;
  isActive: boolean;
  lastCardPlayed?: CardName;
  megacredits: number;
  megacreditProduction: number;
  name: string;
  noTagsCount: number;
  plants: number;
  plantProduction: number;
  protectedResources: Record<string, string>;
  protectedProduction: Record<string, string>;
  tableau: ReadonlyArray<EngineCardModel>;
  steel: number;
  steelProduction: number;
  steelValue: number;
  tags: Record<string, number>;
  terraformRating: number;
  titanium: number;
  titaniumProduction: number;
  titaniumValue: number;
}

export interface EngineGameModel {
  awards: ReadonlyArray<EngineAwardModel>;
  deckSize: number;
  discardPileSize: number;
  otherDeckSizes: {
    corporations: {drawPile: number; discardPile: number};
    preludes: {drawPile: number; discardPile: number} | undefined;
  };
  gameOptions: {
    boardName: BoardNameValue;
  };
  generation: number;
  isTerraformed: boolean;
  milestones: ReadonlyArray<EngineMilestoneModel>;
  oceans: number;
  oxygenLevel: number;
  passedPlayers: ReadonlyArray<Color>;
  phase: string;
  spaces: ReadonlyArray<EngineSpaceModel>;
  temperature: number;
}

export interface EnginePlayerViewModel {
  cardsInHand: ReadonlyArray<EngineCardModel>;
  dealtCorporationCards: ReadonlyArray<EngineCardModel>;
  dealtPreludeCards: ReadonlyArray<EngineCardModel>;
  dealtProjectCards: ReadonlyArray<EngineCardModel>;
  draftedCards: ReadonlyArray<EngineCardModel>;
  game: EngineGameModel;
  color: Color;
  pickedCorporationCard: ReadonlyArray<EngineCardModel>;
  preludeCardsInHand: ReadonlyArray<EngineCardModel>;
  thisPlayer: EnginePublicPlayerModel;
  players: ReadonlyArray<EnginePublicPlayerModel>;
}

export interface IPreludeCard extends IProjectCard {}

export interface IDeck<T extends ICard> {
  drawPile: Array<T>;
  discardPile: Array<T>;
}

export interface IPlayer {
  id: PlayerId;
  name: string;
  color: Color;
  cardCost: number;
  megaCredits: number;
  steel: number;
  titanium: number;
  plants: number;
  energy: number;
  heat: number;
  needsToDraft?: boolean;
  canUseHeatAsMegaCredits: boolean;
  canUseTitaniumAsMegacredits: boolean;
  tableau: {has(name: CardName): boolean};
  production: {asUnits(): Units};
  cardsInHand: Array<IProjectCard>;
  preludeCardsInHand: Array<IProjectCard>;
  dealtCorporationCards: Array<ICorporationCard>;
  dealtPreludeCards: Array<IPreludeCard>;
  dealtProjectCards: Array<IProjectCard>;
  pickedCorporationCard?: ICorporationCard;
  draftHand: Array<IProjectCard>;
  draftedCards: Array<IProjectCard>;
  getWaitingFor(): PlayerInput | undefined;
  setWaitingFor(input: PlayerInput, cb?: () => void): void;
  clearWaitingFor(): void;
  process(input: InputResponse): void;
  getVictoryPoints(): {total: number};
  availableHeat(): number;
  getSteelValue(): number;
  getTitaniumValue(): number;
  getSpendable(resource: SpendableResource): number;
  getCardsWithResources(resource: string): Array<ICard>;
  affordOptionsForCard(card: IProjectCard): {
    cost: number;
    reserveUnits?: Readonly<Units>;
  } & PaymentOptions;
}

export type SerializedGame = Record<string, unknown>;

export interface IGame {
  id: GameId;
  spectatorId: SpectatorId;
  phase: string;
  generation: number;
  draftRound: number;
  gameLog: ReadonlyArray<EngineLogMessage>;
  gameOptions: GameOptions;
  players: Array<IPlayer>;
  playersInGenerationOrder: Array<IPlayer>;
  activePlayer: IPlayer;
  gameAge: number;
  undoCount: number;
  deferredActions: {length: number};
  projectDeck: IDeck<IProjectCard>;
  preludeDeck: IDeck<IPreludeCard>;
  corporationDeck: IDeck<ICorporationCard>;
  researchedPlayers: Set<PlayerId>;
  gameIsOver(): boolean;
  isDoneWithFinalProduction(): boolean;
  serialize(): SerializedGame;
}

interface IDatabase {
  markFinished(gameId: GameId): Promise<void>;
  deleteGameNbrSaves(gameId: GameId, rollbackCount: number): Promise<void>;
  getPlayerCount(gameId: GameId): Promise<number>;
  getGame(gameId: GameId): Promise<SerializedGame>;
  getGameId(id: PlayerId | SpectatorId): Promise<GameId>;
  getGameVersion(gameId: GameId, saveId: number): Promise<SerializedGame>;
  getGameIds(): Promise<Array<GameId>>;
  getSaveIds(gameId: GameId): Promise<Array<number>>;
  initialize(): Promise<void>;
  saveGameResults(gameId: GameId, players: number, generations: number, gameOptions: GameOptions, scores: Array<unknown>): void;
  saveGame(game: IGame): Promise<void>;
  purgeUnfinishedGames(): Promise<Array<GameId>>;
  compressCompletedGames(): Promise<void>;
  stats(): Promise<Record<string, unknown>>;
  storeParticipants(entry: unknown): Promise<void>;
  getParticipants(): Promise<Array<unknown>>;
  createSession(session: unknown): Promise<void>;
  deleteSession(sessionId: string): Promise<void>;
  getSessions(): Promise<Array<unknown>>;
  deleteExpiredSessions(): Promise<number>;
}

interface IGameLoader {
  add(game: IGame): Promise<void>;
  getIds(): Promise<Array<unknown>>;
  getGame(id: GameId | PlayerId | SpectatorId, forceLoad?: boolean): Promise<IGame | undefined>;
  restoreGameAt(gameId: GameId, saveId: number): Promise<IGame>;
  mark(gameId: GameId): void;
  saveGame(game: IGame): Promise<void>;
  completeGame(game: IGame): Promise<void>;
  maintenance(): Promise<void>;
}

// Runtime requires against the compiled CommonJS engine build in tools/simbench/engine/build/src.
const ENGINE_BUILD_ROOT = path.resolve(__dirname, '../../../tools/simbench/engine/build/src');

function engineRequire<T>(relPath: string): T {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require(path.join(ENGINE_BUILD_ROOT, relPath)) as T;
}

const {globalInitialize} = engineRequire<{globalInitialize: () => void}>('server/globalInitialize.js');
const {Game} = engineRequire<{
  Game: {
    newInstance(
      id: GameId,
      players: Array<IPlayer>,
      firstPlayer: IPlayer,
      spectatorId: SpectatorId,
      partialOptions?: Partial<GameOptions>,
      seed?: number,
    ): IGame;
    deserialize(d: SerializedGame): IGame;
  };
}>('server/Game.js');
const {Player} = engineRequire<{
  Player: new (name: string, color: Color, beginner: boolean, handicap: number, id: PlayerId) => IPlayer;
}>('server/Player.js');
const {Database} = engineRequire<{Database: {getInstance: () => IDatabase}}>('server/database/Database.js');
const {GameLoader} = engineRequire<{GameLoader: {getInstance: () => IGameLoader}}>('server/database/GameLoader.js');
const {BoardName} = engineRequire<{BoardName: Record<string, BoardNameValue>}>('common/boards/BoardName.js');
const {Phase} = engineRequire<{
  Phase: Record<string, string> & {
    END: string;
    RESEARCH: string;
    DRAFTING: string;
    INITIALDRAFTING: string;
    PRELUDES: string;
    ACTION: string;
    PRODUCTION: string;
  };
}>('common/Phase.js');
const {RandomMAOptionType} = engineRequire<{RandomMAOptionType: Record<string, string> & {NONE: string}}>('common/ma/RandomMAOptionType.js');
const constants = engineRequire<{
  CORPORATION_CARDS_DEALT_PER_PLAYER: number;
  PRELUDE_CARDS_DEALT_PER_PLAYER: number;
  CEO_CARDS_DEALT_PER_PLAYER: number;
  CARD_COST: number;
  PRODUCTION_MINIMUMS: Record<string, number>;
}>('common/constants.js');
const {Payment: PaymentUtil, DEFAULT_PAYMENT_VALUES} = engineRequire<{
  Payment: {EMPTY: Readonly<Payment>; of: (p: Partial<Payment>) => Payment};
  DEFAULT_PAYMENT_VALUES: Record<SpendableResource, number>;
}>('common/inputs/Payment.js');
const {SPENDABLE_RESOURCES} = engineRequire<{
  SPENDABLE_RESOURCES: ReadonlyArray<SpendableResource>;
}>('common/inputs/Spendable.js');
const {Units: UnitsUtil} = engineRequire<{
  Units: {
    EMPTY: Readonly<Units>;
    keys: ReadonlyArray<keyof Units>;
    of: (u: Partial<Units>) => Units;
  };
}>('common/Units.js');
const {CardName: CardNames} = engineRequire<{CardName: Record<string, string>}>('common/cards/CardName.js');
const {Server} = engineRequire<{
  Server: {
    getPlayerModel(player: IPlayer): EnginePlayerViewModel;
  };
}>('server/models/ServerModel.js');
const {GameCards} = engineRequire<{
  GameCards: new (gameOptions: GameOptions) => {
    getProjectCards(): Array<IProjectCard>;
    getCorporationCards(): Array<ICorporationCard>;
    getPreludeCards(): Array<IPreludeCard>;
  };
}>('server/GameCards.js');
const {newProjectCard, newPrelude, newCorporationCard} = engineRequire<{
  newProjectCard(name: CardName): IProjectCard | undefined;
  newPrelude(name: CardName): IPreludeCard | undefined;
  newCorporationCard(name: CardName): ICorporationCard | undefined;
}>('server/createCard.js');
const {newStandardDraft} = engineRequire<{
  newStandardDraft(game: IGame): {
    startDraft(save?: boolean): void;
    restoreDraft(): void;
  };
}>('server/Draft.js');

export {
  Game,
  Player,
  Server,
  GameCards,
  newProjectCard,
  newPrelude,
  newCorporationCard,
  newStandardDraft,
  BoardName,
  Phase,
  RandomMAOptionType,
  constants,
  PaymentUtil,
  DEFAULT_PAYMENT_VALUES,
  SPENDABLE_RESOURCES,
  UnitsUtil,
  CardNames,
};

let initialized = false;

/**
 * Installs in-memory no-op fakes for Database and GameLoader before calling globalInitialize() once.
 */
export function initializeEngine(): void {
  if (initialized) {
    return;
  }

  const inMemoryGames = new Map<GameId, IGame>();
  const participantToGame = new Map<PlayerId | SpectatorId, GameId>();

  const fakeDatabase: IDatabase = {
    markFinished: () => Promise.resolve(),
    deleteGameNbrSaves: () => Promise.resolve(),
    getPlayerCount: () => Promise.resolve(0),
    getGame: () => Promise.resolve({} as SerializedGame),
    getGameId: () => Promise.resolve('g0' as GameId),
    getGameVersion: () => Promise.resolve({} as SerializedGame),
    getGameIds: () => Promise.resolve([]),
    getSaveIds: () => Promise.resolve([]),
    initialize: () => Promise.resolve(),
    saveGameResults: () => {},
    saveGame: () => Promise.resolve(),
    purgeUnfinishedGames: () => Promise.resolve([]),
    compressCompletedGames: () => Promise.resolve(),
    stats: () => Promise.resolve({}),
    storeParticipants: () => Promise.resolve(),
    getParticipants: () => Promise.resolve([]),
    createSession: () => Promise.resolve(),
    deleteSession: () => Promise.resolve(),
    getSessions: () => Promise.resolve([]),
    deleteExpiredSessions: () => Promise.resolve(0),
  };

  const fakeGameLoader: IGameLoader = {
    add: async (game: IGame) => {
      inMemoryGames.set(game.id, game);
      if (game.spectatorId) {
        participantToGame.set(game.spectatorId, game.id);
      }
      for (const p of game.players) {
        participantToGame.set(p.id, game.id);
      }
    },
    getIds: async () => [],
    getGame: async (id: GameId | PlayerId | SpectatorId) => {
      const gid = inMemoryGames.has(id as GameId) ? (id as GameId) : participantToGame.get(id as PlayerId | SpectatorId);
      return gid ? inMemoryGames.get(gid) : undefined;
    },
    restoreGameAt: async () => {
      throw new Error('restoreGameAt is not supported in adapter in-memory mode');
    },
    mark: () => {},
    saveGame: () => Promise.resolve(),
    completeGame: () => Promise.resolve(),
    maintenance: () => Promise.resolve(),
  };

  Database.getInstance = () => fakeDatabase;
  GameLoader.getInstance = () => fakeGameLoader;

  globalInitialize();
  initialized = true;
}
