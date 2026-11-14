import {
  CardNames,
  DEFAULT_PAYMENT_VALUES,
  ICard,
  IPlayer,
  IProjectCard,
  IStandardProjectCard,
  Payment,
  PaymentOptions,
  PaymentUtil,
  SPENDABLE_RESOURCES,
  SelectProjectCardToPlay,
  SelectStandardProjectToPlay,
  SpendableResource,
  Units,
} from './engine';

export interface PaymentCandidate {
  readonly id: string;
  readonly label: string;
  readonly payment: Payment;
}

export interface ProjectPlayCandidate {
  readonly card: IProjectCard | IStandardProjectCard;
  readonly payments: ReadonlyArray<PaymentCandidate>;
}

const RESOURCE_SHORT_LABELS: Record<SpendableResource, string> = {
  megacredits: 'M€',
  steel: 'steel',
  titanium: 'titanium',
  plants: 'plants',
  heat: 'heat',
  microbes: 'microbes',
  floaters: 'floaters',
  lunaArchivesScience: 'science',
  spireScience: 'spire science',
  seeds: 'seeds',
  auroraiData: 'data',
  graphene: 'graphene',
  kuiperAsteroids: 'asteroids',
};

export function isStandardProjectCard(card: ICard): card is IStandardProjectCard {
  return (
    typeof (card as IStandardProjectCard).canAct === 'function' &&
    typeof (card as IStandardProjectCard).canPayWith === 'function'
  );
}

export function getAvailableResourceAmount(
  resource: SpendableResource,
  player: IPlayer,
  reserveUnits: Readonly<Units> | undefined,
  cardName?: string,
): number {
  switch (resource) {
    case 'megacredits':
      return Math.max(0, player.megaCredits - (reserveUnits?.megacredits ?? 0));
    case 'steel':
      return Math.max(0, player.steel - (reserveUnits?.steel ?? 0));
    case 'titanium':
      return Math.max(0, player.titanium - (reserveUnits?.titanium ?? 0));
    case 'plants':
      return Math.max(0, player.plants - (reserveUnits?.plants ?? 0));
    case 'heat':
      return Math.max(0, player.availableHeat() - (reserveUnits?.heat ?? 0));
    case 'microbes': {
      const avail = player.getSpendable('microbes');
      if (cardName === CardNames.SOIL_ENRICHMENT && avail > 0) {
        const cardsWithMicrobe = player.getCardsWithResources('microbe');
        if (cardsWithMicrobe.length === 1) {
          return Math.max(0, avail - 1);
        }
      }
      return avail;
    }
    case 'floaters': {
      const avail = player.getSpendable('floaters');
      if (cardName === CardNames.STRATOSPHERIC_BIRDS && avail > 0) {
        const cardsWithFloater = player.getCardsWithResources('floater');
        if (cardsWithFloater.length === 1) {
          return Math.max(0, avail - 1);
        }
      }
      return avail;
    }
    default:
      return player.getSpendable(resource);
  }
}

export function getUnitMultiplier(
  resource: SpendableResource,
  player: IPlayer,
  options: Partial<PaymentOptions>,
): number {
  switch (resource) {
    case 'megacredits':
      return 1;
    case 'steel':
      return options.steel ? player.getSteelValue() : 0;
    case 'titanium':
      if (options.titanium) {
        return player.getTitaniumValue();
      }
      if (options.lunaTradeFederationTitanium || player.canUseTitaniumAsMegacredits) {
        return Math.max(0, player.getTitaniumValue() - 1);
      }
      return 0;
    case 'heat':
      return options.heat || player.canUseHeatAsMegaCredits ? 1 : 0;
    case 'plants':
      return options.plants ? DEFAULT_PAYMENT_VALUES.plants : 0;
    default:
      return options[resource] ? DEFAULT_PAYMENT_VALUES[resource] : 0;
  }
}

function formatPaymentId(payment: Payment): string {
  const parts: Array<string> = [];
  for (const res of SPENDABLE_RESOURCES) {
    if (payment[res] > 0) {
      parts.push(`${res}=${payment[res]}`);
    }
  }
  return parts.length > 0 ? parts.join(',') : 'megacredits=0';
}

function formatPaymentLabel(payment: Payment): string {
  const parts: Array<string> = [];
  // Show alternative resources first, megacredits last for readability
  for (const res of SPENDABLE_RESOURCES) {
    if (res !== 'megacredits' && payment[res] > 0) {
      parts.push(`${payment[res]} ${RESOURCE_SHORT_LABELS[res]}`);
    }
  }
  if (payment.megacredits > 0 || parts.length === 0) {
    parts.push(`${payment.megacredits} M€`);
  }
  return parts.join(' + ');
}

interface ResourceEntry {
  readonly resource: SpendableResource;
  readonly avail: number;
  readonly value: number;
}

function permute<T>(items: ReadonlyArray<T>): Array<Array<T>> {
  if (items.length <= 1) {
    return [items.slice()];
  }
  const result: Array<Array<T>> = [];
  for (let i = 0; i < items.length; i++) {
    const head = items[i];
    const rest = items.slice(0, i).concat(items.slice(i + 1));
    for (const sub of permute(rest)) {
      result.push([head, ...sub]);
    }
  }
  return result;
}

function generateOrders(entries: ReadonlyArray<ResourceEntry>): Array<Array<ResourceEntry>> {
  if (entries.length <= 5) {
    return permute(entries);
  }
  // Bounded fallback if an unusual scenario ever has >5 simultaneous spendable resources
  const orders: Array<Array<ResourceEntry>> = [];
  const mcFirst = entries.slice().sort((a, b) => (a.resource === 'megacredits' ? -1 : b.resource === 'megacredits' ? 1 : b.value - a.value));
  const valueDesc = entries.slice().sort((a, b) => b.value - a.value);
  const valueAsc = entries.slice().sort((a, b) => a.value - b.value);
  orders.push(mcFirst, valueDesc, valueAsc);
  for (let i = 0; i < entries.length; i++) {
    const chosen = entries[i];
    const others = entries.filter((_, idx) => idx !== i);
    orders.push([chosen, ...others.sort((a, b) => b.value - a.value)]);
    orders.push([chosen, ...others.sort((a, b) => a.value - b.value)]);
  }
  return orders;
}

function greedySplit(
  requiredCost: number,
  order: ReadonlyArray<ResourceEntry>,
  mode: 'exact' | 'ceil',
): Payment {
  const payment: Record<SpendableResource, number> = {...PaymentUtil.EMPTY};
  const suffixMax = new Array<number>(order.length + 1).fill(0);
  for (let i = order.length - 1; i >= 0; i--) {
    suffixMax[i] = suffixMax[i + 1] + order[i].avail * order[i].value;
  }

  let remaining = requiredCost;
  for (let i = 0; i < order.length && remaining > 0; i++) {
    const entry = order[i];
    const laterMax = suffixMax[i + 1];
    const minNeeded = Math.ceil(Math.max(0, remaining - laterMax) / entry.value);
    const greedyRaw =
      mode === 'exact'
        ? Math.floor(remaining / entry.value)
        : Math.ceil(remaining / entry.value);
    const take = Math.min(entry.avail, Math.max(minNeeded, greedyRaw));
    if (take > 0) {
      payment[entry.resource] = take;
      remaining = Math.max(0, remaining - take * entry.value);
    }
  }
  return payment;
}

function dominates(a: Payment, b: Payment): boolean {
  let strictlyBetterInOne = false;
  for (const res of SPENDABLE_RESOURCES) {
    if (a[res] > b[res]) {
      return false;
    }
    if (a[res] < b[res]) {
      strictlyBetterInOne = true;
    }
  }
  return strictlyBetterInOne;
}

/**
 * Enumerates a bounded canonical set of Pareto-efficient payment splits by spending
 * resources greedily across preference orders, deduplicating, and removing dominated splits.
 */
export function enumerateCanonicalPayments(
  requiredCost: number,
  options: Partial<PaymentOptions>,
  reserveUnits: Readonly<Units> | undefined,
  player: IPlayer,
  cardName?: string,
): Array<PaymentCandidate> {
  if (requiredCost <= 0) {
    const payment = {...PaymentUtil.EMPTY};
    return [
      {
        id: formatPaymentId(payment),
        label: formatPaymentLabel(payment),
        payment,
      },
    ];
  }

  const entries: Array<ResourceEntry> = [];
  let maxPossible = 0;
  for (const resource of SPENDABLE_RESOURCES) {
    const value = getUnitMultiplier(resource, player, options);
    if (value <= 0) {
      continue;
    }
    const avail = getAvailableResourceAmount(resource, player, reserveUnits, cardName);
    if (avail > 0) {
      entries.push({resource, avail, value});
      maxPossible += avail * value;
    }
  }

  if (maxPossible < requiredCost) {
    return [];
  }

  const orders = generateOrders(entries);
  const byId = new Map<string, Payment>();
  for (const order of orders) {
    for (const mode of ['exact', 'ceil'] as const) {
      const payment = greedySplit(requiredCost, order, mode);
      const id = formatPaymentId(payment);
      if (!byId.has(id)) {
        byId.set(id, payment);
      }
    }
  }

  const unique = Array.from(byId.entries()).map(([id, payment]) => ({id, payment}));
  const pareto = unique.filter(
    (cand, idx) => !unique.some((other, oIdx) => oIdx !== idx && dominates(other.payment, cand.payment)),
  );

  // Canonical ordering: least megacredits first (most alternative resources first),
  // breaking ties deterministically by resource amount descending in SPENDABLE_RESOURCES order.
  pareto.sort((a, b) => {
    if (a.payment.megacredits !== b.payment.megacredits) {
      return a.payment.megacredits - b.payment.megacredits;
    }
    for (const res of SPENDABLE_RESOURCES) {
      if (a.payment[res] !== b.payment[res]) {
        return b.payment[res] - a.payment[res];
      }
    }
    return 0;
  });

  return pareto.map(({id, payment}) => ({
    id,
    label: formatPaymentLabel(payment),
    payment,
  }));
}

export function getPlayableProjectCandidates(
  input: SelectProjectCardToPlay | SelectStandardProjectToPlay,
  player: IPlayer,
): Array<ProjectPlayCandidate> {
  const candidates: Array<ProjectPlayCandidate> = [];
  const enabled = input.enabled;

  for (let i = 0; i < input.cards.length; i++) {
    if (enabled?.[i] === false) {
      continue;
    }
    const card = input.cards[i];
    const details = input.extras.get(card.name);

    let payments: Array<PaymentCandidate>;
    if (isStandardProjectCard(card)) {
      if (details?.overriddenCost === undefined && !card.canAct(player)) {
        continue;
      }
      const canPayWith = card.canPayWith(player);
      const paymentOptions: Partial<PaymentOptions> = {
        heat: player.canUseHeatAsMegaCredits,
        steel: canPayWith.steel,
        titanium: canPayWith.titanium,
        lunaTradeFederationTitanium: player.canUseTitaniumAsMegacredits,
        seeds: canPayWith.seeds,
        auroraiData: player.tableau.has(CardNames.AURORAI),
        spireScience: player.tableau.has(CardNames.SPIRE),
        kuiperAsteroids: canPayWith.kuiperAsteroids ? player.tableau.has(CardNames.KUIPER_COOPERATIVE) : false,
      };
      const requiredCost = details?.overriddenCost ?? card.getAdjustedCost(player);
      payments = enumerateCanonicalPayments(requiredCost, paymentOptions, details?.reserveUnits, player, card.name);
    } else {
      const afford = player.affordOptionsForCard(card);
      payments = enumerateCanonicalPayments(
        afford.cost,
        afford,
        afford.reserveUnits ?? details?.reserveUnits,
        player,
        card.name,
      );
    }

    if (payments.length > 0) {
      candidates.push({card, payments});
    }
  }

  candidates.sort((a, b) => (a.card.name < b.card.name ? -1 : a.card.name > b.card.name ? 1 : 0));
  return candidates;
}
