/**
 * Independent deterministic PRNG (Mulberry32) for the benchmark driver.
 * Never uses Math.random and never touches the engine's internal SeededRandom.
 */
export class SeededPrng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  /** Returns a float in [0, 1). */
  public next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Returns an integer in [0, max). */
  public nextInt(max: number): number {
    if (max <= 0) {
      throw new Error(`nextInt max must be > 0, got ${max}`);
    }
    return Math.floor(this.next() * max);
  }

  /** Returns an integer in [min, max] inclusive. */
  public nextIntRange(min: number, max: number): number {
    if (max < min) {
      throw new Error(`nextIntRange invalid range [${min}, ${max}]`);
    }
    return min + this.nextInt(max - min + 1);
  }

  /** Picks one element uniformly from a non-empty readonly array. */
  public pick<T>(items: ReadonlyArray<T>): T {
    if (items.length === 0) {
      throw new Error('Cannot pick from empty array');
    }
    return items[this.nextInt(items.length)];
  }

  /** Returns a shuffled copy of items using Fisher-Yates. */
  public shuffled<T>(items: ReadonlyArray<T>): Array<T> {
    const copy = items.slice();
    for (let i = copy.length - 1; i > 0; i--) {
      const j = this.nextInt(i + 1);
      const tmp = copy[i];
      copy[i] = copy[j];
      copy[j] = tmp;
    }
    return copy;
  }

  /** Samples `count` distinct elements without replacement, preserving no order. */
  public sample<T>(items: ReadonlyArray<T>, count: number): Array<T> {
    if (count < 0 || count > items.length) {
      throw new Error(`Cannot sample ${count} items from ${items.length}`);
    }
    if (count === 0) {
      return [];
    }
    return this.shuffled(items).slice(0, count);
  }
}
