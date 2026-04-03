import type { AnimationConfig, FlowOptions } from './types';

const FILL = 'forwards' as const;

/**
 * Commit the final frame styles and release the animation fill lock.
 * This prevents fill:forwards from permanently overriding CSS properties.
 */
function commitAndCancel(anim: Animation): void {
  try {
    anim.commitStyles();
  } catch {
    // commitStyles can fail if element is disconnected — ignore
  }
  anim.cancel();
}

/**
 * Liquid flow — grow/shrink vertically with opacity.
 * direction='down': 0 → full height (appear).
 * direction='up':   full height → 0 (retract).
 */
export async function flow(el: HTMLElement, opts: FlowOptions): Promise<void> {
  // Force reflow to get accurate height before animating
  const h = el.offsetHeight || 0;
  const appear = [
    { height: '0px', opacity: 0 },
    { height: `${h}px`, opacity: 1 },
  ];
  const keyframes = opts.direction === 'down' ? appear : [...appear].reverse();
  const anim = el.animate(keyframes, {
    duration: opts.duration,
    easing: opts.easing,
    delay: opts.delay ?? 0,
    fill: FILL,
  });
  await anim.finished;
  commitAndCancel(anim);
}

/**
 * Dissolve — fade opacity 1 → 0.
 */
export async function dissolve(el: HTMLElement, opts: AnimationConfig): Promise<void> {
  const anim = el.animate(
    [{ opacity: 1 }, { opacity: 0 }],
    { duration: opts.duration, easing: opts.easing, delay: opts.delay ?? 0, fill: FILL },
  );
  await anim.finished;
  commitAndCancel(anim);
}

/**
 * Fade in — fade opacity 0 → 1.
 */
export async function fadeIn(el: HTMLElement, opts: AnimationConfig): Promise<void> {
  const anim = el.animate(
    [{ opacity: 0 }, { opacity: 1 }],
    { duration: opts.duration, easing: opts.easing, delay: opts.delay ?? 0, fill: FILL },
  );
  await anim.finished;
  commitAndCancel(anim);
}

/**
 * Cross-fade — old element fades out while new element fades in simultaneously.
 */
export async function crossFade(
  oldEl: HTMLElement,
  newEl: HTMLElement,
  opts: AnimationConfig,
): Promise<void> {
  const timing = { duration: opts.duration, easing: opts.easing, delay: opts.delay ?? 0, fill: FILL };
  const outAnim = oldEl.animate([{ opacity: 1 }, { opacity: 0 }], timing);
  const inAnim = newEl.animate([{ opacity: 0 }, { opacity: 1 }], timing);
  await Promise.all([outAnim.finished, inAnim.finished]);
  commitAndCancel(outAnim);
  commitAndCancel(inAnim);
}

/**
 * Sequential reveal — stagger-animate each child of parent.
 * Each child: opacity 0→1, translateY(4px)→0, scale(0.95)→1.
 */
export async function sequentialReveal(
  parent: HTMLElement,
  opts: AnimationConfig & { stagger: number },
): Promise<void> {
  const children = Array.from(parent.children) as HTMLElement[];
  const anims = children.map((child, i) =>
    child.animate(
      [
        { opacity: 0, transform: 'translateY(4px) scale(0.95)' },
        { opacity: 1, transform: 'translateY(0) scale(1)' },
      ],
      {
        duration: opts.duration,
        easing: opts.easing,
        delay: i * opts.stagger,
        fill: FILL,
      },
    ),
  );
  await Promise.all(anims.map(a => a.finished));
  for (const a of anims) commitAndCancel(a);
}

/**
 * Morph — animate element from one DOMRect position/size to another.
 * Used for ghost-bar → modal transitions.
 */
export async function morph(
  el: HTMLElement,
  from: DOMRect,
  to: DOMRect,
  opts: AnimationConfig,
): Promise<void> {
  const anim = el.animate(
    [
      {
        width: `${from.width}px`,
        height: `${from.height}px`,
        transform: `translate(${from.left}px, ${from.top}px)`,
      },
      {
        width: `${to.width}px`,
        height: `${to.height}px`,
        transform: `translate(${to.left}px, ${to.top}px)`,
      },
    ],
    { duration: opts.duration, easing: opts.easing, delay: opts.delay ?? 0, fill: FILL },
  );
  await anim.finished;
  commitAndCancel(anim);
}
