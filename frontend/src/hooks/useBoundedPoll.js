import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Bounded polling hook shared by the Live Training and Leaderboard screens.
 *
 * Replaces the bespoke setTimeout/setInterval pollers that previously had no
 * error cap (and therefore looped forever on a persistently failing endpoint)
 * or stopped permanently on the first transient error. It guarantees:
 *   - cleanup on unmount and when `enabled`/`resetKey` changes,
 *   - a cap on consecutive errors before giving up (no infinite error loops),
 *   - an optional cap on total successful attempts,
 *   - a terminal predicate to stop on completion,
 *   - a `restart()` to resume after a give-up (for "Retry" buttons).
 *
 * `poll` and `stopWhen` are captured in refs so they always see fresh state
 * without restarting the interval on every render.
 *
 * @param {() => Promise<any>} poll - async function run each tick (does its own
 *   fetching + state updates) and returns a value passed to `stopWhen`.
 * @param {object} options
 * @param {boolean} [options.enabled=true] - when false, polling is idle/stopped.
 * @param {number}  [options.intervalMs=2000] - delay between ticks.
 * @param {number}  [options.maxAttempts=0] - max successful ticks (0 = unlimited).
 * @param {number}  [options.maxErrorAttempts=5] - consecutive errors before give-up.
 * @param {(result:any)=>boolean} [options.stopWhen] - stop when this returns true.
 * @param {any}     [options.resetKey] - changing this restarts the poll loop.
 * @returns {{ pollState: 'idle'|'polling'|'stopped'|'capped'|'gave_up', restart: () => void }}
 */
export function useBoundedPoll(poll, options = {}) {
  const {
    enabled = true,
    intervalMs = 2000,
    maxAttempts = 0,
    maxErrorAttempts = 5,
    stopWhen,
    resetKey = null,
  } = options;

  const [pollState, setPollState] = useState('idle');
  const [restartToken, setRestartToken] = useState(0);

  // Keep the latest callbacks without forcing the effect to re-subscribe.
  const pollRef = useRef(poll);
  const stopWhenRef = useRef(stopWhen);
  pollRef.current = poll;
  stopWhenRef.current = stopWhen;

  useEffect(() => {
    if (!enabled) {
      setPollState('idle');
      return undefined;
    }

    let stopped = false;
    let timerId = null;
    let attempts = 0;
    let errorStreak = 0;
    setPollState('polling');

    async function tick() {
      attempts += 1;
      try {
        const result = await pollRef.current();
        if (stopped) return;
        errorStreak = 0;
        if (stopWhenRef.current && stopWhenRef.current(result)) {
          setPollState('stopped');
          return;
        }
        if (maxAttempts > 0 && attempts >= maxAttempts) {
          setPollState('capped');
          return;
        }
        timerId = window.setTimeout(tick, intervalMs);
      } catch (pollError) {
        if (stopped) return;
        errorStreak += 1;
        if (errorStreak >= maxErrorAttempts) {
          setPollState('gave_up');
          return;
        }
        timerId = window.setTimeout(tick, intervalMs);
      }
    }

    tick();

    return () => {
      stopped = true;
      if (timerId) {
        window.clearTimeout(timerId);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, intervalMs, maxAttempts, maxErrorAttempts, resetKey, restartToken]);

  const restart = useCallback(() => {
    setRestartToken((token) => token + 1);
  }, []);

  return { pollState, restart };
}
