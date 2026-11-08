import { useSyncExternalStore } from 'react';

/**
 * DiffusionGemma writes a block of words at once and improves it over several steps. While it does, the engine sends the whole canvas as it
 * stands now (`delta.strata_draft`; the host passes it on as a `draft` frame). Each one replaces the one before. It is a preview only:
 * it lives here, outside the saved chats, so no chat on disk ever holds it, and it is cleared when the turn ends, fails or is stopped.
 */
const drafts = new Map<string, string>();
const listeners = new Set<() => void>();
const emit = () => listeners.forEach(fn => fn());
/** The most that is kept: a canvas is a few hundred words; anything longer is cut so a stray frame cannot grow the page. */
export const DRAFT_LIMIT = 20_000;

export function setDraft(sessionId: string, text: unknown): void {
  const next = typeof text === 'string' ? text.slice(0, DRAFT_LIMIT) : '';
  if (next === (drafts.get(sessionId) ?? '')) return;
  if (next) drafts.set(sessionId, next); else drafts.delete(sessionId);
  emit();
}
export const clearDraft = (sessionId: string) => setDraft(sessionId, '');
export const draftOf = (sessionId: string | null | undefined): string => (sessionId ? drafts.get(sessionId) ?? '' : '');
const subscribe = (fn: () => void) => { listeners.add(fn); return () => { listeners.delete(fn); }; };
export const useDraft = (sessionId: string | null | undefined): string => useSyncExternalStore(subscribe, () => draftOf(sessionId), () => '');
