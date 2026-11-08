import { tr } from '../lib/i18n';
import { useDraft } from '../lib/canvas-draft';

/**
 * DiffusionGemma's canvas while it is still being improved: the whole block as it stands now, faded, after the text already written.
 * A preview, not part of the answer: it cannot be selected or copied, a screen reader skips it (the finished words are what is read),
 * nothing about it moves, and it is gone when the turn ends. It is replaced as a whole each time, never added to.
 */
export function DraftCanvas({ sessionId }: { sessionId: string | null | undefined }) {
    const text = useDraft(sessionId);
    if (!text) return null;
    return <div className="st-canvas-draft" data-canvas-draft="" aria-hidden="true" title={tr('Still being written. This preview changes until it is final.')}>{text}</div>;
}
