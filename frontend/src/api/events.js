function streamEvents(url, handlers = {}, eventName = null, options = {}) {
  const eventSource = new EventSource(url);
  const closeOnTransportError = Boolean(options.closeOnTransportError);

  const handleMessage = (event) => {
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (error) {
      handlers.onError?.(error);
      return;
    }

    handlers.onEvent?.(payload);

    if (payload.type === 'done' || payload.status === 'all_completed') {
      handlers.onDone?.(payload);
      eventSource.close();
    }
    if (payload.type === 'error') {
      handlers.onError?.(payload);
      eventSource.close();
    }
  };

  eventSource.onopen = () => handlers.onOpen?.();

  if (eventName) {
    eventSource.addEventListener(eventName, handleMessage);
  } else {
    eventSource.onmessage = handleMessage;
  }

  eventSource.onerror = (event) => {
    handlers.onError?.(event);
    // Native EventSource automatically reconnects and carries Last-Event-ID.
    // Keep the stream alive for training unless a caller explicitly opts out.
    if (closeOnTransportError) {
      eventSource.close();
    }
  };

  return eventSource;
}

export function streamValidationEvents(sessionId, handlers = {}) {
  const params = new URLSearchParams({ session_id: sessionId });
  return streamEvents(
    `/api/validate/events?${params.toString()}`,
    handlers,
    null,
    { closeOnTransportError: true },
  );
}

export function streamMetadataEvents(sessionId, handlers = {}) {
  const params = new URLSearchParams({ session_id: sessionId });
  return streamEvents(
    `/api/metadata/events?${params.toString()}`,
    handlers,
    null,
    { closeOnTransportError: true },
  );
}

export function streamTrainingEvents(sessionId, handlers = {}) {
  const params = new URLSearchParams({ session_id: sessionId });
  return streamEvents(`/api/training/events?${params.toString()}`, handlers, 'training');
}

/**
 * Stream evaluation-stage events (judge turns, SHAP, overfitting) for a session.
 * Reuses the same /api/training/events SSE endpoint - all pipeline stages share
 * one event bus per session. This export is intended for the leaderboard / evaluation
 * screens so they can show judge progress without the training page being open.
 */
export function streamEvaluationEvents(sessionId, handlers = {}) {
  const params = new URLSearchParams({ session_id: sessionId });
  return streamEvents(`/api/training/events?${params.toString()}`, handlers, 'training');
}

