function streamEvents(url, handlers = {}) {
  const eventSource = new EventSource(url);

  eventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    handlers.onEvent?.(payload);

    if (payload.type === 'done') {
      handlers.onDone?.(payload);
      eventSource.close();
    }
    if (payload.type === 'error') {
      handlers.onError?.(payload);
      eventSource.close();
    }
  };

  eventSource.onerror = (event) => {
    handlers.onError?.(event);
    eventSource.close();
  };

  return eventSource;
}

export function streamValidationEvents(sessionId, handlers = {}) {
  const params = new URLSearchParams({ session_id: sessionId });
  return streamEvents(`/api/validate/events?${params.toString()}`, handlers);
}

export function streamMetadataEvents(sessionId, handlers = {}) {
  const params = new URLSearchParams({ session_id: sessionId });
  return streamEvents(`/api/metadata/events?${params.toString()}`, handlers);
}
