function streamEvents(url, handlers = {}, eventName = null) {
  const eventSource = new EventSource(url);

  const handleMessage = (event) => {
    const payload = JSON.parse(event.data);
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

  if (eventName) {
    eventSource.addEventListener(eventName, handleMessage);
  } else {
    eventSource.onmessage = handleMessage;
  }

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

export function streamTrainingEvents(sessionId, handlers = {}) {
  const params = new URLSearchParams({ session_id: sessionId });
  return streamEvents(`/api/training/events?${params.toString()}`, handlers, 'training');
}
