import assert from 'node:assert/strict';
import test from 'node:test';

import { streamTrainingEvents } from './events.js';

class MockEventSource {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.closed = false;
    this.listeners = {};
    MockEventSource.instances.push(this);
  }

  addEventListener(eventName, listener) {
    this.listeners[eventName] = listener;
  }

  close() {
    this.closed = true;
  }

  emit(eventName, payload) {
    this.listeners[eventName]?.({ data: JSON.stringify(payload) });
  }

  emitRaw(eventName, data) {
    this.listeners[eventName]?.({ data });
  }
}

test('streamTrainingEvents connects to the training SSE endpoint', () => {
  MockEventSource.instances = [];
  globalThis.EventSource = MockEventSource;

  let opened = false;
  const source = streamTrainingEvents('session 1', {
    onOpen: () => {
      opened = true;
    },
  });

  assert.equal(source.url, '/api/training/events?session_id=session+1');
  source.onopen();
  assert.equal(opened, true);
});

test('streamTrainingEvents keeps the stream open on transport errors for reconnect', () => {
  MockEventSource.instances = [];
  globalThis.EventSource = MockEventSource;

  let errorCount = 0;
  const source = streamTrainingEvents('session-1', {
    onError: () => {
      errorCount += 1;
    },
  });

  source.onerror(new Error('temporary network failure'));
  assert.equal(errorCount, 1);
  assert.equal(source.closed, false);
});

test('streamTrainingEvents dispatches events and closes on all_completed', () => {
  MockEventSource.instances = [];
  globalThis.EventSource = MockEventSource;

  const received = [];
  let donePayload = null;
  const source = streamTrainingEvents('session-1', {
    onEvent: (payload) => received.push(payload.status),
    onDone: (payload) => {
      donePayload = payload;
    },
  });

  source.emit('training', { status: 'running', model_id: 'model_001' });
  source.emit('training', { status: 'all_completed', model_id: null });

  assert.deepEqual(received, ['running', 'all_completed']);
  assert.equal(donePayload.status, 'all_completed');
  assert.equal(source.closed, true);
});

test('streamTrainingEvents reports malformed event payloads', () => {
  MockEventSource.instances = [];
  globalThis.EventSource = MockEventSource;

  let parseError = null;
  const source = streamTrainingEvents('session-1', {
    onError: (error) => {
      parseError = error;
    },
  });

  source.emitRaw('training', '{not-json');
  assert.ok(parseError instanceof Error);
  assert.equal(source.closed, false);
});
