import { describe, it, expect } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { useModelStatus } from '../useModelStatus';
import { server } from '../../test/server';
import { http, HttpResponse } from 'msw';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe('useModelStatus Hook with MSW', () => {
  it('successfully fetches and adapts CISD challenge data', async () => {
    const { result } = renderHook(() => useModelStatus('cisd', { retry: false }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.modelId).toBe('cisd');
    expect(result.current.data?.balance).toBe(5120.0);
    expect(result.current.data?.pnlUsd).toBe(120.0);
  });

  it('handles server 500 error gracefully', async () => {
    server.use(
      http.get('/api/challenges/cisd', () => {
        return new HttpResponse(null, { status: 500 });
      })
    );

    const { result } = renderHook(() => useModelStatus('cisd', { retry: false }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error?.message).toContain('500');
  });
});
