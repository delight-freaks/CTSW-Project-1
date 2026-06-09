import { useMemo } from 'react';

export function useDevMode() {
  const isDevMode = useMemo(() => {
    // Check URL query parameter
    const params = new URLSearchParams(window.location.search);
    if (params.get('dev') === 'true') {
      return true;
    }

    // Check environment variable (if available)
    if (import.meta.env.DEV_MODE === 'true') {
      return true;
    }

    return false;
  }, []);

  return isDevMode;
}
