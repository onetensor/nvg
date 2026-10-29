import { describe, expect, it } from 'vitest';
import { isAppleMobileWebKit } from '../src/platform';

describe('Apple mobile WebKit detection', () => {
  it('detects iPhone Safari', () => {
    expect(isAppleMobileWebKit(
      'Mozilla/5.0 (iPhone; CPU iPhone OS 18_6 like Mac OS X) AppleWebKit/605.1.15 Version/18.6 Mobile/15E148 Safari/604.1',
      'iPhone',
      5,
    )).toBe(true);
  });

  it('detects an iPad requesting the desktop site', () => {
    expect(isAppleMobileWebKit(
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) AppleWebKit/605.1.15 Version/18.0 Safari/605.1.15',
      'MacIntel',
      5,
    )).toBe(true);
  });

  it('does not classify desktop Chrome as Apple mobile WebKit', () => {
    expect(isAppleMobileWebKit(
      'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140.0 Safari/537.36',
      'Linux x86_64',
      0,
    )).toBe(false);
  });
});
