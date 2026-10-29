export function isAppleMobileWebKit(
  userAgent = navigator.userAgent,
  platform = navigator.platform,
  maxTouchPoints = navigator.maxTouchPoints,
): boolean {
  const iosDevice = /iPad|iPhone|iPod/i.test(userAgent);
  const ipadDesktopMode = platform === 'MacIntel' && maxTouchPoints > 1;
  return /AppleWebKit/i.test(userAgent) && (iosDevice || ipadDesktopMode);
}
