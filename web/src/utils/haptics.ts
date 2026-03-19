export function haptic(ms: number = 50) {
  if (navigator?.vibrate) navigator.vibrate(ms);
}
