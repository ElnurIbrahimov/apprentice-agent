export function haptic(ms: number = 50) {
  if (navigator?.vibrate) navigator.vibrate(ms);
}

export const haptics = {
  light: () => haptic(10),       // tab switch, button tap
  medium: () => haptic(25),      // send message, action confirm
  heavy: () => haptic(50),       // error, long-press trigger
  success: () => { if (navigator?.vibrate) navigator.vibrate([10, 50, 10]); },
};
