import type { DetectionClassName, FingerRoi } from '@/types/biometrics';

// Parmakları kullanıcı arayüzünde ve kayıt metadata'sında kullanılacak sabit sırayla eşler.
const FINGER_DISPLAY_ORDER: Record<DetectionClassName, number> = {
  index: 0,
  middle: 1,
  ring: 2,
  pinky: 3,
  unknown: 4,
};

// Verilen ROI listesini değiştirmeden işaret, orta, yüzük ve serçe sırasına dizer.
export function sortFingerRois<T extends Pick<FingerRoi, 'className'>>(
  fingerRois: readonly T[]
) {
  return [...fingerRois].sort(
    (left, right) =>
      FINGER_DISPLAY_ORDER[left.className] -
      FINGER_DISPLAY_ORDER[right.className]
  );
}
