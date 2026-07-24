// İlk prototipte desteklenen parmak etiketi kümesi; çoklu parmak aşamasında genişletilecek.
export type FingerLabel = 'right_index' | 'left_index' | 'unknown';

// Kalite durumları UI, kayıt ve enrollment kararlarında ortak dil olarak kullanılır.
export type QualityStatus = 'unknown' | 'poor' | 'usable' | 'good';

// Capture sonrası hesaplanan temel görüntü kalite metrikleri.
export type QualityMetrics = {
  blurScore: number;
  glareRatio: number;
  brightnessMean: number;
  roiCoverage: number;
  status: QualityStatus;
};

// OBB modelinin desteklediği parmak ucu sınıf adları.
export type DetectionClassName = 'index' | 'middle' | 'pinky' | 'ring' | 'unknown';

// Modelden gelen açılı kutuyu görüntüye oranlı dört köşe noktasıyla taşır.
export type DetectedObbBox = {
  id: string;
  classId: number;
  className: DetectionClassName;
  confidence: number;
  center: {
    x: number;
    y: number;
  };
  size: {
    width: number;
    height: number;
  };
  angle: number;
  points: {
    x: number;
    y: number;
  }[];
};

// Tek bir kamera yakalamasının ham görsel, ROI ve kalite metadata'sını taşır.
export type CaptureSample = {
  id: string;
  createdAt: string;
  rawImageUri: string;
  roiImageUri?: string;
  processedRoiImageUri?: string;
  roiCrop?: {
    originX: number;
    originY: number;
    width: number;
    height: number;
  };
  rawImageSize?: {
    width: number;
    height: number;
  };
  qualityMetrics?: QualityMetrics;
  processedQualityMetrics?: QualityMetrics;
  detections?: DetectedObbBox[];
  deviceModel?: string;
  fingerLabel: FingerLabel;
  sessionId: string;
  qualityStatus: QualityStatus;
  accepted: boolean;
};

// Yerel galeriye eklenecek kişi kaydının temel modeli.
export type Person = {
  id: string;
  displayName: string;
  createdAt: string;
};

// Bir kişinin belirli parmağı için kabul edilmiş capture örneklerini gruplayan enrollment modeli.
export type Enrollment = {
  id: string;
  personId: string;
  fingerLabel: FingerLabel;
  samples: string[];
  templateVersion: string;
};

// 1:N arama sonucunda aday kişiyi skor ve sıralama bilgisiyle döndürecek tip.
export type SearchResult = {
  personId: string;
  score: number;
  rank: number;
};
