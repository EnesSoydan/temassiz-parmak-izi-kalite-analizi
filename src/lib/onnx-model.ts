import { Asset } from 'expo-asset';
import { Platform } from 'react-native';

type OrtModule = typeof import('onnxruntime-react-native');
type OrtSession = Awaited<ReturnType<OrtModule['InferenceSession']['create']>>;

// Mobil inference icin uygulamaya gomulen ONNX model asset'ini temsil eder.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const FINGERTIP_OBB_MODEL = require('../../assets/models/fingertip_obb.onnx');

let cachedSession: OrtSession | null = null;
let cachedSessionPromise: Promise<OrtSession> | null = null;
let cachedOrt: OrtModule | null = null;

// ONNX modelini bir kez indirip hızlandırılmış InferenceSession olarak hazırlar; sonraki çağrılarda cache kullanır.
export async function loadFingertipObbSession() {
  if (cachedSession) {
    return cachedSession;
  }

  if (cachedSessionPromise) {
    return cachedSessionPromise;
  }

  cachedSessionPromise = (async () => {
    const ort = await loadOnnxRuntime();
    const [modelAsset] = await Asset.loadAsync(FINGERTIP_OBB_MODEL);

    if (!modelAsset.localUri) {
      throw new Error('ONNX model asset dosyası cihaza indirilemedi.');
    }

    cachedSession = await createAcceleratedSession(ort, modelAsset.localUri);
    return cachedSession;
  })();

  try {
    return await cachedSessionPromise;
  } catch (error) {
    cachedSessionPromise = null;
    throw error;
  }
}

// Android'de NNAPI, iOS'ta CoreML dener; cihaz hızlandırıcısı açılamazsa XNNPACK/CPU'ya geri döner.
async function createAcceleratedSession(ort: OrtModule, modelUri: string) {
  try {
    if (Platform.OS === 'android') {
      return await ort.InferenceSession.create(modelUri, {
        executionProviders: [
          {
            name: 'nnapi',
            useFP16: true,
            cpuDisabled: true,
          },
          'cpu',
        ],
        graphOptimizationLevel: 'all',
      });
    }

    if (Platform.OS === 'ios') {
      return await ort.InferenceSession.create(modelUri, {
        executionProviders: [
          {
            name: 'coreml',
            useCPUAndGPU: true,
          },
          'cpu',
        ],
        graphOptimizationLevel: 'all',
      });
    }
  } catch (error) {
    console.warn('Cihaz model hızlandırıcısı açılamadı; XNNPACK kullanılacak.', error);
  }

  return ort.InferenceSession.create(modelUri, {
    executionProviders: ['xnnpack', 'cpu'],
    graphOptimizationLevel: 'all',
  });
}

// Inference adımında Tensor constructor ve hazır session birlikte gerektiği için runtime'ı tek noktadan döndürür.
export async function getFingertipObbRuntime() {
  const ort = await loadOnnxRuntime();
  const session = await loadFingertipObbSession();

  return { ort, session };
}

// Expo Go içinde native ONNX Runtime modülü bulunmaz; import'u geciktirerek uygulamanın tamamen kırılmasını engeller.
async function loadOnnxRuntime(): Promise<OrtModule> {
  if (cachedOrt) {
    return cachedOrt;
  }

  try {
    cachedOrt = await import('onnxruntime-react-native');
    return cachedOrt;
  } catch (error) {
    throw new Error(`ONNX Runtime yüklenemedi. Bu özellik Expo Go yerine development build ister. ${String(error)}`);
  }
}
