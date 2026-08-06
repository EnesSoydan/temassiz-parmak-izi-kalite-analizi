import { Buffer } from 'buffer';
import * as Crypto from 'expo-crypto';
import * as FileSystem from 'expo-file-system/legacy';
import * as SecureStore from 'expo-secure-store';

import { ensureCaptureStorage } from '@/lib/capture-storage';
import {
  FINGERPRINT_POSITIONS,
  isFingerprintPosition,
  type BiometricDatabase,
  type Enrollment,
  type FingerRoi,
  type FingerprintPosition,
  type Person,
} from '@/types/biometrics';

const DATABASE_FILE = `${FileSystem.documentDirectory ?? ''}fingerprint-captures/biometric-database.enc`;
const ENCRYPTION_KEY_NAME = 'fingerprint-biometric-database-aes-key-v1';

let databaseQueue: Promise<void> = Promise.resolve();

export class BiometricDatabaseError extends Error {
  constructor(message: string, cause?: unknown) {
    super(message);
    this.name = 'BiometricDatabaseError';
    if (cause !== undefined) this.cause = cause;
  }
}

export class EnrollmentValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EnrollmentValidationError';
  }
}

export type EnrollmentCaptureSample = {
  sourceCaptureId: string;
  fingerRois: FingerRoi[];
};

function createEmptyDatabase(): BiometricDatabase {
  return {
    schemaVersion: 1,
    people: [],
    enrollments: [],
    updatedAt: new Date(0).toISOString(),
  };
}

function createId(prefix: string) {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

async function getEncryptionKey(createIfMissing = true) {
  try {
    const storedKey = await SecureStore.getItemAsync(ENCRYPTION_KEY_NAME);
    if (storedKey) {
      return await Crypto.AESEncryptionKey.import(storedKey, 'base64');
    }

    if (!createIfMissing) {
      throw new BiometricDatabaseError(
        'Biyometrik veritabanı anahtarı bulunamadı.'
      );
    }

    const key = await Crypto.AESEncryptionKey.generate(Crypto.AESKeySize.AES256);
    const encodedKey = await key.encoded('base64');
    await SecureStore.setItemAsync(ENCRYPTION_KEY_NAME, encodedKey);
    return key;
  } catch (error) {
    throw new BiometricDatabaseError('Biyometrik şifreleme anahtarı açılamadı.', error);
  }
}

function parseDatabase(value: string): BiometricDatabase {
  const parsed = JSON.parse(value) as Partial<BiometricDatabase>;
  if (
    parsed.schemaVersion !== 1 ||
    !Array.isArray(parsed.people) ||
    !Array.isArray(parsed.enrollments)
  ) {
    throw new BiometricDatabaseError('Biyometrik veritabanı sürümü desteklenmiyor.');
  }

  return {
    schemaVersion: 1,
    people: parsed.people as Person[],
    enrollments: parsed.enrollments as Enrollment[],
    updatedAt: parsed.updatedAt ?? new Date(0).toISOString(),
  };
}

async function readDatabaseUnlocked(): Promise<BiometricDatabase> {
  await ensureCaptureStorage();
  const info = await FileSystem.getInfoAsync(DATABASE_FILE);
  if (!info.exists) return createEmptyDatabase();

  try {
    const encryptedBase64 = await FileSystem.readAsStringAsync(DATABASE_FILE);
    const key = await getEncryptionKey(false);
    // Expo Crypto'nun combined formatı byte dizisidir. Dosyada Base64 tuttuğumuz
    // için çözmeden önce Base64 metnini tekrar byte dizisine dönüştürürüz.
    const encryptedBytes = Buffer.from(encryptedBase64, 'base64');
    const sealedData = Crypto.AESSealedData.fromCombined(encryptedBytes, {
      ivLength: 12,
      tagLength: 16,
    });
    const plaintextBase64 = await Crypto.aesDecryptAsync(sealedData, key, {
      output: 'base64',
    });
    const plaintext = Buffer.from(plaintextBase64, 'base64').toString('utf8');
    return parseDatabase(plaintext);
  } catch (error) {
    if (error instanceof BiometricDatabaseError) throw error;
    throw new BiometricDatabaseError(
      'Biyometrik veritabanı doğrulanamadı veya okunamadı.',
      error
    );
  }
}

async function writeDatabaseUnlocked(database: BiometricDatabase) {
  await ensureCaptureStorage();

  try {
    const key = await getEncryptionKey();
    const plaintext = JSON.stringify({
      ...database,
      updatedAt: new Date().toISOString(),
    });
    const plaintextBase64 = Buffer.from(plaintext, 'utf8').toString('base64');
    const sealedData = await Crypto.aesEncryptAsync(plaintextBase64, key, {
      nonce: { length: 12 },
      tagLength: 16,
    });
    const encryptedBase64 = await sealedData.combined('base64');
    await FileSystem.writeAsStringAsync(DATABASE_FILE, encryptedBase64);
  } catch (error) {
    throw new BiometricDatabaseError(
      'Biyometrik veritabanı şifrelenip kaydedilemedi.',
      error
    );
  }
}

async function withDatabaseLock<T>(operation: () => Promise<T>) {
  const result = databaseQueue.then(operation, operation);
  databaseQueue = result.then(
    () => undefined,
    () => undefined
  );
  return result;
}

export async function loadBiometricDatabase() {
  return readDatabaseUnlocked();
}

export async function enrollPersonFromFingerRoiSamples({
  displayName,
  samples,
}: {
  displayName: string;
  samples: [EnrollmentCaptureSample, EnrollmentCaptureSample, EnrollmentCaptureSample];
}) {
  const normalizedName = displayName.trim();
  if (!normalizedName) {
    throw new EnrollmentValidationError('Kayıt için bir kullanıcı adı gir.');
  }

  if (samples.length !== 3) {
    throw new EnrollmentValidationError('Kayıt için üç geçerli çekim gerekir.');
  }
  const validatedSamples = samples.map((sample) => ({
    ...sample,
    byPosition: validateEnrollmentFingerRois(sample.fingerRois),
  }));

  return withDatabaseLock(async () => {
    const database = await readDatabaseUnlocked();
    const now = new Date().toISOString();
    const person: Person = {
      id: createId('person'),
      displayName: normalizedName,
      createdAt: now,
      schemaVersion: 1,
    };
    const enrollments: Enrollment[] = validatedSamples.flatMap(
      (sample, sampleIndex) =>
        FINGERPRINT_POSITIONS.map((position) => {
          const fingerRoi = sample.byPosition.get(position)!;
          const quality = fingerRoi.quality!;
          return {
            id: createId('enrollment'),
            personId: person.id,
            fingerPosition: position,
            template: fingerRoi.minutiaeTemplate!,
            qualitySnapshot: {
              overallScore: quality.globalScore,
              biometricStatus: 'sufficient',
              ridgeScore: quality.ridgePeriodicity,
              orientationScore: quality.orientationCoherence,
            },
            sourceCaptureId: sample.sourceCaptureId,
            createdAt: now,
            sampleIndex: sampleIndex as 0 | 1 | 2,
            coordinateFrame: fingerRoi.coordinateFrame,
            templateVersion: fingerRoi.minutiaeTemplate!.version,
          };
        })
    );

    const nextDatabase: BiometricDatabase = {
      schemaVersion: 1,
      people: [...database.people, person],
      enrollments: [...database.enrollments, ...enrollments],
      updatedAt: now,
    };
    await writeDatabaseUnlocked(nextDatabase);
    return person;
  });
}

// Eski tek örnekli çağrıyı derleme uyumluluğu için korur; yeni kayıt akışı üç örnek ister.
export async function enrollPersonFromFingerRois({
  displayName,
  sourceCaptureId,
  fingerRois,
}: {
  displayName: string;
  sourceCaptureId: string;
  fingerRois: FingerRoi[];
}) {
  void displayName;
  void sourceCaptureId;
  void fingerRois;
  throw new EnrollmentValidationError(
    'Bu sürümde kayıt için aynı elden üç ayrı geçerli çekim gerekir.'
  );
}

export function validateEnrollmentFingerRois(fingerRois: FingerRoi[]) {
  const byPosition = new Map<FingerprintPosition, FingerRoi>();
  for (const fingerRoi of fingerRois) {
    if (!isFingerprintPosition(fingerRoi.className)) continue;
    if (byPosition.has(fingerRoi.className)) {
      throw new EnrollmentValidationError('Aynı parmak birden fazla tespit edildi.');
    }
    byPosition.set(fingerRoi.className, fingerRoi);
  }

  const missingPositions = FINGERPRINT_POSITIONS.filter(
    (position) => !byPosition.has(position)
  );
  if (missingPositions.length > 0) {
    throw new EnrollmentValidationError(
      `Dört parmağın tamamı bulunamadı: ${missingPositions.join(', ')}`
    );
  }

  const invalidPosition = FINGERPRINT_POSITIONS.find((position) => {
    const fingerRoi = byPosition.get(position);
    return (
      !fingerRoi?.minutiaeTemplate ||
      fingerRoi.quality?.biometricStatus !== 'sufficient' ||
      fingerRoi.minutiaeTemplate.version !== 'minutiae-v2' ||
      fingerRoi.coordinateFrame !== 'homography-canonical'
    );
  });
  if (invalidPosition) {
    throw new EnrollmentValidationError(
      `${invalidPosition} parmak için yeterli biyometrik şablon oluşmadı.`
    );
  }

  return byPosition;
}

export async function deletePerson(personId: string) {
  return withDatabaseLock(async () => {
    const database = await readDatabaseUnlocked();
    const nextDatabase: BiometricDatabase = {
      ...database,
      people: database.people.filter((person) => person.id !== personId),
      enrollments: database.enrollments.filter(
        (enrollment) => enrollment.personId !== personId
      ),
      updatedAt: new Date().toISOString(),
    };
    await writeDatabaseUnlocked(nextDatabase);
    return nextDatabase;
  });
}

export async function clearBiometricDatabase() {
  return withDatabaseLock(async () => {
    await FileSystem.deleteAsync(DATABASE_FILE, { idempotent: true });
    await SecureStore.deleteItemAsync(ENCRYPTION_KEY_NAME);
  });
}
